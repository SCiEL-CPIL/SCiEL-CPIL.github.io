# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import deque

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Generic greedy for small n
    return greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    N = q ** n  # 729

    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor lists (words within Hamming distance <= radius)
    # For radius=1, each word has 1 + n*(q-1) = 1 + 6*2 = 13 neighbors
    neighbor_indices = np.zeros((N, 13), dtype=np.int32)
    for i, w in enumerate(all_words):
        neighbors = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neighbors.append(word_to_idx[tuple(new_w)])
        neighbor_indices[i] = neighbors

    # Initial code: Tetracode x 3^2 (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts: how many codewords cover each word
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        counts[neighbor_indices[idx]] += 1
        
    # Active set - use a set for O(1) membership test
    active_set = set(code_indices)
    active_list = list(code_indices)
    
    def can_remove(idx):
        """Check if word idx can be removed (all its neighbors have count > 1)"""
        return np.all(counts[neighbor_indices[idx]] > 1)
    
    def remove_word(idx):
        """Remove a word from the code and update counts"""
        counts[neighbor_indices[idx]] -= 1
        active_set.discard(idx)
        # O(n) removal from list, but list is small (~73-81 elements)
        if idx in active_list:
            active_list.remove(idx)
    
    def add_word(idx):
        """Add a word to the code and update counts"""
        counts[neighbor_indices[idx]] += 1
        active_set.add(idx)
        if idx not in active_list:
            active_list.append(idx)
    
    def try_remove_all_redundant(max_passes=10):
        """Iteratively remove all redundant words"""
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            removed_any = False
            # Shuffle to break symmetry
            shuffled = active_list.copy()
            random.shuffle(shuffled)
            for idx in shuffled:
                if idx not in active_set:
                    continue
                if can_remove(idx):
                    remove_word(idx)
                    removed_any = True
            if not removed_any:
                break
        return len(active_list)
    
    # Initial peeling
    try_remove_all_redundant()
    
    best_size = len(active_list)
    best_active_set = set(active_set)
    best_counts = counts.copy()
    
    # Phase 2: Iterated Local Search
    # Strategy: 
    # 1. Try to remove one random word -> if it creates a hole, fill it with best word
    # 2. Then try to remove redundant words
    # 3. If we end up with smaller code, accept
    # 4. If stuck, perturb by adding a random word and removing others
    
    no_improvement_count = 0
    max_no_improvement = 200
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            break
            
        # Strategy A: Try to swap - remove a random word, add the best replacement
        # Find a word that, if removed, leaves exactly one uncovered word (or few)
        # Then add the word that covers that uncovered word best
        
        # First, try pure removal: can we remove any word?
        # This is already done by try_remove_all_redundant, so we need a different approach
        
        # Strategy B: Add a random word, then aggressively remove
        random_idx = random.randint(0, N - 1)
        
        if random_idx in active_set:
            continue
            
        # Save state before modification
        old_counts = counts.copy()
        old_active_set = set(active_set)
        old_active_list = list(active_list)
        
        # Add the random word
        add_word(random_idx)
        
        # Now try to remove redundant words - be aggressive with multiple passes
        removals = 0
        for _ in range(5):  # up to 5 passes
            if time.time() > deadline:
                break
            pass_removed = False
            shuffled = active_list.copy()
            random.shuffle(shuffled)
            for idx in shuffled:
                if idx not in active_set:
                    continue
                if can_remove(idx):
                    remove_word(idx)
                    removals += 1
                    pass_removed = True
            if not pass_removed:
                break
        
        # Check if we improved
        current_size = len(active_list)
        if current_size < best_size:
            best_size = current_size
            best_active_set = set(active_set)
            best_counts = counts.copy()
            no_improvement_count = 0
        elif current_size <= len(old_active_list):
            # No improvement but didn't get worse - keep the new state for further exploration
            # This allows gradual drift
            no_improvement_count += 1
        else:
            # Revert if we got worse
            counts = old_counts
            active_set = old_active_set
            active_list = old_active_list
            no_improvement_count += 1
        
        # Strategy C: If stuck, try a more aggressive perturbation
        if no_improvement_count % 20 == 0 and no_improvement_count > 0:
            # Add 2-3 random words, then try to reduce
            old_counts = counts.copy()
            old_active_set = set(active_set)
            old_active_list = list(active_list)
            
            for _ in range(2):
                if time.time() > deadline:
                    break
                random_idx = random.randint(0, N - 1)
                if random_idx not in active_set:
                    add_word(random_idx)
            
            # Aggressive removal
            for _ in range(8):
                if time.time() > deadline:
                    break
                pass_removed = False
                shuffled = active_list.copy()
                random.shuffle(shuffled)
                for idx in shuffled:
                    if idx not in active_set:
                        continue
                    if can_remove(idx):
                        remove_word(idx)
                        pass_removed = True
                if not pass_removed:
                    break
            
            current_size = len(active_list)
            if current_size < best_size:
                best_size = current_size
                best_active_set = set(active_set)
                best_counts = counts.copy()
                no_improvement_count = 0
            else:
                # Revert
                counts = old_counts
                active_set = old_active_set
                active_list = old_active_list
    
    # Restore best state
    active_set = best_active_set
    counts = best_counts
    active_list = list(active_set)
    
    # Final cleanup - try to remove any remaining redundant words
    # Multiple passes with different random orderings
    for _ in range(20):
        if time.time() > deadline:
            break
        removed_any = False
        shuffled = active_list.copy()
        random.shuffle(shuffled)
        for idx in shuffled:
            if idx not in active_set:
                continue
            if can_remove(idx):
                remove_word(idx)
                removed_any = True
        if not removed_any:
            break
    
    # Convert back to list of lists
    return [list(all_words[idx]) for idx in active_list]

def greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    if n > 8:
        if q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n >= 4:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                return [[0]*n]
        return [[0]*n]
        
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    def get_neighbors(c):
        neighbors = {c}
        for i in range(n):
            orig = c[i]
            for val in range(q):
                if val == orig: continue
                lst = list(c)
                lst[i] = val
                neighbors.add(tuple(lst))
        return neighbors

    while uncovered and time.time() < deadline:
        best_word = None
        best_count = -1
        
        for w in all_words:
            if w in code: continue
            cnt = 0
            for x in get_neighbors(w):
                if x in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
                
        if best_word is None or best_count == 0:
            break
            
        code.append(list(best_word))
        uncovered.difference_update(get_neighbors(best_word))
        
    return code
# EVOLVE-BLOCK-END
