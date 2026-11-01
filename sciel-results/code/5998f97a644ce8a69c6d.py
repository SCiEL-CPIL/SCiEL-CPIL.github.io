# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

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
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    N = q ** n  # 729

    # Precompute neighbors for each word index as a numpy array
    neighbor_lists = []
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbor_lists.append(neigh)
    
    # Convert to numpy arrays for faster operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbor_lists]
    
    # Initial code: Direct Sum (Tetracode x 3^2) -> 81 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts using numpy
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        counts[neighbor_arrays[idx]] += 1
        
    active_set = set(code_indices)
    active_list = list(code_indices)
    
    def can_remove_vectorized(idx, counts_arr):
        """Check if word idx can be removed (all neighbors have count > 1)"""
        return np.all(counts_arr[neighbor_arrays[idx]] > 1)
    
    def try_remove_all_redundant(counts_arr, active_set_local):
        """Iteratively remove all redundant words until none remain"""
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            items = list(active_set_local)
            random.shuffle(items)
            for idx in items:
                if idx not in active_set_local:
                    continue
                if can_remove_vectorized(idx, counts_arr):
                    counts_arr[neighbor_arrays[idx]] -= 1
                    active_set_local.remove(idx)
                    removed_any = True

    try_remove_all_redundant(counts, active_set)
    
    rng = random.Random(42)
    
    def get_uncovered_mask(counts_arr):
        """Return indices of words with count 0"""
        return np.where(counts_arr == 0)[0]
    
    def fast_repair(active_set_local, counts_local):
        """Add codewords to cover all words with count 0 using numpy vectorization."""
        while True:
            if time.time() > deadline:
                return
            
            uncovered = get_uncovered_mask(counts_local)
            if len(uncovered) == 0:
                break
            
            # Create boolean mask for uncovered words
            uncovered_mask = np.zeros(N, dtype=bool)
            uncovered_mask[uncovered] = True
            
            # For each candidate not in active set, compute gain
            # We'll iterate all N candidates
            best_idx = -1
            best_gain = 0
            
            for i in range(N):
                if i in active_set_local:
                    continue
                # Count how many uncovered words are in neighbors[i]
                gain = np.sum(uncovered_mask[neighbor_arrays[i]])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            # Add best_idx
            active_set_local.add(best_idx)
            counts_local[neighbor_arrays[best_idx]] += 1
            
    def fast_reduce(active_set_local, counts_local):
        """Remove redundant codewords."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            items = list(active_set_local)
            random.shuffle(items)
            for idx in items:
                if idx not in active_set_local:
                    continue
                if can_remove_vectorized(idx, counts_local):
                    counts_local[neighbor_arrays[idx]] -= 1
                    active_set_local.remove(idx)
                    changed = True

    # Main ILS Loop
    current_set = set(active_set)
    current_counts = counts.copy()
    
    best_size = len(current_set)
    best_set = set(current_set)
    best_counts = current_counts.copy()
    
    no_improvement = 0
    max_no_improvement = 200
    
    # Precompute a list of all indices for random sampling
    all_indices = list(range(N))
    
    while time.time() < deadline - 3:
        if no_improvement > max_no_improvement:
            break
            
        # Perturbation: Choose between remove-k and swap
        if rng.random() < 0.5:
            # Remove k random codewords
            k = rng.randint(2, 4)
            if len(current_set) < k:
                k = max(1, len(current_set) // 2)
            to_remove = rng.sample(list(current_set), k)
            for idx in to_remove:
                current_set.remove(idx)
                current_counts[neighbor_arrays[idx]] -= 1
        else:
            # Swap: Remove one, add one random
            if len(current_set) > 1:
                to_remove = rng.choice(list(current_set))
                current_set.remove(to_remove)
                current_counts[neighbor_arrays[to_remove]] -= 1
                
                # Add a random codeword not in current set
                # Sample from all_indices until we find one not in current_set
                attempts = 0
                while attempts < 100:
                    candidate = rng.choice(all_indices)
                    if candidate not in current_set:
                        current_set.add(candidate)
                        current_counts[neighbor_arrays[candidate]] += 1
                        break
                    attempts += 1
        
        # Repair
        fast_repair(current_set, current_counts)
        
        # Reduce
        fast_reduce(current_set, current_counts)
        
        # Evaluate
        current_size = len(current_set)
        
        if current_size < best_size:
            best_size = current_size
            best_set = set(current_set)
            best_counts = current_counts.copy()
            no_improvement = 0
        else:
            no_improvement += 1
        
        # Occasionally reset to best to avoid drift
        if no_improvement % 50 == 0 and no_improvement > 0:
            current_set = set(best_set)
            current_counts = best_counts.copy()

    # Restore best state
    active_set = best_set
    active_list = list(active_set)
    
    # Final cleanup
    final_counts = np.zeros(N, dtype=np.int32)
    for idx in active_list:
        final_counts[neighbor_arrays[idx]] += 1
        
    final_set = set(active_list)
    fast_reduce(final_set, final_counts)
    
    return [list(idx_to_word[idx]) for idx in final_set]

def greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    # Placeholder for generic greedy, used as fallback
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
        
        # Greedy choice: pick word covering most uncovered
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
