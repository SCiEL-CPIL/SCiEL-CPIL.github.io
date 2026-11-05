import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate all words as a numpy array
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Build word_to_idx
    word_tuples = [tuple(w) for w in words]
    word_to_idx = {t: i for i, t in enumerate(word_tuples)}
    
    # Build neighbors as a numpy array of shape (N, 1+n*(q-1))
    num_neighbors = 1 + n * (q - 1)  # 13
    neighbors = np.empty((N, num_neighbors), dtype=np.int32)
    
    for i in range(N):
        c = words[i]
        neighbors[i, 0] = i
        idx = 1
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                neighbors[i, idx] = word_to_idx[tuple(new_w)]
                idx += 1
    
    # Symmetry-breaking: Fix the first codeword to a canonical form
    # By symmetry, we can assume the first codeword is all zeros
    # This reduces the search space significantly
    
    # Greedy initialization with symmetry constraint
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    # First, add the all-zeros word (canonical form due to symmetry)
    zero_idx = 0
    selected.append(zero_idx)
    uncovered[neighbors[zero_idx, :]] = False
    
    while uncovered.any():
        # Count how many uncovered neighbors each candidate covers
        counts = np.sum(uncovered[neighbors], axis=1)
        best_idx = int(np.argmax(counts))
        
        if counts[best_idx] == 0:
            break
            
        selected.append(best_idx)
        uncovered[neighbors[best_idx, :]] = False
        
        if time.time() > deadline:
            break
    
    # Maintain coverage counts
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        coverage_count[neighbors[j, :]] += 1
    
    # Helper function to check if a codeword is redundant
    def is_redundant(j, cov):
        return np.all(cov[neighbors[j, :]] > 1)
    
    # Helper function to try to reduce a set
    def try_reduce(selected_set, cov, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(selected_set)):
                j = selected_set[idx]
                if is_redundant(j, cov):
                    cov[neighbors[j, :]] -= 1
                    selected_set.pop(idx)
                    changed = True
                    break
        return selected_set

    # Local Search with improved swap strategy
    best_selected = list(selected)
    best_cov = coverage_count.copy()
    
    # First, try to reduce the initial greedy solution
    best_selected = try_reduce(best_selected, best_cov, deadline)
    
    random.seed(42)
    
    current_size = len(best_selected)
    selected_set = set(best_selected)
    
    # Track best solution ever found
    global_best = list(best_selected)
    global_best_size = current_size
    
    # Precompute for faster swap evaluation
    # For each word, precompute its neighbor set as a set for O(1) membership
    # But we'll use numpy for efficiency
    
    # Improved local search with multiple restarts and better move selection
    restart_count = 0
    max_restarts = 5
    
    while time.time() < deadline - 2.0:
        # 1. Try to remove redundant codewords
        new_selected = try_reduce(best_selected, best_cov, deadline)
        if len(new_selected) < current_size:
            current_size = len(new_selected)
            best_selected = new_selected
            selected_set = set(best_selected)
            
            if current_size < global_best_size:
                global_best_size = current_size
                global_best = list(best_selected)
            continue
        
        if len(best_selected) == 0:
            break
        
        # 2. Try a swap: remove one codeword, add another
        # Choose the codeword to remove based on redundancy score
        # Words that cover the most "weakly covered" positions are better candidates to keep
        
        # Calculate redundancy score for each selected word
        # A word is more "redundant" if its neighbors are all covered multiple times
        remove_candidates = []
        for i_word in best_selected:
            # Count how many of its neighbors have coverage exactly 1 (critical)
            critical_count = np.sum(best_cov[neighbors[i_word, :]] == 1)
            # Prefer removing words with fewer critical neighbors
            remove_candidates.append((critical_count, i_word))
        
        # Sort by critical count (ascending) - remove least critical first
        remove_candidates.sort()
        
        # Try removing the least critical word
        if remove_candidates:
            _, i_word = remove_candidates[0]
            
            # Try to find a replacement
            # Remove i_word temporarily
            best_cov[neighbors[i_word, :]] -= 1
            
            # Find uncovered positions
            uncovered_mask = best_cov < 1
            
            if not np.any(uncovered_mask):
                # No uncovered positions, so we can just remove i_word
                # This means i_word was redundant
                best_selected.remove(i_word)
                selected_set.remove(i_word)
                current_size -= 1
                
                if current_size < global_best_size:
                    global_best_size = current_size
                    global_best = list(best_selected)
                continue
            
            # Find candidates that cover the most uncovered positions
            # Only consider words not in selected_set
            uncovered_indices = np.where(uncovered_mask)[0]
            
            # Prioritize words that are neighbors of uncovered positions (they can cover them)
            candidate_pool = set()
            for u_idx in uncovered_indices[:30]:  # Limit to 30 for speed
                # Add neighbors of this uncovered position as candidates
                for nb in neighbors[u_idx, :]:
                    if nb not in selected_set:
                        candidate_pool.add(nb)
            
            # Add some random candidates for exploration
            for _ in range(10):
                cand = random.randint(0, N - 1)
                if cand not in selected_set:
                    candidate_pool.add(cand)
            
            # Score candidates by how many uncovered positions they cover
            best_j = -1
            best_score = -1
            for j_word in candidate_pool:
                covered = np.sum(best_cov[neighbors[j_word, :]] == 0)
                if covered > best_score:
                    best_score = covered
                    best_j = j_word
            
            if best_j >= 0 and best_score > 0:
                # Perform the swap
                best_cov[neighbors[best_j, :]] += 1
                i_idx = best_selected.index(i_word)
                best_selected[i_idx] = best_j
                selected_set.remove(i_word)
                selected_set.add(best_j)
                
                # Try to reduce
                new_selected = try_reduce(best_selected, best_cov, deadline)
                if len(new_selected) < current_size:
                    current_size = len(new_selected)
                    best_selected = new_selected
                    selected_set = set(best_selected)
                    
                    if current_size < global_best_size:
                        global_best_size = current_size
                        global_best = list(best_selected)
                    continue
            else:
                # Revert
                best_cov[neighbors[i_word, :]] += 1
        
        # 3. Random swap for exploration
        i_idx = random.randint(0, len(best_selected) - 1)
        i_word = best_selected[i_idx]
        j_word = random.randint(0, N - 1)
        if j_word in selected_set:
            continue
        
        # Attempt swap i -> j
        old_cov = best_cov.copy()
        best_cov[neighbors[i_word, :]] -= 1
        
        valid = np.all(best_cov >= 1)
        
        if not valid:
            best_cov = old_cov
            continue
        
        best_cov[neighbors[j_word, :]] += 1
        
        temp_selected = list(best_selected)
        temp_selected[i_idx] = j_word
        
        new_selected = try_reduce(temp_selected, best_cov, deadline)
        
        if len(new_selected) < current_size:
            best_selected = new_selected
            selected_set = set(best_selected)
            current_size = len(new_selected)
            
            if current_size < global_best_size:
                global_best_size = current_size
                global_best = list(best_selected)
            continue
        else:
            # Accept with probability 0.3 for random walk
            if random.random() < 0.3:
                best_selected = new_selected
                selected_set = set(best_selected)
            else:
                best_cov[neighbors[j_word, :]] -= 1
                best_cov[neighbors[i_word, :]] += 1
        
        # Check for restart
        if current_size >= global_best_size and random.random() < 0.01:
            restart_count += 1
            if restart_count >= max_restarts:
                break
            # Restart from a different initial solution
            uncovered = np.ones(N, dtype=bool)
            selected = []
            
            # Add a random word first
            first_idx = random.randint(0, N - 1)
            selected.append(first_idx)
            uncovered[neighbors[first_idx, :]] = False
            
            while uncovered.any() and time.time() < deadline - 5.0:
                counts = np.sum(uncovered[neighbors], axis=1)
                best_idx = int(np.argmax(counts))
                if counts[best_idx] == 0:
                    break
                selected.append(best_idx)
                uncovered[neighbors[best_idx, :]] = False
            
            best_cov = np.zeros(N, dtype=np.int32)
            for j in selected:
                best_cov[neighbors[j, :]] += 1
            
            best_selected = try_reduce(list(selected), best_cov, deadline)
            current_size = len(best_selected)
            selected_set = set(best_selected)
            restart_count = 0

    # Final polish: try to remove any remaining redundant codewords
    final_cov = np.zeros(N, dtype=np.int32)
    for j in global_best:
        final_cov[neighbors[j, :]] += 1
    
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        for idx in range(len(global_best)):
            j = global_best[idx]
            if np.all(final_cov[neighbors[j, :]] > 1):
                final_cov[neighbors[j, :]] -= 1
                global_best.pop(idx)
                changed = True
                break
    
    codewords = [words[i].tolist() for i in global_best]
    return codewords
