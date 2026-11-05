import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays for fast operations
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
    
    # Generate the van Laarhoven 486-word code
    # The code is based on a ternary linear code of length 8
    # Using the construction from van Laarhoven et al. (1989)
    # We'll use a systematic approach: generate all 3^6 = 729 words from a base code,
    # but actually the 486-word code has a specific structure.
    
    # Alternative: Use a greedy approach to find a good starting code
    # But we know 486 is achievable, so let's try to construct it.
    
    # Actually, let's use a different approach: start with a large random set and reduce
    # Or better: use the fact that we can construct a covering code iteratively
    
    # Let's use a fast greedy to get close to 486, then local search
    start_time = time.time()
    time_limit = 115.0
    
    # Fast greedy initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    # Precompute neighbor counts for efficiency
    while np.any(uncovered) and time.time() - start_time < time_limit:
        # Find word covering most uncovered
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
    
    # Initial reduction
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
    
    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts
    
    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Aggressive local search
    # Strategy: 
    # 1. Try to remove codewords
    # 2. Try swaps to enable removals
    # 3. Random kicks to escape local minima
    
    # Precompute for each codeword its unique points
    def find_unique_points(c, counts):
        u_indices = neighbors[c]
        unique_mask = counts[u_indices] == 1
        return u_indices[unique_mask]
    
    # Local search loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Try a random restart or kick
            if time.time() - start_time > time_limit:
                break
            # Random kick
            if not C:
                break
            c_idx = random.randint(0, len(C)-1)
            c = C[c_idx]
            cand = neighbors[c]
            c_prime = cand[random.randint(0, len(cand)-1)]
            if c_prime in C:
                continue
            
            old_C = C.copy()
            old_counts = counts.copy()
            
            C.remove(c)
            C.append(c_prime)
            counts[neighbors[c]] -= 1
            counts[neighbors[c_prime]] += 1
            
            if np.any(counts < 1):
                C = old_C
                counts = old_counts
                continue
            
            new_C, new_counts = reduce_code(C, counts)
            C = new_C
            counts = new_counts
            no_improve_count = 0
            continue
        
        if time.time() - start_time > time_limit:
            break
        
        # 2. Swap Pass - try to find a swap that enables reduction
        if len(C) == old_size:
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                unique_points = find_unique_points(c, counts)
                
                if len(unique_points) == 0:
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 3:
                    continue
                
                # Find candidates that cover all unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                else:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                    for u in unique_points[1:]:
                        candidates_set &= set(neighbors[u].tolist())
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                
                c_prime = random.choice(list(candidates_set))
                
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break
                else:
                    C = new_C
                    counts = new_counts
                    improved = True
                    break
            
            if not improved:
                # Try more aggressive moves
                # 2-for-1 swap: remove 2 codewords, add 1
                if time.time() - start_time < time_limit:
                    improved2 = False
                    # Try removing two codewords and adding one that covers their unique points
                    c1_idx = random.randint(0, len(C)-1)
                    c2_idx = random.randint(0, len(C)-1)
                    if c1_idx == c2_idx:
                        continue
                    c1 = C[c1_idx]
                    c2 = C[c2_idx]
                    
                    u1 = find_unique_points(c1, counts)
                    u2 = find_unique_points(c2, counts)
                    
                    if len(u1) + len(u2) > 4:
                        continue
                    
                    # Find a word that covers all unique points from both
                    all_unique = list(set(u1.tolist()) | set(u2.tolist()))
                    
                    if not all_unique:
                        continue
                    
                    if len(all_unique) == 1:
                        candidates_set = set(neighbors[all_unique[0]].tolist())
                    else:
                        candidates_set = set(neighbors[all_unique[0]].tolist())
                        for u in all_unique[1:]:
                            candidates_set &= set(neighbors[u].tolist())
                            if not candidates_set:
                                break
                    
                    if not candidates_set:
                        continue
                    
                    candidates_set.difference_update(C)
                    
                    if not candidates_set:
                        continue
                    
                    c_prime = random.choice(list(candidates_set))
                    
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c1)
                    C.remove(c2)
                    C.append(c_prime)
                    
                    counts[neighbors[c1]] -= 1
                    counts[neighbors[c2]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue
                    
                    new_C, new_counts = reduce_code(C, counts)
                    
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        improved2 = True
                        break
    
    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
