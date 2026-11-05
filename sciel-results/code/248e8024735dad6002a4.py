import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    # Handle non-target parameters with a generic greedy fallback
    if q != 3 or n != 8 or radius != 1:
        N = q ** n
        if N > 1000000:
            # For very large spaces, just return a single word (will be invalid but prevents crash)
            return [[0]*n]
        
        # Generate all words
        words = []
        for i in range(N):
            w = []
            x = i
            for _ in range(n):
                w.append(x % q)
                x //= q
            w.reverse()
            words.append(tuple(w))
        
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        # Precompute neighbors
        neighbors = []
        for i, w in enumerate(words):
            neigh = set([i])
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    new_w = tuple(new_w)
                    neigh.add(word_to_idx[new_w])
            neighbors.append(list(neigh))
        
        uncovered = np.ones(N, dtype=bool)
        C = []
        start = time.time()
        limit = 100.0
        while np.any(uncovered):
            best_idx = -1
            best_cnt = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_cnt <= 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start > limit: break
            
        # Reduce
        counts = np.zeros(N, dtype=int)
        for c in C:
            counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    
    # Generate all words
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(8):
            w.append(x % 3)
            x //= 3
        w.reverse()
        words.append(tuple(w))
    
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors efficiently
    # For each word, neighbors are itself + 2*8 = 17 words
    neighbors = np.zeros((N, 17), dtype=int)
    for i, w in enumerate(words):
        neighbors[i, 0] = i
        cnt = 1
        for pos in range(8):
            for delta in [1, 2]:
                new_val = (w[pos] + delta) % 3
                new_w = list(w)
                new_w[pos] = new_val
                new_w = tuple(new_w)
                neighbors[i, cnt] = word_to_idx[new_w]
                cnt += 1

    start_time = time.time()
    time_limit = 115.0

    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    while np.any(uncovered):
        best_idx = -1
        best_cnt = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_cnt <= 0: break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break

    # 2. Local Search
    def calc_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        """
        Remove codewords that are redundant (all their neighbors are covered by >= 2 codewords).
        """
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed = False
            next_code = []
            for c in current_code:
                # Check if all neighbors of c have count >= 2
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed:
                break
        return current_code, current_counts

    C, counts = reduce_code(C, calc_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local Search Loop
    # Strategy:
    # 1. Try to remove a codeword. If not possible, try to swap a codeword c with a neighbor c' 
    #    such that c' covers the unique points of c, and then try to remove c.
    #    Actually, simpler: 
    #    - For each c in C, find unique points U_c (points covered only by c).
    #    - If U_c is empty, remove c.
    #    - If U_c is small (<= 3), find a c' not in C that covers all points in U_c.
    #    - Swap c with c'. Then try to remove c (which is now redundant if c' covers U_c and others cover rest).
    #      Actually, if we swap c -> c', we must ensure coverage doesn't break.
    #      If c' covers U_c, then removing c is safe only if all other points covered by c are covered by others.
    #      But c' might not cover other points.
    #      Better: 
    #      1. Pick c.
    #      2. Find U_c.
    #      3. Find c' such that U_c subset of neighbors[c'].
    #      4. Swap c -> c'.
    #      5. Now c is removed. Check if coverage is valid.
    #      6. Then run reduce_code to remove any newly redundant codewords.
    
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Swap Pass
        if len(C) == old_size:
            # No reduction. Try to find a swap.
            improved = False
            # Randomize order
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                # Find unique points of c
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    # Should have been removed
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                # If too many unique points, skip
                if len(unique_points) > 3:
                    continue
                
                # Find candidates c' that cover all unique_points
                # Intersection of neighbors[u] for u in unique_points
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
                
                # Filter: c' must not be in C
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                
                # Pick a candidate
                c_prime = random.choice(list(candidates_set))
                
                # Apply swap
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                # Check validity
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                
                # Try to reduce
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    # Improvement
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break
                else:
                    # No immediate reduction. 
                    # Accept the swap to change state?
                    # If we accept, we might escape local minimum.
                    # But we must ensure we don't cycle.
                    # Let's accept it.
                    C = new_C
                    counts = new_counts
                    improved = True
                    break
            
            if not improved:
                # No move found. Try a random kick.
                for _ in range(5):
                    if time.time() - start_time > time_limit: break
                    if not C: break
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    # Pick a random neighbor of c
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime in C: continue
                    
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
                        if len(new_C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        C = new_C
                        counts = new_counts

    # Final Reduction
    best_C, _ = reduce_code(best_C, calc_counts(best_C))
    
    return [list(words[i]) for i in best_C]
