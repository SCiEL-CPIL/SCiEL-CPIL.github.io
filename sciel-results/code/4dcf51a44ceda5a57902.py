import itertools
import random
import time
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Standard recursive construction
    def standard_superpermutation(k):
        word = [1]
        for m in range(2, k + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return word

    # Get standard S6
    s6_std = standard_superpermutation(6)
    
    # Extract order of perms from s6_std
    def get_perm_order(word, m):
        order = []
        seen = set()
        for i in range(len(word) - m + 1):
            p = tuple(word[i:i+m])
            if p not in seen:
                seen.add(p)
                order.append(p)
        return order

    order6 = get_perm_order(s6_std, 6)
    N = len(order6)  # 720
    
    # Precompute overlap matrix
    def calc_overlap(p1, p2):
        for k in range(5, -1, -1):
            if p1[-k:] == p2[:k]:
                return k
        return 0

    # Build overlap matrix using numpy for speed
    import numpy as np
    
    # Store permutations as arrays for faster comparison
    perm_arrays = [np.array(p, dtype=np.int8) for p in order6]
    
    # Precompute overlap matrix
    ov_mat = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        pi = perm_arrays[i]
        for j in range(N):
            if i != j:
                pj = perm_arrays[j]
                # Check overlaps from 5 down to 0
                ov = 0
                for k in range(5, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if np.array_equal(pi[-k:], pj[:k]):
                        ov = k
                        break
                ov_mat[i][j] = ov
    
    # Initial sum of overlaps
    current_order = list(order6)
    current_overlap_sum = 0
    for i in range(N - 1):
        current_overlap_sum += ov_mat[i][i+1]
        
    best_order = current_order.copy()
    best_overlap_sum = current_overlap_sum
    
    # SA Parameters
    T = 10.0
    T_min = 0.05
    alpha = 0.9995
    steps = 200000
    
    current_T = T
    step = 0
    N_minus_1 = N - 1
    
    while step < steps and time.time() < deadline:
        # Move: Swap two random permutations
        i, j = random.sample(range(N), 2)
        if i > j:
            i, j = j, i
        
        # Calculate delta overlap for swap of positions i and j
        # Edges removed: (i-1,i), (i,i+1), (j-1,j), (j,j+1)
        # Edges added: (i-1,j), (j,i), (i,j-1), (j-1,i+1)
        # Special cases: if i==0, no left edge for i; if j==N-1, no right edge for j
        # If j == i+1, edges are different
        
        delta = 0
        
        if j == i + 1:
            # Adjacent swap
            # Old edges: (i-1,i), (i,i+1), (i+1,i+2)
            # New edges: (i-1,i+1), (i+1,i), (i,i+2)
            if i > 0:
                delta -= ov_mat[i-1][i]
                delta += ov_mat[i-1][i+1]
            delta -= ov_mat[i][i+1]
            delta += ov_mat[i+1][i]
            if i + 2 < N:
                delta -= ov_mat[i+1][i+2]
                delta += ov_mat[i][i+2]
        else:
            # Non-adjacent swap (j > i+1)
            # Old edges: (i-1,i), (i,i+1), (j-1,j), (j,j+1)
            # New edges: (i-1,j), (j,i), (i,j-1), (j-1,i+1)
            if i > 0:
                delta -= ov_mat[i-1][i]
                delta += ov_mat[i-1][j]
            else:
                # No left neighbor for i
                pass
            
            delta -= ov_mat[i][i+1]
            delta += ov_mat[j][i]
            
            delta -= ov_mat[j-1][j]
            delta += ov_mat[i][j-1]
            
            if j + 1 < N:
                delta -= ov_mat[j][j+1]
                delta += ov_mat[j-1][i+1]
            else:
                # No right neighbor for j
                pass
        
        if delta > 0 or random.random() < math.exp(delta / current_T):
            current_order[i], current_order[j] = current_order[j], current_order[i]
            current_overlap_sum += delta
            if current_overlap_sum > best_overlap_sum:
                best_overlap_sum = current_overlap_sum
                best_order = current_order.copy()
                
        current_T *= alpha
        step += 1
    
    # Construct S6 from best_order
    s6_best = list(best_order[0])
    for i in range(1, len(best_order)):
        p1 = best_order[i-1]
        p2 = best_order[i]
        ov = calc_overlap(p1, p2)
        s6_best.extend(p2[ov:])
    
    # Expand to S7
    word = s6_best
    m = 7
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        overlap = 0
        for o in range(min(len(new), len(block)), -1, -1):
            if new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
        
    s7_final = new
    
    # Compare with standard S7
    s7_std = standard_superpermutation(7)
    
    if len(s7_final) < len(s7_std):
        return "".join(map(str, s7_final))
    else:
        return "".join(map(str, s7_std))
