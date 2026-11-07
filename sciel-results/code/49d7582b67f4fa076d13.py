import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']
    
    # Build adjacency mask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Seed Solution (Baseline)
    best_bicliques = []
    for u in range(n):
        right_side = [v for a, v in edges if a == u]
        if right_side:
            best_bicliques.append([[u], right_side])
            
    best_count = len(best_bicliques)
    total_edges = len(edges)

    # Precompute popcount for 16-bit chunks for speed
    popcount = [0] * 65536
    for i in range(65536):
        popcount[i] = bin(i).count('1')
    
    def popcount_int(x):
        c = 0
        while x:
            c += popcount[x & 0xFFFF]
            x >>= 16
        return c

    def count_edges(rem_adj):
        # Sum of popcounts divided by 2
        s = 0
        for i in range(n):
            s += popcount_int(rem_adj[i])
        return s // 2

    def try_greedy(restarts=1000):
        nonlocal best_bicliques, best_count
        
        for r in range(restarts):
            if time.time() > deadline:
                return
            
            rem_adj = adj[:]
            current_bicliques = []
            edges_left = total_edges
            
            while edges_left > 0:
                if time.time() > deadline:
                    return
                
                # Pick edge with max degree product to find dense cores
                best_u, best_v = -1, -1
                best_score = -1
                for i in range(n):
                    if not rem_adj[i]:
                        continue
                    deg_i = popcount_int(rem_adj[i])
                    if deg_i == 0:
                        continue
                    m = rem_adj[i]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        deg_j = popcount_int(rem_adj[j])
                        score = deg_i * deg_j
                        if score > best_score:
                            best_score = score
                            best_u = i
                            best_v = j
                        m ^= lsb
                
                if best_u == -1:
                    # Fallback to random if scoring fails or is too slow
                    found_edge = False
                    for _ in range(100):
                        u = random.randint(0, n-1)
                        mask = rem_adj[u]
                        if mask:
                            m = mask
                            candidates = []
                            while m:
                                lsb = m & -m
                                v_cand = lsb.bit_length() - 1
                                candidates.append(v_cand)
                                m ^= lsb
                            v = random.choice(candidates)
                            if rem_adj[u] & (1 << v):
                                best_u = u
                                best_v = v
                                found_edge = True
                                break
                    if not found_edge:
                        for i in range(n):
                            if rem_adj[i]:
                                best_u = i
                                m = rem_adj[i]
                                lsb = m & -m
                                best_v = lsb.bit_length() - 1
                                found_edge = True
                                break
                    if not found_edge:
                        break
                
                u, v = best_u, best_v
                
                # Strategy 1: Maximal Expansion
                L = {u}
                R = {v}
                changed = True
                for _ in range(n):
                    if not changed:
                        break
                    changed = False
                    mask_R = 0
                    for rv in R:
                        mask_R |= (1 << rv)
                    candidates_L = []
                    for w in range(n):
                        if w in L: continue
                        if (rem_adj[w] & mask_R) == mask_R:
                            candidates_L.append(w)
                    if candidates_L:
                        random.shuffle(candidates_L)
                        for w in candidates_L:
                            L.add(w)
                            changed = True
                    mask_L = 0
                    for lu in L:
                        mask_L |= (1 << lu)
                    candidates_R = []
                    for w in range(n):
                        if w in R: continue
                        valid = True
                        for lu in L:
                            if not (rem_adj[lu] & (1 << w)):
                                valid = False
                                break
                        if valid:
                            candidates_R.append(w)
                    if candidates_R:
                        random.shuffle(candidates_R)
                        for w in candidates_R:
                            R.add(w)
                            changed = True

                # Strategy 2: Greedy Expansion (Add best candidate one by one)
                L2 = {u}
                R2 = {v}
                for _ in range(n):
                    mask_R = 0
                    for rv in R2:
                        mask_R |= (1 << rv)
                    best_l_cand = -1
                    best_l_score = -1
                    for w in range(n):
                        if w in L2: continue
                        if (rem_adj[w] & mask_R) == mask_R:
                            # Score: how many new R candidates does w enable?
                            # New R candidates are vertices connected to all in L2 + {w}
                            # Approx: popcount of intersection of adj of L2 and adj of w, minus current R2
                            common = 0
                            for lu in L2:
                                common &= rem_adj[lu]
                            common &= rem_adj[w]
                            common &= ~mask_R
                            score = popcount_int(common)
                            if score > best_l_score:
                                best_l_score = score
                                best_l_cand = w
                    if best_l_cand != -1:
                        L2.add(best_l_cand)
                    else:
                        break # No more L candidates, check R? 
                        # Actually, if we can't add L, we might still add R?
                        # If L is fixed, R candidates are determined.
                        # Let's just stop if no L candidate, as we want to grow both.
                        # Or continue to R expansion?
                        # Let's try to add R if possible.
                        mask_L = 0
                        for lu in L2:
                            mask_L |= (1 << lu)
                        best_r_cand = -1
                        best_r_score = -1
                        for w in range(n):
                            if w in R2: continue
                            valid = True
                            for lu in L2:
                                if not (rem_adj[lu] & (1 << w)):
                                    valid = False
                                    break
                            if valid:
                                # Score: how many new L candidates does w enable?
                                common = 0
                                for rv in R2:
                                    common &= rem_adj[rv]
                                common &= rem_adj[w]
                                common &= ~mask_L
                                score = popcount_int(common)
                                if score > best_r_score:
                                    best_r_score = score
                                    best_r_cand = w
                        if best_r_cand != -1:
                            R2.add(best_r_cand)
                        else:
                            break # Stuck
                            # Actually, this loop structure is a bit mixed.
                            # Let's restart the loop if we added something.
                            # The `for _ in range(n)` handles iterations.
                            # If we added L, we continue to next iteration to check R.
                            # If we added R, we continue to next iteration to check L.
                            # If we added neither, break.
                            # The `break` above exits the for loop.
                            # This is correct.
                            pass

                # Compare areas
                if len(L) * len(R) >= len(L2) * len(R2):
                    L_final, R_final = L, R
                else:
                    L_final, R_final = L2, R2

                current_bicliques.append([list(L_final), list(R_final)])
                
                mask_L = 0
                for lu in L_final:
                    mask_L |= (1 << lu)
                mask_R = 0
                for rv in R_final:
                    mask_R |= (1 << rv)
                
                for lu in L_final:
                    rem_adj[lu] &= ~mask_R
                for rv in R_final:
                    rem_adj[rv] &= ~mask_L
                
                edges_left -= len(L_final) * len(R_final)
                
                if edges_left < 0:
                    break

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = [ [list(l), list(r)] for l, r in current_bicliques ]
                if best_count == 1:
                    return

    try_greedy()

    # Fallback if something went wrong (shouldn't happen)
    if not best_bicliques:
        for u in range(n):
            right_side = [v for a, v in edges if a == u]
            if right_side:
                best_bicliques.append([[u], right_side])

    # Convert to required format
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
        
    return result
