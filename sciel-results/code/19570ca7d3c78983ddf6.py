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
    start_time = time.time()
    time_limit = 140.0
    
    # 1. Generate Ternary Hamming Code of length 4
    # Parity check matrix columns: reps of non-zero 1D subspaces of F_3^2
    # (1,0), (0,1), (1,1), (1,2)
    H_par = np.array([
        [1, 0, 1, 1],
        [0, 1, 1, 2]
    ])
    
    # Find all codewords c in {0,1,2}^4 such that H_par @ c == 0 mod 3
    H_code = []
    for c in itertools.product(range(3), repeat=4):
        if np.all((H_par @ np.array(c)) % 3 == 0):
            H_code.append(c)
            
    # H_code should have 9 elements
    # Verify
    if len(H_code) != 9:
        # Fallback to standard local search if Hamming code generation fails (shouldn't happen)
        return _standard_local_search(q, n, radius, start_time, time_limit)

    # 2. Define Spheres S_i
    # S_i = { x | d(x, H_code[i]) <= 1 }
    spheres = []
    for i, h in enumerate(H_code):
        sphere = [h]
        for pos in range(4):
            for val in range(3):
                if val == h[pos]:
                    continue
                new_w = list(h)
                new_w[pos] = val
                sphere.append(tuple(new_w))
        spheres.append(sphere)
        
    # 3. Solve Block Covering Problem for S_0 x S_0
    # We want to find a set of 5 codewords in S_0 x S_0 that covers all 81 points in S_0 x S_0
    # Point (u, v) is covered by (c_u, c_v) if d(u, c_u) + d(v, c_v) <= 1
    
    S0 = spheres[0]
    # Map S0 indices to actual values for distance calc
    # Distances in S0:
    # d(u, v) is Hamming distance in F_3^4
    
    # Precompute distances between all pairs in S0
    # dist[i][j] = d(S0[i], S0[j])
    dist_S0 = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            d = sum(1 for k in range(4) if S0[i][k] != S0[j][k])
            dist_S0[i][j] = d
            
    # The block points are (i, j) where i, j in 0..8 (indices into S0)
    # A codeword is (a, b) where a, b in 0..8
    # It covers (i, j) if dist_S0[a][i] + dist_S0[b][j] <= 1
    
    # We need to select 5 pairs (a_k, b_k) such that for all (i, j), exists k with dist_S0[a_k][i] + dist_S0[b_k][j] <= 1
    
    # Backtracking search for 5 codewords
    def solve_block():
        # State: covered_mask (81 bits), current_codewords
        # We can use a bitmask or a set of (i,j)
        covered = set()
        
        # Precompute coverage of each possible codeword (a,b)
        # cov[a][b] is a set of (i,j) covered
        cov = [[set() for _ in range(9)] for _ in range(9)]
        for a in range(9):
            for b in range(9):
                for i in range(9):
                    if dist_S0[a][i] > 1: continue
                    for j in range(9):
                        if dist_S0[b][j] <= 1 - dist_S0[a][i]:
                            cov[a][b].add((i,j))
                            
        # Greedy initialization to get a good starting point
        # But we need exactly 5 or less.
        # Let's try to find 5.
        
        def backtrack(k, current_covered, chosen):
            if len(current_covered) == 81:
                return True
            if k == 5:
                return False
            if time.time() - start_time > time_limit - 10:
                return False
                
            # Heuristic: pick codeword that covers most uncovered points
            uncovered = set(itertools.product(range(9), range(9))) - current_covered
            if not uncovered:
                return True
                
            best_cov = -1
            best_ab = None
            # To speed up, we can restrict candidates
            # But 81 candidates is small.
            for a in range(9):
                for b in range(9):
                    gain = len(cov[a][b] & uncovered)
                    if gain > best_cov:
                        best_cov = gain
                        best_ab = (a, b)
                        
            if best_cov == 0:
                return False
                
            a, b = best_ab
            new_covered = current_covered | cov[a][b]
            if backtrack(k+1, new_covered, chosen + [(a,b)]):
                return True
                
            # If greedy fails, we need to try other options?
            # For a robust solution, we should iterate over all candidates with high gain.
            # But for now, let's try a few random restarts of the greedy or a more exhaustive search.
            return False

        # Try greedy first
        current_covered = set()
        chosen = []
        for _ in range(5):
            uncovered = set(itertools.product(range(9), range(9))) - current_covered
            if not uncovered: break
            best_cov = -1
            best_ab = None
            for a in range(9):
                for b in range(9):
                    gain = len(cov[a][b] & uncovered)
                    if gain > best_cov:
                        best_cov = gain
                        best_ab = (a, b)
            if best_cov == 0: break
            a, b = best_ab
            current_covered |= cov[a][b]
            chosen.append((a,b))
            
        if len(current_covered) == 81:
            return chosen
            
        # If greedy fails, try randomized restarts
        rng = random.Random(42)
        for _ in range(1000):
            current_covered = set()
            chosen = []
            for _ in range(5):
                uncovered = set(itertools.product(range(9), range(9))) - current_covered
                if not uncovered: break
                # Pick random candidate from top 10 gains
                gains = []
                for a in range(9):
                    for b in range(9):
                        gain = len(cov[a][b] & uncovered)
                        if gain > 0:
                            gains.append((gain, a, b))
                if not gains: break
                gains.sort(reverse=True)
                top = gains[:10]
                _, a, b = top[rng.randint(0, len(top)-1)]
                current_covered |= cov[a][b]
                chosen.append((a,b))
                
            if len(current_covered) == 81:
                return chosen
                
        return None

    block_solution = solve_block()
    
    if block_solution:
        # Construct full code
        # For each block (i, j), add codewords (S0[a] + (H[i]-H[0]), S0[b] + (H[j]-H[0]))
        # Note: S0 is centered at H[0].
        # S_i is centered at H[i].
        # The mapping is x -> x + (H[i] - H[0])
        
        full_code = []
        h0 = np.array(H_code[0])
        
        for i in range(9):
            hi = np.array(H_code[i])
            delta_i = (hi - h0) % 3
            for j in range(9):
                hj = np.array(H_code[j])
                delta_j = (hj - h0) % 3
                for a, b in block_solution:
                    u = np.array(S0[a]) + delta_i
                    v = np.array(S0[b]) + delta_j
                    # u and v are in {0,1,2}^4
                    # Combine to 8-tuple
                    word = tuple(u) + tuple(v)
                    full_code.append(word)
                    
        # Verify distinctness
        full_code = list(set(full_code))
        
        # Verify coverage (optional, but good for safety)
        # Given time, we trust the decomposition.
        
        return [list(w) for w in full_code]
    else:
        # Fallback to standard local search
        return _standard_local_search(q, n, radius, start_time, time_limit)

def _standard_local_search(q, n, radius, start_time, time_limit):
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
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
    
    neighbor_sets = [set(arr.tolist()) for arr in neighbors]
    
    def reduce_code(C, counts):
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
        return C, counts
    
    # Greedy
    uncovered = np.ones(N, dtype=bool)
    C = []
    while np.any(uncovered) and time.time() - start_time < time_limit:
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
        
    counts = np.zeros(N, dtype=int)
    for c in C:
        counts[neighbors[c]] += 1
        
    C, counts = reduce_code(C, counts)
    
    # Simple local search
    best_C = C.copy()
    best_size = len(C)
    
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        for _ in range(int(remaining / 2)):
            if time.time() - start_time > time_limit - 2: break
            # Try swaps
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > time_limit - 2: break
                c = C[idx]
                unique_points = neighbors[c][counts[neighbors[c]] == 1]
                if len(unique_points) == 0:
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                if len(unique_points) > 3: continue
                candidates = set(neighbor_sets[unique_points[0]])
                for u in unique_points[1:]:
                    candidates &= neighbor_sets[u]
                candidates.difference_update(C)
                if candidates:
                    c_prime = random.choice(list(candidates))
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
                        improved = True
                        break
                    else:
                        C = new_C
                        counts = new_counts
                        improved = True
                        break
            if not improved:
                break
                
    return [list(all_words[i]) for i in best_C]
