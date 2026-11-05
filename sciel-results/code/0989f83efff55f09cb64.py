import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute conflict masks using Python big ints as bitsets
    # conflict_mask[v] has bit j set if (v*j) % (v+j) == 0 and j != v
    # Bit i (0-indexed) represents vertex (i+1)
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask

    all_mask = (1 << N) - 1  # bits 0..255 all set

    def bit_count(x):
        return x.bit_count()

    def get_vertex(x):
        """Get the lowest set bit position (0-indexed)."""
        return (x & (-x)).bit_length() - 1

    def greedy_min_degree(available, randomize=False, k=1, score_func=None):
        """
        Greedy independent set using min-degree heuristic.
        available: bitmask of available vertices
        score_func: optional function (v, rem) -> float to rank vertices
        Returns: (solution_list, solution_mask)
        """
        sol = []
        sol_mask = 0
        rem = available
        
        while rem:
            # Find vertex with minimum degree in induced subgraph
            if score_func is not None:
                # Use custom scoring
                best_v = -1
                best_score = float('-inf')
                temp = rem
                while temp:
                    lsb = temp & (-temp)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    score = score_func(v, rem, deg)
                    if score > best_score:
                        best_score = score
                        best_v = v
                    temp ^= lsb
                v = best_v
            elif randomize and k > 1:
                # Find top-k min-degree vertices and pick randomly
                temp = rem
                best_deg = float('inf')
                # First pass: find min degree
                t = rem
                while t:
                    lsb = t & (-t)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg < best_deg:
                        best_deg = deg
                    t ^= lsb
                
                # Collect vertices with degree within best_deg + small tolerance
                candidates = []
                t = rem
                while t:
                    lsb = t & (-t)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg <= best_deg + 1:  # allow slight tolerance
                        candidates.append(v)
                    t ^= lsb
                
                if not candidates:
                    # fallback: pick lowest
                    lsb = rem & (-rem)
                    v = lsb.bit_length()
                else:
                    kk = min(k, len(candidates))
                    v = candidates[random.randrange(kk)]
            else:
                # Deterministic min-degree
                min_v = -1
                min_deg = float('inf')
                temp = rem
                while temp:
                    lsb = temp & (-temp)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                    temp ^= lsb
                v = min_v
            
            if v < 0:
                break
                
            sol.append(v)
            sol_mask |= (1 << (v - 1))
            
            # Remove v and its neighbors from rem
            rem &= ~(1 << (v - 1))
            rem &= ~conflict_mask[v]
        
        return sol, sol_mask

    # --- Phase 1: Get a good initial solution via multiple greedy runs ---
    best_sol = []
    best_mask = 0
    best_size = 0
    
    # Deterministic greedy
    sol, sol_mask = greedy_min_degree(all_mask, randomize=False)
    if len(sol) > best_size:
        best_size = len(sol)
        best_sol = sol
        best_mask = sol_mask

    # Randomized greedy restarts to find better initial solution
    for it in range(300):
        if time.time() > deadline:
            break
        sol, sol_mask = greedy_min_degree(all_mask, randomize=True, k=5)
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_mask = sol_mask
            # If we've done well, do more
            if best_size >= 130:
                break

    # --- Phase 2: Local improvement - try to add vertices ---
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find all vertices that can be added
        for v in range(1, N + 1):
            if best_mask & (1 << (v - 1)):
                continue
            # Check if v conflicts with any in best_sol
            if conflict_mask[v] & best_mask:
                continue
            # Can add v
            best_mask |= (1 << (v - 1))
            best_size += 1
            improved = True

    # --- Phase 3: Spectral-inspired greedy using eigenvector-like scores ---
    # Build complement adjacency as NumPy matrix for fast power iteration
    non_conflict_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        non_conflict_mask[v] = all_mask ^ conflict_mask[v] ^ (1 << (v - 1))
    
    # Build adjacency matrix for complement graph using NumPy
    adj = np.zeros((N, N), dtype=np.float64)
    for v in range(1, N + 1):
        mask = non_conflict_mask[v]
        t = mask
        while t:
            lsb = t & (-t)
            j_idx = lsb.bit_length() - 1
            adj[v - 1, j_idx] = 1.0
            t ^= lsb
    
    # Power iteration using NumPy for speed
    def power_iteration_np(adj_mat, n_iter=30):
        """Find principal eigenvector using NumPy power iteration."""
        dim = adj_mat.shape[0]
        vec = np.random.rand(dim)
        vec /= np.linalg.norm(vec)
        for _ in range(n_iter):
            new_vec = adj_mat @ vec
            norm = np.linalg.norm(new_vec)
            if norm < 1e-12:
                break
            vec = new_vec / norm
        return vec
    
    # Compute spectral scores
    spectral_vec = power_iteration_np(adj, n_iter=30)
    
    # Use spectral scores to guide greedy: prefer vertices with high spectral score
    # and low degree in remaining set
    def spectral_score_func(v, rem, deg):
        return spectral_vec[v - 1] * 1000.0 - deg
    
    # Try spectral-guided greedy multiple times
    for it in range(200):
        if time.time() > deadline:
            break
        sol, sol_mask = greedy_min_degree(all_mask, score_func=spectral_score_func)
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_mask = sol_mask
            # Polish
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if best_mask & bit:
                    continue
                if conflict_mask[v] & best_mask:
                    continue
                best_mask |= bit
                best_size += 1

    # --- Phase 4: 1-for-k swap improvements ---
    def try_improve_swap(best_mask, best_size, k_out=1, k_in=2):
        """Try to find k_out vertices to remove and k_in vertices to add (k_in > k_out)."""
        sol_list = [i+1 for i in range(N) if best_mask & (1 << i)]
        if len(sol_list) < k_out:
            return None, best_size
        
        for _ in range(100):
            if time.time() > deadline:
                break
            # Pick random vertices to remove
            outs = random.sample(sol_list, k_out)
            new_mask = best_mask
            for v in outs:
                new_mask &= ~(1 << (v - 1))
            
            # Find vertices that can be added given new_mask
            addable = []
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if new_mask & bit:
                    continue
                if conflict_mask[v] & new_mask:
                    continue
                addable.append(v)
            
            if len(addable) < k_in:
                continue
            
            # Try adding k_in random addable vertices that don't conflict with each other
            candidates = random.sample(addable, min(k_in + 5, len(addable)))
            chosen = []
            for c in candidates:
                if len(chosen) == k_in:
                    break
                # Check c doesn't conflict with already chosen
                ok = True
                for d in chosen:
                    if conflict_mask[c] & (1 << (d - 1)):
                        ok = False
                        break
                if ok:
                    chosen.append(c)
            
            if len(chosen) == k_in:
                new_mask2 = new_mask
                for c in chosen:
                    new_mask2 |= (1 << (c - 1))
                return new_mask2, best_size - k_out + k_in
        
        return None, best_size

    # Try several rounds of swap improvements (1-for-2, 1-for-3, 2-for-3)
    for round_num in range(80):
        if time.time() > deadline:
            break
        improved = False
        # Try 1-for-2
        new_mask, new_size = try_improve_swap(best_mask, best_size, 1, 2)
        if new_mask is not None:
            best_mask = new_mask
            best_size = new_size
            improved = True
        # Try 1-for-3
        if not improved:
            new_mask, new_size = try_improve_swap(best_mask, best_size, 1, 3)
            if new_mask is not None:
                best_mask = new_mask
                best_size = new_size
                improved = True
        # Try 2-for-3
        if not improved:
            new_mask, new_size = try_improve_swap(best_mask, best_size, 2, 3)
            if new_mask is not None:
                best_mask = new_mask
                best_size = new_size
                improved = True
        if not improved:
            # Also try adding single vertices
            added = False
            for v in range(1, N + 1):
                if time.time() > deadline:
                    break
                bit = 1 << (v - 1)
                if best_mask & bit:
                    continue
                if conflict_mask[v] & best_mask:
                    continue
                best_mask |= bit
                best_size += 1
                added = True
            if not added:
                break

    # --- Phase 5: More randomized min-degree greedy with different randomizations ---
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Randomized min-degree greedy with varying k
        k = random.choice([3, 5, 7, 10])
        sol, sol_mask = greedy_min_degree(all_mask, randomize=True, k=k)
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_mask = sol_mask
            best_sol = sol
            # Polish: try to add more
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if best_mask & bit:
                    continue
                if conflict_mask[v] & best_mask:
                    continue
                best_mask |= bit
                best_size += 1

    # Final polish: greedy add (multiple passes to catch cascading additions)
    for _ in range(3):
        if time.time() > deadline:
            break
        changed = False
        for v in range(1, N + 1):
            if time.time() > deadline:
                break
            bit = 1 << (v - 1)
            if best_mask & bit:
                continue
            if conflict_mask[v] & best_mask:
                continue
            best_mask |= bit
            best_size += 1
            changed = True
        if not changed:
            break

    # Reconstruct solution list from mask
    final_sol = [i + 1 for i in range(N) if best_mask & (1 << i)]
    
    # Ensure nonempty
    if not final_sol:
        final_sol = [1]
    
    return final_sol
