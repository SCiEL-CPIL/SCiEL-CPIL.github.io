import time
import random
import math
import numpy as np

def objective(A):
    """Compute max(|A+A|, |A*A|) using NumPy vectorization."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    A_col = A[:, np.newaxis]
    A_row = A[np.newaxis, :]
    sums = A_col + A_row
    prods = A_col * A_row
    s = len(np.unique(sums))
    p = len(np.unique(prods))
    return max(s, p)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 20
    MAX_VAL = 10000

    # Search in exponent space: A = {2^i * 3^j : (i,j) in S}
    # Products: 2^(i+k) * 3^(j+l) for (i,j),(k,l) in S
    # So |A*A| = |{(i+k, j+l) : (i,j),(k,l) in S}| (the sumset of S with itself)
    # Sums: 2^i*3^j + 2^k*3^l — harder to predict, need to compute
    
    # Precompute all valid (i,j) pairs where 2^i * 3^j <= MAX_VAL
    valid_exponents = []
    for i in range(0, 15):  # 2^14 = 16384 > 10000, so i up to 13
        for j in range(0, 15):  # 3^14 is huge, but 2^i*3^j <= 10000
            val = (2 ** i) * (3 ** j)
            if val <= MAX_VAL:
                valid_exponents.append((i, j, val))
    
    # Also include some non-lattice numbers for diversity
    # But primary search is in lattice space
    
    def exponent_set_to_values(exp_set):
        """Convert set of (i,j) exponents to list of integer values."""
        return [2**i * 3**j for (i, j) in exp_set]
    
    def exp_sumset_size(exp_set):
        """Compute |S+S| for exponent set S (this gives |A*A| for lattice numbers)."""
        exps = list(exp_set)
        pairs = set()
        for idx, (i1, j1) in enumerate(exps):
            for jdx in range(idx, len(exps)):
                i2, j2 = exps[jdx]
                pairs.add((i1 + i2, j1 + j2))
        return len(pairs)
    
    def objective_from_exponents(exp_set):
        """Compute max(|A+A|, |A*A|) from exponent set."""
        values = exponent_set_to_values(exp_set)
        if len(set(values)) != len(values):
            return 10**9  # invalid
        A = np.array(values, dtype=np.int64)
        n_val = len(A)
        A_col = A[:, np.newaxis]
        A_row = A[np.newaxis, :]
        sums = A_col + A_row
        prods = A_col * A_row
        s = len(np.unique(sums))
        p = len(np.unique(prods))
        return max(s, p)
    
    def local_search_exponents(current_exp_set, max_iters=200, deadline=None):
        """Steepest descent local search in exponent space."""
        current_exp_set = list(current_exp_set)
        current_obj = objective_from_exponents(current_exp_set)
        
        for _ in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            improved = False
            best_neighbor = None
            best_neighbor_obj = current_obj
            
            for i in range(len(current_exp_set)):
                old_i, old_j = current_exp_set[i]
                
                # Try nearby exponent values
                candidates = set()
                for di in [-2, -1, 1, 2]:
                    for dj in [-2, -1, 1, 2]:
                        ni, nj = old_i + di, old_j + dj
                        if ni >= 0 and nj >= 0:
                            val = (2 ** ni) * (3 ** nj)
                            if val <= MAX_VAL:
                                candidates.add((ni, nj))
                
                # Also try some random valid exponents
                for _ in range(3):
                    idx = random.randint(0, len(valid_exponents) - 1)
                    ri, rj, rv = valid_exponents[idx]
                    candidates.add((ri, rj))
                
                for new_exp in candidates:
                    if new_exp == (old_i, old_j):
                        continue
                    neighbor = current_exp_set[:]
                    neighbor[i] = new_exp
                    # Check uniqueness of values
                    vals = [2**a * 3**b for (a, b) in neighbor]
                    if len(set(vals)) != len(vals):
                        continue
                    n_obj = objective_from_exponents(neighbor)
                    if n_obj < best_neighbor_obj:
                        best_neighbor_obj = n_obj
                        best_neighbor = neighbor[:]
            
            if best_neighbor is not None and best_neighbor_obj < current_obj:
                current_exp_set = best_neighbor
                current_obj = best_neighbor_obj
                improved = True
            else:
                break
        
        return current_exp_set, current_obj

    def simulated_annealing_exponents(start_exp_set, max_iters=3000, deadline=None):
        """Simulated annealing in exponent space."""
        current_exp_set = list(start_exp_set)
        current_obj = objective_from_exponents(current_exp_set)
        
        T = 15.0
        T_min = 0.1
        cooling_rate = 0.995
        
        best_local_exp_set = current_exp_set[:]
        best_local_obj = current_obj
        
        for i in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            neighbor = current_exp_set[:]
            move_type = random.random()
            
            if move_type < 0.6:
                # Replace one exponent
                idx = random.randint(0, len(current_exp_set) - 1)
                # Pick a random valid exponent
                ri, rj, rv = valid_exponents[random.randint(0, len(valid_exponents) - 1)]
                neighbor[idx] = (ri, rj)
            elif move_type < 0.8:
                # Perturb one exponent
                idx = random.randint(0, len(current_exp_set) - 1)
                di, dj = current_exp_set[idx]
                new_i = di + random.choice([-2, -1, 1, 2])
                new_j = dj + random.choice([-2, -1, 1, 2])
                if new_i >= 0 and new_j >= 0:
                    val = (2 ** new_i) * (3 ** new_j)
                    if val <= MAX_VAL:
                        neighbor[idx] = (new_i, new_j)
            else:
                # Swap two exponents with new values
                if len(current_exp_set) >= 2:
                    i1, i2 = random.sample(range(len(current_exp_set)), 2)
                    ri1, rj1, _ = valid_exponents[random.randint(0, len(valid_exponents) - 1)]
                    ri2, rj2, _ = valid_exponents[random.randint(0, len(valid_exponents) - 1)]
                    neighbor[i1] = (ri1, rj1)
                    neighbor[i2] = (ri2, rj2)
            
            # Check validity (distinct values)
            vals = [2**a * 3**b for (a, b) in neighbor]
            if len(set(vals)) != len(vals):
                continue
            
            neighbor_obj = objective_from_exponents(neighbor)
            delta_obj = neighbor_obj - current_obj
            
            if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
                current_exp_set = neighbor
                current_obj = neighbor_obj
                
                if current_obj < best_local_obj:
                    best_local_obj = current_obj
                    best_local_exp_set = current_exp_set[:]
            
            T = max(T * cooling_rate, T_min)
        
        # Polish with local search
        final_exp_set, final_obj = local_search_exponents(best_local_exp_set, max_iters=100, deadline=deadline)
        if final_obj < best_local_obj:
            best_local_exp_set = final_exp_set
            best_local_obj = final_obj
        
        return best_local_exp_set, best_local_obj

    # Generate diverse seeds in exponent space
    random.seed(42)
    seeds = []
    
    # 1. Diagonal: (0,0), (1,1), (2,2), ...
    diag = [(i, i) for i in range(n) if (2**i)*(3**i) <= MAX_VAL]
    if len(diag) >= n:
        seeds.append(diag[:n])
    
    # 2. Horizontal line: (0,0), (0,1), (0,2), ...
    horiz = [(0, j) for j in range(n) if (3**j) <= MAX_VAL]
    if len(horiz) >= n:
        seeds.append(horiz[:n])
    
    # 3. Vertical line: (0,0), (1,0), (2,0), ...
    vert = [(i, 0) for i in range(n) if (2**i) <= MAX_VAL]
    if len(vert) >= n:
        seeds.append(vert[:n])
    
    # 4. Mixed: first half diagonal, second half random
    mixed = [(i, i) for i in range(n // 2)]
    while len(mixed) < n:
        ri, rj, rv = valid_exponents[random.randint(0, len(valid_exponents) - 1)]
        if (ri, rj) not in mixed:
            mixed.append((ri, rj))
    seeds.append(mixed)
    
    # 5. Random exponent sets
    for _ in range(8):
        exp_set = []
        attempts = 0
        while len(exp_set) < n and attempts < 200:
            ri, rj, rv = valid_exponents[random.randint(0, len(valid_exponents) - 1)]
            if (ri, rj) not in exp_set:
                exp_set.append((ri, rj))
            attempts += 1
        if len(exp_set) == n:
            seeds.append(exp_set)
    
    # 6. "Compact" 2D cluster: small range of i and j
    for i_base in range(0, 3):
        for j_base in range(0, 3):
            cluster = []
            for di in range(0, 5):
                for dj in range(0, 5):
                    if len(cluster) >= n:
                        break
                    ni, nj = i_base + di, j_base + dj
                    val = (2 ** ni) * (3 ** nj)
                    if val <= MAX_VAL and (ni, nj) not in cluster:
                        cluster.append((ni, nj))
                if len(cluster) >= n:
                    break
            if len(cluster) == n:
                seeds.append(cluster)
    
    # 7. "Staircase": (0,0), (1,1), (2,2), ..., (k,k), then (k+1,0), (k+2,0), ...
    stair = []
    for i in range(n):
        stair.append((i, min(i, 3)))
    # Check validity
    vals = [2**a * 3**b for (a, b) in stair]
    if len(set(vals)) == n and all(v <= MAX_VAL for v in vals):
        seeds.append(stair)
    
    best_found_obj = float('inf')
    best_found_exp_set = None

    # Run SA on each seed
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
        
        if len(seed) != n:
            continue
        # Validate seed
        vals = [2**a * 3**b for (a, b) in seed]
        if len(set(vals)) != n or any(v > MAX_VAL or v < 1 for v in vals):
            continue
        
        sa_exp_set, sa_obj = simulated_annealing_exponents(seed, max_iters=2000, deadline=deadline - 10.0)
        
        if sa_obj < best_found_obj:
            best_found_obj = sa_obj
            best_found_exp_set = sa_exp_set[:]
        
        # Early exit if we find a very good solution
        if best_found_obj <= 100:
            break

    # Final intensive polish
    if best_found_exp_set is not None and time.time() < deadline - 3.0:
        polished_exp_set, polished_obj = local_search_exponents(best_found_exp_set, max_iters=300, deadline=deadline - 2.0)
        if polished_obj < best_found_obj:
            best_found_obj = polished_obj
            best_found_exp_set = polished_exp_set

    # Convert to final integer set
    if best_found_exp_set is None:
        # Fallback: simple AP
        final_set = list(range(1, n + 1))
    else:
        final_set = [2**a * 3**b for (a, b) in best_found_exp_set]
    
    # Verify final set
    if len(set(final_set)) != n or any(x > MAX_VAL or x < 1 for x in final_set):
        # Fallback to scaled AP
        ap = list(range(1, n + 1))
        sc = MAX_VAL // max(ap)
        final_set = [x * sc for x in ap]
    
    return final_set
