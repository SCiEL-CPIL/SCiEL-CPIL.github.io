import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3, non-residue mod 7
    total_points = p * p * p * p  # 2401
    num_dirs = p * p + 1  # 50 directions

    def point_flat(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute line membership: for each direction d and point x,
    # line_pts[d][x] = array of 49 point indices (including x) on L(d, x)
    # line_other[d][x] = array of 48 point indices (excluding x)
    
    line_other = np.zeros((num_dirs, total_points, p*p - 1), dtype=np.int32)
    
    # Direction 0: v = (0, 1)
    d = 0
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = point_flat(xa, xb, ya, yb)
                    t_idx = 0
                    for ta in range(p):
                        for tb in range(p):
                            if ta == 0 and tb == 0:
                                continue
                            nya = (ya + ta) % p
                            nyb = (yb + tb) % p
                            line_other[d, x_idx, t_idx] = point_flat(xa, xb, nya, nyb)
                            t_idx += 1

    # Directions 1..49: v = (1, m) for m in F_49
    d = 1
    for ma in range(p):
        for mb in range(p):
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            x_idx = point_flat(xa, xb, ya, yb)
                            t_idx = 0
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    nxa = (xa + ta) % p
                                    nxb = (xb + tb) % p
                                    tma = (ta * ma + tb * mb * k) % p
                                    tmb = (ta * mb + tb * ma) % p
                                    nya = (ya + tma) % p
                                    nyb = (yb + tmb) % p
                                    line_other[d, x_idx, t_idx] = point_flat(nxa, nxb, nya, nyb)
                                    t_idx += 1
            d += 1

    # Verification: for a given in_N boolean array, check Nikodym property
    def verify_nikodym(in_N):
        if time.time() - start_time > deadline:
            return False
        all_in = np.all(in_N[line_other], axis=2)  # shape (50, 2401)
        valid_points = np.any(all_in, axis=0)
        return bool(np.all(valid_points))

    # Count violations for a given in_N: number of points that are NOT covered
    def count_violations(in_N):
        all_in = np.all(in_N[line_other], axis=2)  # shape (50, 2401)
        valid_points = np.any(all_in, axis=0)
        return int(np.sum(~valid_points))

    # Start with a known valid construction:
    # Hermitian unital: U = {(xa, xb, ya, yb) : 2*xa + ya^2 - 3*yb^2 = 0 mod 7}
    # Complement S = {points in U where y_a is a square mod 7}
    # N = F_49^2 \ S

    in_S = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        if ya in {0, 1, 2, 4}:
                            in_S[point_flat(xa, xb, ya, yb)] = True

    in_N = ~in_S

    # Verify initial construction
    if not verify_nikodym(in_N):
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    initial_S_size = np.sum(in_S)
    best_in_S = in_S.copy()
    best_S_size = initial_S_size

    # Simulated annealing to maximize |S| (minimize |N|)
    # State: in_S boolean array
    # Move: flip one point (add or remove from S)
    # Energy: -|S| + penalty * violations
    # We want to maximize |S| while keeping violations = 0
    
    np.random.seed(12345)
    
    T = 100.0  # initial temperature
    T_min = 0.01
    cooling = 0.9999
    
    current_in_S = best_in_S.copy()
    current_S_size = best_S_size
    current_violations = 0
    
    # Precompute for efficiency: for each point, precompute which (d, x) pairs it affects
    # Actually, let's just do direct verification but with incremental updates
    
    # For speed, let's try a different approach:
    # 1. Start from valid state
    # 2. Try adding random points to S
    # 3. If valid, keep; if not, try removing some points to restore validity
    
    # Phase 1: Greedy addition with random restarts
    for restart in range(30):
        if time.time() - start_time > deadline - 60:
            break
        
        current_in_S = best_in_S.copy()
        current_S_size = best_S_size
        
        all_point_indices = np.arange(total_points)
        np.random.shuffle(all_point_indices)
        
        for p_idx in all_point_indices:
            if time.time() - start_time > deadline - 60:
                break
            
            if current_in_S[p_idx]:
                continue
            
            current_in_S[p_idx] = True
            current_in_N = ~current_in_S
            
            if verify_nikodym(current_in_N):
                current_S_size += 1
            else:
                current_in_S[p_idx] = False
        
        if current_S_size > best_S_size:
            best_S_size = current_S_size
            best_in_S = current_in_S.copy()
    
    # Phase 2: Simulated annealing with swaps
    # Try to find configurations where we can swap points in S
    current_in_S = best_in_S.copy()
    current_S_size = best_S_size
    
    T = 50.0
    steps_per_temp = 100
    
    while T > T_min and time.time() - start_time < deadline - 30:
        for _ in range(steps_per_temp):
            if time.time() - start_time > deadline - 30:
                break
            
            # Pick a random point to try adding to S
            p_idx = np.random.randint(total_points)
            
            if current_in_S[p_idx]:
                continue
            
            # Try adding p_idx to S
            old_state = current_in_S[p_idx]
            current_in_S[p_idx] = True
            current_in_N = ~current_in_S
            
            violations = count_violations(current_in_N)
            
            if violations == 0:
                # Accept: valid state with larger S
                current_S_size += 1
                if current_S_size > best_S_size:
                    best_S_size = current_S_size
                    best_in_S = current_in_S.copy()
            else:
                # Penalty for violations
                delta_energy = -1 + violations * 10  # -1 for gaining point, +10*violations penalty
                if np.random.rand() < np.exp(-delta_energy / T):
                    # Accept with penalty
                    pass  # Keep the state temporarily
                else:
                    # Reject
                    current_in_S[p_idx] = old_state
        
        T *= cooling
    
    # Phase 3: Final greedy cleanup - try to add any remaining points
    for p_idx in range(total_points):
        if time.time() - start_time > deadline - 10:
            break
        
        if best_in_S[p_idx]:
            continue
        
        test_in_S = best_in_S.copy()
        test_in_S[p_idx] = True
        test_in_N = ~test_in_S
        
        if verify_nikodym(test_in_N):
            best_in_S[p_idx] = True
            best_S_size += 1
    
    # Build output from best_in_S
    in_N_final = ~best_in_S
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if not best_in_S[point_flat(xa, xb, ya, yb)]:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
