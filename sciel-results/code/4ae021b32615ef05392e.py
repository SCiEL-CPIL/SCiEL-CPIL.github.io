import numpy as np
import time

def construct():
    N = 128
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Precompute all 4-term APs (1-indexed positions a, a+d, a+2d, a+3d with a+3d <= N)
    # 0-indexed: i, i+d, i+2d, i+3d with i+3d < N
    ap_list = []
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 < N:
                ap_list.append((a, a + d, a + 2 * d, i3))
    
    num_aps = len(ap_list)
    ap_arr = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    
    # Map each index to the APs it participates in, with position multiplier
    # For key = c0*K^3 + c1*K^2 + c2*K + c3
    # index_to_details[idx] = list of (ap_idx, multiplier)
    index_to_details = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        for pos in range(4):
            idx = ap_arr[ap_idx, pos]
            mult = (3 - pos)  # exponent: pos 0 -> K^3, pos 1 -> K^2, pos 2 -> K^1, pos 3 -> K^0
            index_to_details[idx].append((ap_idx, mult))
    
    def check_validity(colors):
        """Check if coloring is valid. colors is 0-indexed array of length N."""
        for ap in ap_list:
            c = tuple(colors[i] for i in ap)
            if len(set(c)) < 3:
                return False
        return True
    
    def count_violations(colors, K):
        """Count number of APs with < 3 distinct colors."""
        # Build bad lookup
        K4 = K ** 4
        bad_lookup = np.zeros(K4, dtype=bool)
        for c0 in range(K):
            for c1 in range(K):
                for c2 in range(K):
                    for c3 in range(K):
                        key = c0 * K**3 + c1 * K**2 + c2 * K + c3
                        if len({c0, c1, c2, c3}) < 3:
                            bad_lookup[key] = True
        
        c0 = colors[ap_arr[:, 0]]
        c1 = colors[ap_arr[:, 1]]
        c2 = colors[ap_arr[:, 2]]
        c3 = colors[ap_arr[:, 3]]
        keys = (c0 * K**3 + c1 * K**2 + c2 * K + c3).astype(np.int32)
        return np.sum(bad_lookup[keys])
    
    def try_sa(K, time_limit, seed=None):
        """Simulated annealing to find valid coloring with K colors."""
        if seed is not None:
            np.random.seed(seed)
        
        start_time = time.time()
        
        # Precompute bad lookup
        K4 = K ** 4
        bad_lookup = np.zeros(K4, dtype=bool)
        for c0 in range(K):
            for c1 in range(K):
                for c2 in range(K):
                    for c3 in range(K):
                        key = c0 * K**3 + c1 * K**2 + c2 * K + c3
                        if len({c0, c1, c2, c3}) < 3:
                            bad_lookup[key] = True
        
        best_solution = None
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit or time.time() > deadline:
                break
            
            # Random initialization
            colors = np.random.randint(0, K, N, dtype=np.int32)
            
            # Compute initial keys and violations
            c0 = colors[ap_arr[:, 0]]
            c1 = colors[ap_arr[:, 1]]
            c2 = colors[ap_arr[:, 2]]
            c3 = colors[ap_arr[:, 3]]
            keys = (c0 * K**3 + c1 * K**2 + c2 * K + c3).astype(np.int32)
            violations = bad_lookup[keys]
            current_penalty = int(np.sum(violations))
            
            if current_penalty == 0:
                return colors.tolist()
            
            # Simulated annealing
            T = 2.0
            cooling = 0.9995
            min_T = 0.001
            
            while T > min_T:
                if time.time() - start_time > time_limit or time.time() > deadline:
                    break
                
                # Pick random index
                i = np.random.randint(0, N)
                old_color = colors[i]
                new_color = np.random.randint(0, K)
                if new_color == old_color:
                    T *= cooling
                    continue
                
                # Get affected APs
                details = index_to_details[i]
                
                old_violations = np.array([violations[ap_idx] for ap_idx, _ in details])
                old_keys = np.array([keys[ap_idx] for ap_idx, _ in details])
                
                # Apply new color
                colors[i] = new_color
                
                new_violations = []
                new_keys = []
                for ap_idx, mult in details:
                    old_key = keys[ap_idx]
                    new_key = old_key - old_color * (K ** mult) + new_color * (K ** mult)
                    new_keys.append(new_key)
                    new_violations.append(bad_lookup[new_key])
                    keys[ap_idx] = new_key
                    violations[ap_idx] = new_violations[-1]
                
                old_pen = int(np.sum(old_violations))
                new_pen = int(np.sum(new_violations))
                delta = new_pen - old_pen
                current_penalty += delta
                
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    pass  # accept
                else:
                    # revert
                    colors[i] = old_color
                    for k, v, (ap_idx, _) in zip(old_keys, old_violations, details):
                        keys[ap_idx] = k
                        violations[ap_idx] = v
                    current_penalty -= delta
                
                T *= cooling
                
                if current_penalty == 0:
                    return colors.tolist()
        
        return None
    
    # Try with 4 colors first
    for K in [4, 5, 6]:
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time() - 5.0  # leave 5s margin
        if remaining_time < 2.0:
            break
        
        # Try multiple seeds
        best = None
        num_attempts = min(10, int(remaining_time / 5.0))
        for attempt in range(num_attempts):
            if time.time() > deadline - 5.0:
                break
            sol = try_sa(K, time_limit=min(10.0, deadline - time.time() - 5.0), seed=42 + attempt)
            if sol is not None:
                best = sol
                break
        
        if best is not None:
            # Verify
            if check_validity(np.array(best)):
                return best
    
    # Fallback: greedy construction
    # Use a simple greedy that picks the smallest valid color
    colors = []
    for x in range(1, N + 1):
        found = False
        for color in range(128):
            valid = True
            # Check all 4-APs ending at x (x is the 4th term)
            # x = a + 3d, so a = x - 3d, need a >= 1
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                b = a + d
                c = a + 2 * d
                # Colors at positions a-1, b-1, c-1 in 0-indexed
                if a - 1 < len(colors) and b - 1 < len(colors) and c - 1 < len(colors):
                    existing = {colors[a-1], colors[b-1], colors[c-1]}
                    if len(existing | {color}) < 3:
                        valid = False
                        break
            # Also check 4-APs where x is not the last term
            # x = a (1st term): a+d, a+2d, a+3d must exist
            # x = a+d (2nd term): a, a+2d, a+3d
            # x = a+2d (3rd term): a, a+d, a+3d
            # These involve future positions, so we can't check them in forward greedy.
            # The standard greedy only checks completed APs.
            
            if valid:
                colors.append(color)
                found = True
                break
        
        if not found:
            # Should not happen with 128 colors
            colors.append(0)
    
    # Verify fallback
    if check_validity(np.array(colors)):
        return colors
    
    # Last resort: return a trivially valid but high-color solution
    # Each position gets its own color
    return list(range(N))
