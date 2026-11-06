import numpy as np
import time
from itertools import combinations

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(ap_idx)
    
    # Convert to numpy arrays for speed
    pos_aps_arr = [np.array(pa, dtype=np.int32) if pa else np.array([], dtype=np.int32) for pa in pos_aps]

    def count_bad(colors):
        """Count number of 4-APs with < 3 distinct colors."""
        if len(colors) != N:
            return N * N
        ap_colors = colors[aps_arr]
        s = np.sort(ap_colors, axis=1)
        bad = (s[:, 0] == s[:, 3]) | \
              (s[:, 0] == s[:, 2]) | \
              (s[:, 1] == s[:, 3]) | \
              ((s[:, 0] == s[:, 1]) & (s[:, 2] == s[:, 3]) & (s[:, 1] != s[:, 2]))
        return int(np.sum(bad))

    def local_search(colors, max_iters=500000):
        """Greedy local search to minimize bad APs."""
        colors = np.array(colors, dtype=np.int32)
        cost = count_bad(colors)
        
        if cost == 0:
            return colors, 0
        
        # Precompute bad status
        def get_bad_mask(colors):
            ap_colors = colors[aps_arr]
            s = np.sort(ap_colors, axis=1)
            bad = (s[:, 0] == s[:, 3]) | \
                  (s[:, 0] == s[:, 2]) | \
                  (s[:, 1] == s[:, 3]) | \
                  ((s[:, 0] == s[:, 1]) & (s[:, 2] == s[:, 3]) & (s[:, 1] != s[:, 2]))
            return bad
        
        bad_mask = get_bad_mask(colors)
        improved = True
        while improved and cost > 0 and time.time() < deadline:
            improved = False
            for i in range(N):
                old_color = colors[i]
                for new_color in range(4):  # Try 0,1,2,3
                    if new_color == old_color:
                        continue
                    colors[i] = new_color
                    new_cost = count_bad(colors)
                    if new_cost < cost:
                        cost = new_cost
                        bad_mask = get_bad_mask(colors)
                        improved = True
                        if cost == 0:
                            return colors, 0
                    else:
                        colors[i] = old_color
                colors[i] = old_color
        
        return colors, cost

    def greedy_construct(num_colors, time_limit=5.0):
        """Greedy construction: assign each position the smallest color that minimizes new violations."""
        end_time = time.time() + time_limit
        colors = np.full(N, -1, dtype=np.int32)
        
        for pos in range(N):
            if time.time() > end_time:
                break
            best_color = -1
            best_cost = float('inf')
            
            for c in range(num_colors):
                colors[pos] = c
                # Only check APs that include this position
                affected_aps = pos_aps_arr[pos]
                if len(affected_aps) == 0:
                    colors[pos] = c
                    best_cost = 0
                    break
                # Check if all positions in affected APs are assigned
                valid = True
                for ap_idx in affected_aps:
                    ap = aps_arr[ap_idx]
                    for p in ap:
                        if colors[p] == -1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Count bad among affected
                    ap_colors = colors[aps_arr[affected_aps]]
                    s = np.sort(ap_colors, axis=1)
                    bad = (s[:, 0] == s[:, 3]) | \
                          (s[:, 0] == s[:, 2]) | \
                          (s[:, 1] == s[:, 3]) | \
                          ((s[:, 0] == s[:, 1]) & (s[:, 2] == s[:, 3]) & (s[:, 1] != s[:, 2]))
                    cost_c = int(np.sum(bad))
                else:
                    cost_c = float('inf')
                
                if cost_c < best_cost:
                    best_cost = cost_c
                    best_color = c
            
            if best_color >= 0:
                colors[pos] = best_color
            else:
                colors[pos] = 0
        
        # Fill any unassigned
        for i in range(N):
            if colors[i] == -1:
                colors[i] = 0
        
        return colors

    # Strategy: Try 3 colors first with multiple restarts, then 4 colors
    best_result = None
    best_cost = float('inf')
    
    # Phase 1: Try 3-coloring with local search
    num_restarts_3 = 8
    for r in range(num_restarts_3):
        if time.time() > deadline - 10:
            break
        np.random.seed(42 + r)
        init = np.random.randint(0, 3, N)
        result, cost = local_search(init, max_iters=200000)
        if cost < best_cost:
            best_cost = cost
            best_result = result
        if cost == 0:
            return result.tolist()
    
    # Phase 2: Try 4-coloring if 3 failed
    if best_cost > 0:
        num_restarts_4 = 6
        for r in range(num_restarts_4):
            if time.time() > deadline - 5:
                break
            np.random.seed(1000 + r)
            init = np.random.randint(0, 4, N)
            result, cost = local_search(init, max_iters=200000)
            if cost < best_cost:
                best_cost = cost
                best_result = result
            if cost == 0:
                return result.tolist()
    
    # Phase 3: Greedy construction for 4 colors if still failing
    if best_cost > 0 and time.time() < deadline - 5:
        greedy_res = greedy_construct(4, time_limit=3.0)
        cost = count_bad(greedy_res)
        if cost < best_cost:
            best_cost = cost
            best_result = greedy_res
        if cost == 0:
            return greedy_res.tolist()
        
        # Local search on greedy result
        if cost > 0 and time.time() < deadline - 3:
            result, cost2 = local_search(greedy_res, max_iters=100000)
            if cost2 < best_cost:
                best_cost = cost2
                best_result = result
            if cost2 == 0:
                return result.tolist()
    
    # Fallback: return best found
    if best_result is not None:
        return best_result.tolist()
    else:
        # Absolute fallback: alternating pattern
        return [i % 3 for i in range(N)]
