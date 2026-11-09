import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 58s margin (90s limit)
    
    n = 128
    # Precompute all 4-term APs
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    num_aps = len(aps_list)
    aps = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Precompute: for each AP and each position, the other 3 indices
    other_indices = []
    pos_in_ap = [[-1] * n for _ in range(num_aps)]
    for i, ap in enumerate(aps_list):
        ap_others = []
        for pos_idx in range(4):
            others = tuple(ap[j] for j in range(4) if j != pos_idx)
            ap_others.append(others)
            pos_in_ap[i][ap[pos_idx]] = pos_idx
        other_indices.append(ap_others)

    def count_bad_colors_vec(colors):
        """Count APs with <= 2 distinct colors using numpy."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        # Bad if set size <= 2
        # Equivalent: NOT (at least 3 distinct)
        # At least 3 distinct means: not all same, and not exactly 2 values
        # Faster: count distinct via comparisons
        # A 4-tuple has <=2 distinct iff:
        # (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2)
        #   or (c0==c1 and c0==c2) or (c0==c1 and c0==c3) or (c0==c2 and c0==c3)
        #   or (c1==c2 and c1==c3) or (c1==c2 and c1==c0) or (c1==c3 and c1==c0)
        #   or (c2==c3 and c2==c0) or (c2==c3 and c2==c1) or (c3==c0 and c3==c1)
        # Simpler: use the is_bad logic vectorized
        # Bad patterns for 4 values with <=2 distinct:
        # All same: c0==c1==c2==c3
        # Exactly 2: various patterns
        # Let's just compute: number of distinct <= 2
        # distinct_count = 4 - (number of equality pairs that are "redundant")
        # Actually simplest vectorized: 
        bad = (
            ((c0 == c1) & (c2 == c3)) |
            ((c0 == c2) & (c1 == c3)) |
            ((c0 == c3) & (c1 == c2)) |
            ((c0 == c1) & (c0 == c2)) |
            ((c0 == c1) & (c0 == c3)) |
            ((c0 == c2) & (c0 == c3)) |
            ((c1 == c2) & (c1 == c3)) |
            ((c1 == c2) & (c1 == c0)) |
            ((c1 == c3) & (c1 == c0)) |
            ((c2 == c3) & (c2 == c0)) |
            ((c2 == c3) & (c2 == c1)) |
            ((c3 == c0) & (c3 == c1)) |
            ((c3 == c2) & (c3 == c1))
        )
        return int(np.sum(bad))

    def is_bad_ap(c0, c1, c2, c3):
        if c0 == c1:
            if c2 == c3:
                return 1
            if c2 == c0 or c3 == c0:
                return 1
            return 0
        elif c0 == c2:
            if c1 == c3:
                return 1
            if c1 == c0 or c3 == c0:
                return 1
            return 0
        elif c0 == c3:
            if c1 == c2:
                return 1
            if c1 == c0 or c2 == c0:
                return 1
            return 0
        else:
            if c1 == c2:
                if c3 == c1:
                    return 1
                return 0
            elif c1 == c3:
                if c2 == c1:
                    return 1
                return 0
            elif c2 == c3:
                if c1 == c2:
                    return 1
                return 0
            else:
                return 0

    def compute_delta(colors, i, new_color):
        """Compute delta in bad count when changing colors[i] to new_color."""
        old_color = colors[i]
        if new_color == old_color:
            return 0
        delta = 0
        for ap_idx in idx_to_aps[i]:
            pos = pos_in_ap[ap_idx][i]
            others = other_indices[ap_idx][pos]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            old_bad = is_bad_ap(c1, c2, c3, old_color)
            new_bad = is_bad_ap(c1, c2, c3, new_color)
            delta += new_bad - old_bad
        return delta

    best_solution = None
    best_cost = float('inf')

    # Try k=3 first, then 4, 5
    for k in [3, 4, 5]:
        if time.time() > deadline - 5:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        max_restarts = 200
        restart_count = 0
        
        while time.time() < deadline - 5 and restart_count < max_restarts:
            restart_count += 1
            
            # Random initialization
            colors = np.random.randint(0, k, n, dtype=np.int32)
            current_cost = count_bad_colors_vec(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
            
            # Simulated Annealing with adaptive parameters
            T = 2.0
            min_T = 0.005
            cooling = 0.9995
            no_improve = 0
            max_no_improve = 300
            step = 0
            
            while time.time() < deadline - 5:
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                step += 1
                
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                delta = compute_delta(colors, i, new_color)
                
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                    no_improve = 0
                else:
                    if delta == 0:
                        # Accept neutral moves to escape plateaus
                        if random.random() < 0.5:
                            colors[i] = new_color
                        else:
                            no_improve += 1
                    else:
                        if random.random() < np.exp(-delta / T):
                            colors[i] = new_color
                            current_cost += delta
                        else:
                            no_improve += 1
                            if no_improve > max_no_improve:
                                break
                            
                T *= cooling
                if T < min_T:
                    T = min_T

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            return list(best_k_solution)

    # Fallback: greedy construction
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check 4-term APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    break
                a = x - 1 - 3 * d
                if len({color} | {colors[a], colors[a + d], colors[a + 2 * d]}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check 3-term APs where x is not the last (to ensure we don't create 2-color 3-subsets that could extend)
            # Actually the constraint is only on 4-term APs, so we only need to check 4-term APs
            # But we also need to ensure no 4-term AP with x in non-last position has <=2 colors
            # Let's check all 4-term APs containing x
            for d in range(1, 42):
                # x could be at position 0,1,2,3 in the AP
                # pos 0: x, x+d, x+2d, x+3d
                if x + 3 * d < n:
                    c = [color, colors[x + d], colors[x + 2 * d], colors[x + 3 * d]]
                    if len(set(c)) < 3:
                        valid = False
                        break
                # pos 1: x-d, x, x+d, x+2d
                if x - d >= 0 and x + 2 * d < n:
                    c = [colors[x - d], color, colors[x + d], colors[x + 2 * d]]
                    if len(set(c)) < 3:
                        valid = False
                        break
                # pos 2: x-2d, x-d, x, x+d
                if x - 2 * d >= 0 and x + d < n:
                    c = [colors[x - 2 * d], colors[x - d], color, colors[x + d]]
                    if len(set(c)) < 3:
                        valid = False
                        break
                # pos 3: x-3d, x-2d, x-d, x
                if x - 3 * d >= 0:
                    c = [colors[x - 3 * d], colors[x - 2 * d], colors[x - d], color]
                    if len(set(c)) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            # Should not happen, but fallback
            colors.append(0)
    return colors
