import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 128
    
    # Precompute all 4-term progressions
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # Map index to list of AP indices it belongs to
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos].append(idx)
            
    # Precompute AP neighbors for fast cost update?
    # Actually, just iterating over ap_indices[pos] is fast enough.
    
    def get_violation(ap_idx, colors):
        a, b, c, d = aps[ap_idx]
        ca = colors[a]
        cb = colors[b]
        cc = colors[c]
        cd = colors[d]
        # Count distinct colors
        # Since colors are small ints, we can use a set or manual check
        # Manual check is faster for 4 elements
        distinct = 1
        if cb != ca: distinct += 1
        if cc != ca and cc != cb: distinct += 1
        if cd != ca and cd != cb and cd != cc: distinct += 1
        if distinct >= 3:
            return 0
        else:
            return 1

    def total_cost(colors):
        cost = 0
        for idx in range(len(aps)):
            cost += get_violation(idx, colors)
        return cost

    def local_search(k, time_limit):
        """
        Attempt to find a valid coloring using at most k colors.
        Returns (valid, colors_array)
        """
        end_time = time.time() + time_limit
        best_cost = float('inf')
        best_colors = None
        
        # Try multiple restarts
        while time.time() < end_time:
            # Initialize random
            colors = np.random.randint(0, k, N)
            cost = total_cost(colors)
            
            if cost == 0:
                return True, colors
                
            current_cost = cost
            current_colors = colors.copy()
            
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            alpha = 0.9999
            no_improve_count = 0
            max_no_improve = 1000
            
            while time.time() < end_time:
                # Pick random position
                pos = np.random.randint(0, N)
                old_color = current_colors[pos]
                
                # Try all other colors
                best_delta = float('inf')
                best_new_color = old_color
                
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    # Calculate delta cost
                    # Only APs containing pos are affected
                    delta = 0
                    for ap_idx in ap_indices[pos]:
                        # Save old state? No, just compute old and new violation
                        # We need to temporarily change or compute both
                        # Since N is small, we can just compute violation for the AP
                        # with old color and new color
                        a, b, c, d = aps[ap_idx]
                        
                        # Get current colors
                        ca = current_colors[a]
                        cb = current_colors[b]
                        cc = current_colors[c]
                        cd = current_colors[d]
                        
                        # Compute old violation
                        distinct_old = 1
                        if cb != ca: distinct_old += 1
                        if cc != ca and cc != cb: distinct_old += 1
                        if cd != ca and cd != cb and cd != cc: distinct_old += 1
                        v_old = 0 if distinct_old >= 3 else 1
                        
                        # Compute new violation
                        # If pos is a, b, c, or d, replace accordingly
                        if a == pos:
                            ca = new_color
                        elif b == pos:
                            cb = new_color
                        elif c == pos:
                            cc = new_color
                        elif d == pos:
                            cd = new_color
                        
                        distinct_new = 1
                        if cb != ca: distinct_new += 1
                        if cc != ca and cc != cb: distinct_new += 1
                        if cd != ca and cd != cb and cd != cc: distinct_new += 1
                        v_new = 0 if distinct_new >= 3 else 1
                        
                        delta += v_new - v_old
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                
                # Accept best move
                if best_delta < 0:
                    current_colors[pos] = best_new_color
                    current_cost += best_delta
                    no_improve_count = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = current_colors.copy()
                        if best_cost == 0:
                            return True, best_colors
                else:
                    # Accept with probability exp(-delta/T)
                    if np.random.rand() < np.exp(-best_delta / T):
                        current_colors[pos] = best_new_color
                        current_cost += best_delta
                    no_improve_count += 1
                    
                T *= alpha
                if T < T_min:
                    T = T_min
                
                if no_improve_count > max_no_improve:
                    break

        if best_cost == 0:
            return True, best_colors
        return False, best_colors

    # Try k from 3 upwards
    # Allocate time for each k
    # If we find a solution with k, we can stop.
    # We want to minimize k.
    
    # Estimate time per k
    # If we spend 5s on k=3 and fail, try k=4 with 5s, etc.
    # Total budget 50s.
    
    # However, if k=3 is impossible, we waste time.
    # But we don't know.
    
    # Let's try a few random restarts for small k first.
    
    # Fallback: Greedy
    def greedy_fallback():
        colors = []
        for x in range(1, N + 1):
            # Try colors 0..127
            found = False
            for color in range(128):
                valid = True
                # Check all 4-APs ending at x?
                # Actually, we need to check all APs containing x?
                # No, if we assign x, we only violate APs where x is the last element?
                # No, an AP (a, b, c, d) is violated if the set of colors has size < 3.
                # When assigning x, we can only affect APs containing x.
                # But the greedy in the seed checks:
                # all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                # This checks APs where x is the 4th term: (x-3d, x-2d, x-d, x).
                # AND
                # all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
                # This checks APs where x is the 3rd term: (x-2d, x-d, x, x+d)? No, x+d is not assigned yet.
                # Wait, the seed's logic is:
                # 1. For all d such that x-3d >= 1: The AP (x-3d, x-2d, x-d, x) must have >= 3 colors.
                #    This ensures that adding color to x doesn't make this AP have < 3 colors.
                # 2. For all d such that x-2d >= 1: The AP (x-2d, x-d, x, x+d) is not fully assigned yet.
                #    But the condition `any(colors[x-1-j*d]!=color for j in (1,2))` means:
                #    color != c_{x-d} OR color != c_{x-2d}.
                #    This ensures that c_x is not equal to BOTH c_{x-d} and c_{x-2d}.
                #    Why? If c_x == c_{x-d} == c_{x-2d}, then the prefix (x-2d, x-d, x) is monochromatic.
                #    Then for the full AP (x-2d, x-d, x, x+d), if x+d takes the same color, it's 1 color.
                #    If x+d takes a different color, it's 2 colors.
                #    In either case, it's < 3 colors.
                #    So preventing the prefix of length 3 from being monochromatic is necessary.
                
                # My local search handles all constraints globally, so it's safer.
                # But for fallback, let's use a similar logic or just random valid completion?
                # Actually, if local search fails, we can just return a valid coloring from a simple heuristic.
                # The seed's greedy is valid. We can just use it.
                
                # Let's implement the seed's greedy to ensure validity if SA fails.
                pass
            
            # This is getting complicated to replicate seed's greedy exactly.
            # Let's just use the seed's greedy code as a fallback function.
            pass

    # Implementing seed's greedy as fallback
    def seed_greedy():
        colors = []
        for x in range(1, N + 1):
            for color in range(128):
                # Check condition 1: APs ending at x
                cond1 = True
                for d in range(1, (x - 1) // 3 + 1):
                    idx1 = x - 1 - 3 * d
                    idx2 = x - 1 - 2 * d
                    idx3 = x - 1 - d
                    # colors list is 0-indexed for positions 1..x-1
                    c1 = colors[idx1]
                    c2 = colors[idx2]
                    c3 = colors[idx3]
                    s = {color, c1, c2, c3}
                    if len(s) < 3:
                        cond1 = False
                        break
                if not cond1:
                    continue
                    
                # Check condition 2: APs where x is 3rd term
                cond2 = True
                for d in range(1, (x - 1) // 2 + 1):
                    idx1 = x - 1 - 2 * d
                    idx2 = x - 1 - d
                    c1 = colors[idx1]
                    c2 = colors[idx2]
                    # If color == c1 and color == c2, then c1==c2==color.
                    # Then the AP (x-2d, x-d, x, x+d) will have at most 2 colors.
                    if color == c1 and color == c2:
                        cond2 = False
                        break
                if not cond2:
                    continue
                    
                colors.append(color)
                break
            else:
                # Should not happen
                colors.append(0)
        return colors

    # Try local search for k=3, 4, 5...
    # Time allocation:
    # 10s for k=3
    # 10s for k=4
    # 10s for k=5
    # 10s for k=6
    # Remaining for fallback
    
    time_per_k = 10
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
        found, colors_arr = local_search(k, time_per_k)
        if found:
            # Compress colors
            unique_colors = np.unique(colors_arr)
            if len(unique_colors) < k:
                # Map to 0..m-1
                mapping = {c: i for i, c in enumerate(unique_colors)}
                compressed = np.array([mapping[c] for c in colors_arr])
                return compressed.tolist()
            return colors_arr.tolist()

    # Fallback to seed greedy
    return seed_greedy()
