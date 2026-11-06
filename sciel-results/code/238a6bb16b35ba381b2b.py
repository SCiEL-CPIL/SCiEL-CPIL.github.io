import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 128
    
    # Precompute all 4-term APs
    all_aps = []
    for d in range(1, n // 3 + 1):
        for i in range(n - 3 * d):
            all_aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # Precompute for each position, which APs it's in and its index in that AP
    pos_in_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(all_aps):
        pos_in_aps[a].append((idx, 0))
        pos_in_aps[b].append((idx, 1))
        pos_in_aps[c].append((idx, 2))
        pos_in_aps[d].append((idx, 3))
    
    def count_violations(colors):
        """Count number of 4-term APs with < 3 distinct colors."""
        count = 0
        for a, b, c, d in all_aps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                count += 1
        return count
    
    def is_valid(colors):
        """Check if coloring is valid: every 4-term AP has >= 3 distinct colors."""
        for a, b, c, d in all_aps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True
    
    def greedy_init(max_color=8, randomize=False, seed=None):
        """
        Greedy initialization: assign valid color at each position.
        """
        if seed is not None:
            random.seed(seed)
        
        colors = [None] * n
        
        for i in range(n):
            if time.time() > deadline:
                return None
            
            valid_colors = []
            for c in range(max_color):
                # Check all 4-term APs where i is the last element
                ok = True
                for ap_idx, pos_idx in pos_in_aps[i]:
                    if pos_idx != 3:
                        continue
                    a, b, c_pos, d_pos = all_aps[ap_idx]
                    if colors[a] is None or colors[b] is None or colors[c_pos] is None:
                        continue
                    vals = {colors[a], colors[b], colors[c_pos], c}
                    if len(vals) < 3:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Lookahead: for 3-term APs ending at i, ensure we don't force
                # the 4th term into a 2-color situation
                # Check: if there's a 3-term AP (a, b, i) where colors[a]==c and colors[b]==c
                # then the 4th term would need to be different, but might be forced
                for d in range(1, i // 2 + 1):
                    a = i - 2 * d
                    b = i - d
                    if a >= 0 and b >= 0 and colors[a] is not None and colors[b] is not None:
                        if colors[a] == c and colors[b] == c:
                            ok = False
                            break
                if not ok:
                    continue
                
                valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            if randomize:
                colors[i] = random.choice(valid_colors)
            else:
                colors[i] = valid_colors[0]
        
        if is_valid(colors):
            return colors
        return None
    
    def simulated_annealing(initial_colors, max_iters=50000):
        """
        Simulated annealing to minimize number of colors while maintaining validity.
        """
        colors = list(initial_colors)
        current_violations = count_violations(colors)
        current_colors = set(colors)
        current_energy = current_violations * 1000 + len(current_colors)
        
        best_colors = list(colors)
        best_energy = current_energy
        best_violations = current_violations
        best_num_colors = len(current_colors)
        
        T = 50.0
        cooling = 0.9998
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Random move: recolor a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            new_color = random.randint(0, 127)
            
            if new_color == old_color:
                continue
            
            colors[pos] = new_color
            
            # Compute new violations (incremental would be faster, but for simplicity)
            new_violations = count_violations(colors)
            new_colors = set(colors)
            new_energy = new_violations * 1000 + len(new_colors)
            
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or random.random() < np.exp(-delta_e / T):
                current_violations = new_violations
                current_colors = new_colors
                current_energy = new_energy
                
                if new_violations == 0 and len(new_colors) < best_num_colors:
                    best_colors = list(colors)
                    best_energy = new_energy
                    best_violations = new_violations
                    best_num_colors = len(new_colors)
                elif new_violations == 0 and len(new_colors) == best_num_colors and new_energy < best_energy:
                    best_colors = list(colors)
                    best_energy = new_energy
            else:
                colors[pos] = old_color
            
            T *= cooling
        
        return best_colors, best_violations, best_num_colors
    
    def try_merge_color(colors, color_to_remove, max_iters=1000):
        """
        Try to remove a color by recoloring all its positions to other colors.
        Returns new coloring if successful, None otherwise.
        """
        colors = list(colors)
        positions_with_color = [i for i in range(n) if colors[i] == color_to_remove]
        
        if not positions_with_color:
            return colors
        
        # Try to recolor each position with this color to a different color
        success = True
        for pos in positions_with_color:
            if time.time() > deadline - 10:
                return None
            
            old_color = colors[pos]
            other_colors = [c for c in set(colors) if c != color_to_remove]
            random.shuffle(other_colors)
            
            found = False
            for new_color in other_colors:
                colors[pos] = new_color
                if count_violations(colors) == 0:
                    found = True
                    break
                colors[pos] = old_color
            
            if not found:
                colors[pos] = old_color
                success = False
                break
        
        if success:
            new_colors = set(colors)
            if color_to_remove not in new_colors:
                return colors
            else:
                # Color was still used somewhere else, or our recoloring didn't work
                return None
        else:
            return None
    
    best_result = None
    best_num_colors = n + 1
    
    # Strategy 1: Get a valid initial coloring with greedy
    for max_c in [4, 5, 6, 8, 10, 12]:
        if time.time() > deadline - 30:
            break
        
        result = greedy_init(max_c, randomize=False)
        if result is not None:
            num_used = len(set(result))
            if num_used < best_num_colors:
                best_num_colors = num_used
                best_result = result
                break
    
    # Also try randomized greedy
    if best_result is None or time.time() < deadline - 30:
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            for max_c in [4, 5, 6]:
                result = greedy_init(max_c, randomize=True, seed=attempt * 100 + max_c)
                if result is not None:
                    num_used = len(set(result))
                    if num_used < best_num_colors:
                        best_num_colors = num_used
                        best_result = result
                    break
    
    if best_result is None:
        # Fallback: use a simple valid coloring
        return list(range(n))
    
    # Strategy 2: Simulated annealing to reduce colors
    if time.time() < deadline - 20:
        sa_colors, sa_viols, sa_num_colors = simulated_annealing(best_result, max_iters=30000)
        if sa_viols == 0 and sa_num_colors < best_num_colors:
            best_num_colors = sa_num_colors
            best_result = sa_colors
    
    # Strategy 3: Try to merge colors one by one
    if time.time() < deadline - 15:
        colors = list(best_result)
        current_colors = sorted(set(colors))
        
        # Try to remove each color from the palette
        for color in current_colors[:]:
            if time.time() > deadline - 15:
                break
            
            new_colors = try_merge_color(colors, color, max_iters=500)
            if new_colors is not None:
                num_used = len(set(new_colors))
                if num_used < best_num_colors:
                    best_num_colors = num_used
                    best_result = new_colors
                    colors = list(new_colors)
                    current_colors = sorted(set(colors))
    
    # Strategy 4: Final local search - try single-position recoloring to reduce colors
    if time.time() < deadline - 10:
        colors = list(best_result)
        current_colors = set(colors)
        
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for pos in range(n):
                if time.time() > deadline - 10:
                    break
                
                old_color = colors[pos]
                
                # Try each color in the current palette
                for new_color in sorted(current_colors):
                    if new_color == old_color:
                        continue
                    
                    colors[pos] = new_color
                    v = count_violations(colors)
                    
                    if v == 0:
                        new_colors = set(colors)
                        if len(new_colors) < len(current_colors):
                            current_colors = new_colors
                            improved = True
                        else:
                            colors[pos] = old_color
                    else:
                        colors[pos] = old_color
                
                if improved:
                    break
            
            if not improved:
                break
        
        if is_valid(colors):
            num_used = len(set(colors))
            if num_used < best_num_colors:
                best_num_colors = num_used
                best_result = colors
    
    if best_result is not None:
        return [int(x) for x in best_result]
    else:
        return list(range(n))
