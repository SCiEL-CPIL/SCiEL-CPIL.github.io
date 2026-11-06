import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    
    # Precompute all 4-APs
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                aps.append((a - 1, b - 1, c - 1, e - 1))
    
    num_aps = len(aps)
    
    # For each position, list the APs it's in
    pos_to_aps = [[] for _ in range(N)]
    for i, (a, b, c, e) in enumerate(aps):
        for pos in (a, b, c, e):
            pos_to_aps[pos].append(i)
    
    def count_violations(colors):
        """Count number of 4-APs with < 3 distinct colors."""
        violations = 0
        for (a, b, c, e) in aps:
            cs = {colors[a], colors[b], colors[c], colors[e]}
            if len(cs) < 3:
                violations += 1
        return violations
    
    def check_valid(colors):
        for (a, b, c, e) in aps:
            cs = {colors[a], colors[b], colors[c], colors[e]}
            if len(cs) < 3:
                return False
        return True
    
    def get_num_colors(colors):
        return len(set(colors))
    
    best_colors = None
    best_num_colors = 128  # worst case
    
    # Try with different numbers of colors
    for num_c in range(3, 8):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Random initial assignment
            colors = [random.randint(0, num_c - 1) for _ in range(N)]
            
            # Min-conflicts local search
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                # Find all violating APs
                violating_aps = []
                for i, (a, b, c, e) in enumerate(aps):
                    cs = {colors[a], colors[b], colors[c], colors[e]}
                    if len(cs) < 3:
                        violating_aps.append((a, b, c, e))
                
                if not violating_aps:
                    # Valid solution found
                    nc = get_num_colors(colors)
                    if nc < best_num_colors:
                        best_num_colors = nc
                        best_colors = colors[:]
                    break
                
                # Pick a random violating AP
                a, b, c, e = random.choice(violating_aps)
                
                # Pick a random position in this AP to recolor
                pos = random.choice([a, b, c, e])
                
                # Try all colors and pick the one that minimizes violations
                best_color = colors[pos]
                best_viol = None
                
                for new_color in range(num_c):
                    if new_color == colors[pos]:
                        continue
                    colors[pos] = new_color
                    # Count violations involving pos
                    v = 0
                    for ap_idx in pos_to_aps[pos]:
                        pa, pb, pc, pe = aps[ap_idx]
                        cs = {colors[pa], colors[pb], colors[pc], colors[pe]}
                        if len(cs) < 3:
                            v += 1
                    if best_viol is None or v < best_viol:
                        best_viol = v
                        best_color = new_color
                
                colors[pos] = best_color
            
            # After local search, check if valid
            if check_valid(colors):
                nc = get_num_colors(colors)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_colors = colors[:]
                
                # Try to reduce colors used
                # Greedily try to eliminate unused color labels
                used = set(colors)
                for color_to_remove in list(used):
                    if time.time() > deadline:
                        break
                    # Map color_to_remove to another used color
                    other_colors = [c for c in used if c != color_to_remove]
                    if not other_colors:
                        break
                    for target in other_colors:
                        # Try replacing color_to_remove with target
                        test_colors = colors[:]
                        for i in range(N):
                            if test_colors[i] == color_to_remove:
                                test_colors[i] = target
                        if check_valid(test_colors):
                            colors = test_colors
                            used.remove(color_to_remove)
                            break
                    if color_to_remove in used:
                        continue
                    # Check if we can remove more
                    used = set(colors)
                
                nc = get_num_colors(colors)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_colors = colors[:]
        
        # If we found a solution with num_c colors, no need to try more
        if best_num_colors <= num_c:
            break
    
    # Fallback: use the seed's greedy approach
    if best_colors is None or best_num_colors >= 128:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    prevs = [colors[x - 1 - j * d] for j in (1, 2, 3)]
                    if len({color} | set(prevs)) < 3:
                        ok = False
                        break
                if not ok:
                    continue
                # Also check APs where x is not the last term but involves previous positions
                # Actually, since we color sequentially, any 4-AP fully within 1..x has max = x
                # So only APs ending at x need checking
                colors.append(color)
                break
        best_colors = colors
        best_num_colors = get_num_colors(colors)
    
    # Ensure labels are in 0..127
    result = best_colors[:]
    for i in range(N):
        if result[i] >= 128:
            result[i] = result[i] % 128
    
    return result
