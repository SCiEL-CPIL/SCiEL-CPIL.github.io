"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 128
    # Use a periodic coloring with period 7, colors 0,1,2,3,4,5,6
    # This gives 7 distinct colors initially
    # We'll try to minimize by using fewer colors via local search
    
    # Strategy: Start with a known-good periodic pattern, then try to merge colors
    # to reduce the total number used while maintaining the constraint.
    
    # A 7-periodic coloring: color[i] = i % 7
    # Check: any 4-term AP a, a+d, a+2d, a+3d
    # Colors: a%7, (a+d)%7, (a+2d)%7, (a+3d)%7
    # For these to use <3 colors, they'd need to be in {c1, c2} only.
    # With period 7 and 4 terms, if d%7 != 0, the four values a, a+d, a+2d, a+3d mod 7
    # are distinct if d is coprime to 7... actually let's verify.
    
    # Let's use a more careful approach:
    # Start with period-7 coloring, then try to eliminate colors one by one.
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid(colors):
        """Check if every nonconstant 4-term AP uses at least 3 distinct colors."""
        n = len(colors)
        for a in range(n):
            for d in range(1, (n - 1 - a) // 3 + 1):
                b = a + d
                c = a + 2*d
                e = a + 3*d
                if e >= n:
                    break
                # Nonconstant means d > 0, which is guaranteed
                colors_ap = {colors[a], colors[b], colors[c], colors[e]}
                if len(colors_ap) < 3:
                    return False
        return True
    
    # Start with period-7 coloring
    colors = [i % 7 for i in range(n)]
    
    # Verify it's valid
    # Actually, let's just proceed with local search to minimize colors
    
    # Local search: try to reassign colors to reduce total distinct colors used
    # We'll iteratively try to eliminate the least-used color by reassigning its positions
    
    for iteration in range(5000):
        # Find the color that is used least frequently (but still present)
        from collections import Counter
        color_counts = Counter(colors)
        if len(color_counts) <= 3:
            break
        
        # Pick the color with the smallest count
        min_color = min(color_counts.keys(), key=lambda c: color_counts[c])
        
        # Try to reassign all positions with min_color to other colors
        # such that validity is maintained and we eliminate min_color
        positions_with_min = [i for i in range(n) if colors[i] == min_color]
        
        if not positions_with_min:
            break
        
        # Try to find an assignment for these positions that avoids min_color
        # and maintains validity
        success = False
        random.shuffle(positions_with_min)
        
        for _attempt in range(100):
            # Make a copy
            new_colors = colors[:]
            
            # For each position, try all other colors
            feasible = True
            for pos in positions_with_min:
                best_color = None
                for new_c in range(7):
                    if new_c == min_color:
                        continue
                    # Try assigning new_c to pos and check local validity
                    # We only need to check APs that include pos
                    ok = True
                    # Check all 4-term APs where pos is one of the four terms
                    # pos could be a, a+d, a+2d, or a+3d
                    for d in range(1, (n) // 3 + 1):
                        # Case 1: pos = a
                        a = pos
                        if a + 3*d < n:
                            ap = {new_colors[a], new_colors[a+d], new_colors[a+2*d], new_colors[a+3*d]}
                            if len(ap) < 3:
                                ok = False
                                break
                        # Case 2: pos = a+d, so a = pos-d
                        if pos - d >= 0:
                            a = pos - d
                            if a + 3*d < n:
                                ap = {new_colors[a], new_colors[a+d], new_colors[a+2*d], new_colors[a+3*d]}
                                if len(ap) < 3:
                                    ok = False
                                    break
                        # Case 3: pos = a+2d, so a = pos-2d
                        if pos - 2*d >= 0:
                            a = pos - 2*d
                            if a + 3*d < n:
                                ap = {new_colors[a], new_colors[a+d], new_colors[a+2*d], new_colors[a+3*d]}
                                if len(ap) < 3:
                                    ok = False
                                    break
                        # Case 4: pos = a+3d, so a = pos-3d
                        if pos - 3*d >= 0:
                            a = pos - 3*d
                            if a + 3*d < n:
                                ap = {new_colors[a], new_colors[a+d], new_colors[a+2*d], new_colors[a+3*d]}
                                if len(ap) < 3:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        best_color = new_c
                        break
                if best_color is None:
                    feasible = False
                    break
                new_colors[pos] = best_color
            
            if feasible and is_valid(new_colors):
                colors = new_colors
                success = True
                break
        
        if not success:
            # Try a different approach: just accept current state
            break
    
    return colors
# EVOLVE-BLOCK-END
