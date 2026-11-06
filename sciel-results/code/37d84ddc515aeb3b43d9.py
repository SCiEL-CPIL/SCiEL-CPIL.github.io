import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute 4-term APs ending at each position
    # aps_ending[i] = list of (a, d) such that a+3d = i (0-indexed)
    aps_ending = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 3 + 1):
            a = i - 3 * d
            if a >= 0:
                aps_ending[i].append((a, d))
    
    # Precompute 3-term APs ending at each position for lookahead
    # threes_ending[i] = list of (a, d) such that a+2d = i (0-indexed)
    threes_ending = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 2 + 1):
            a = i - 2 * d
            if a >= 0:
                threes_ending[i].append((a, d))
    
    def is_valid(colors):
        """Check if the coloring is valid for all 4-term APs."""
        for i in range(n):
            for a, d in aps_ending[i]:
                vals = {colors[a], colors[a+d], colors[a+2*d], colors[i]}
                if len(vals) < 3:
                    return False
        return True
    
    def deterministic_greedy(max_color):
        """
        Deterministic greedy: assign smallest valid color at each position.
        Uses lookahead constraint: for any 3-term AP (a, a+d, a+2d) ending at i,
        the color c must satisfy c != colors[a] OR c != colors[a+d].
        This prevents the 4th term from being forced into a 2-color situation.
        """
        colors = [None] * n
        
        for i in range(n):
            if time.time() > deadline:
                return None
            
            # Find valid colors for position i
            valid_colors = []
            for c in range(max_color):
                # Check all 4-term APs ending at i
                ok_4 = True
                for a, d in aps_ending[i]:
                    vals = {colors[a], colors[a+d], colors[a+2*d], c}
                    if len(vals) < 3:
                        ok_4 = False
                        break
                if not ok_4:
                    continue
                
                # Lookahead: check 3-term APs where position i is the 3rd term
                # For AP (a, a+d, a+2d, a+3d) where a+2d = i,
                # we need that c != colors[a] OR c != colors[a+d]
                # This ensures the 4th term can be colored with at least 3 distinct colors
                ok_3 = True
                for a, d in threes_ending[i]:
                    if i + d < n:  # Only check if 4th term exists
                        c0 = colors[a]
                        c1 = colors[a+d]
                        # The 4th term at position i+d will need to form a valid 4-term AP
                        # We need that c is different from at least one of c0 or c1
                        if c == c0 and c == c1:
                            ok_3 = False
                            break
                if not ok_3:
                    continue
                
                valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Pick the smallest valid color
            colors[i] = valid_colors[0]
        
        # Final verification
        if is_valid(colors):
            return colors
        return None
    
    # Try with increasing number of colors
    for num_colors in [3, 4, 5, 6, 8, 10, 12, 16]:
        result = deterministic_greedy(num_colors)
        if result is not None:
            return result
    
    # Fallback: return all distinct colors
    return list(range(n))
