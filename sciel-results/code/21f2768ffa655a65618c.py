"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 128
    LOOKAHEAD = 4
    
    def valid_assignment(colors, pos, color):
        """Check if assigning 'color' to position 'pos' (1-indexed) is valid given current colors."""
        # Check all 4-term APs ending at or containing pos
        # A 4-term AP: a, a+d, a+2d, a+3d where all terms are in [1, n]
        # For position pos, it can be at index 0, 1, 2, or 3 in the AP
        
        # Case 1: pos is the 4th term (index 3): a = pos - 3d
        for d in range(1, (pos - 1) // 3 + 1):
            a = pos - 3 * d
            if a < 1:
                break
            # Terms: a, a+d, a+2d, pos
            c0 = colors[a - 1]
            c1 = colors[a + d - 1]
            c2 = colors[a + 2 * d - 1]
            # Need at least 3 distinct colors among {c0, c1, c2, color}
            if len({c0, c1, c2, color}) < 3:
                return False
        
        # Case 2: pos is the 3rd term (index 2): a = pos - 2d, 4th term = pos + d
        for d in range(1, (n - pos) // 1 + 1):
            a = pos - 2 * d
            if a < 1:
                break
            # Terms: a, a+d, pos, pos+d
            # pos+d must be <= n
            if pos + d > n:
                break
            c0 = colors[a - 1]
            c1 = colors[a + d - 1]
            c3 = colors[pos + d - 1]
            # Need at least 3 distinct colors among {c0, c1, color, c3}
            if len({c0, c1, color, c3}) < 3:
                return False
        
        # Case 3: pos is the 2nd term (index 1): a = pos - d, terms: a, pos, pos+d, pos+2d
        for d in range(1, (n - pos) // 2 + 1):
            a = pos - d
            if a < 1:
                break
            if pos + 2 * d > n:
                break
            c0 = colors[a - 1]
            c2 = colors[pos + d - 1]
            c3 = colors[pos + 2 * d - 1]
            # Need at least 3 distinct colors among {c0, color, c2, c3}
            if len({c0, color, c2, c3}) < 3:
                return False
        
        # Case 4: pos is the 1st term (index 0): a = pos, terms: pos, pos+d, pos+2d, pos+3d
        for d in range(1, (n - pos) // 3 + 1):
            if pos + 3 * d > n:
                break
            c1 = colors[pos + d - 1]
            c2 = colors[pos + 2 * d - 1]
            c3 = colors[pos + 3 * d - 1]
            # Need at least 3 distinct colors among {color, c1, c2, c3}
            if len({color, c1, c2, c3}) < 3:
                return False
        
        return True
    
    def simulate_greedy(colors, start_pos, max_pos):
        """Simulate greedy assignment from start_pos to max_pos. Return True if no dead-end."""
        sim_colors = colors[:]
        for p in range(start_pos, max_pos + 1):
            found = False
            for c in range(128):
                if valid_assignment(sim_colors, p, c):
                    sim_colors.append(c)
                    found = True
                    break
            if not found:
                return False
        return True
    
    colors = []
    
    for x in range(1, n + 1):
        assigned = False
        for c in range(128):
            # Check if this color is valid for position x
            if not valid_assignment(colors, x, c):
                continue
            
            # Look-ahead: simulate next LOOKAHEAD positions
            temp_colors = colors + [c]
            end_pos = min(x + LOOKAHEAD, n)
            if simulate_greedy(temp_colors, x + 1, end_pos):
                colors.append(c)
                assigned = True
                break
        
        if not assigned:
            # Fallback: just use the first valid color without look-ahead
            for c in range(128):
                if valid_assignment(colors, x, c):
                    colors.append(c)
                    break
    
    return colors
# EVOLVE-BLOCK-END
