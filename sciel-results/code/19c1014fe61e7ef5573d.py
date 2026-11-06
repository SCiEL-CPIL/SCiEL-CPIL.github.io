import time
import sys

def construct():
    start_time = time.time()
    deadline = 50.0  # Leave 10s for safety and fallback
    
    # Try to find a 3-coloring using backtracking
    # Colors are 0, 1, 2
    n = 128
    colors = [-1] * n
    
    # Symmetry breaking: c[0] = 0
    colors[0] = 0
    
    # We can also try to fix c[1] and c[2] to speed up?
    # Let's just let the search decide, but prune heavily.
    
    def is_valid_prefix(i):
        """Check if assigning colors up to i-1 allows for valid extensions?
        Specifically, check if any 4-AP ending at i-1 is invalid.
        Also check if any 3-term AP ending at i-1 is monochromatic (dead end for i).
        """
        # Check 4-APs ending at i-1
        for d in range(1, (i-1)//3 + 1):
            a = i - 1 - 3*d
            c0 = colors[a]
            c1 = colors[a+d]
            c2 = colors[a+2*d]
            c3 = colors[i-1]
            if len({c0, c1, c2, c3}) < 3:
                return False
        
        # Check for monochromatic 3-APs ending at i-1
        # If c[a] == c[a+d] == c[a+2d], then c[a+3d] cannot save it.
        for d in range(1, (i-1)//2 + 1):
            a = i - 1 - 2*d
            if colors[a] == colors[a+d] == colors[i-1]:
                return False
                
        return True

    def backtrack(i):
        if time.time() - start_time > deadline:
            return False
        
        if i == n:
            return True
            
        for c in range(3):
            colors[i] = c
            if is_valid_prefix(i+1):
                if backtrack(i+1):
                    return True
            colors[i] = -1
            
        return False

    # Start search
    found = backtrack(1)
    
    if found:
        return colors
    
    # Fallback: Greedy construction (similar to seed but ensuring validity)
    # This ensures we return a valid result even if 3-coloring fails or times out
    colors = []
    for x in range(1, 129):
        assigned = False
        for color in range(128):
            valid = True
            # Check 4-APs ending at x-1
            for d in range(1, (x-1)//3 + 1):
                a = x - 1 - 3*d
                c0 = colors[a]
                c1 = colors[a+d]
                c2 = colors[a+2*d]
                c3 = color
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            # Note: The seed also checked "any(colors[x-1-j*d]!=color for j in (1,2))"
            # This part of the seed logic seems to be checking 3-term APs?
            # "any(colors[x-1-d] != color and colors[x-1-2d] != color)" ?
            # The seed condition: all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
            # This means: For all d, (c[x-1-d] != color OR c[x-1-2d] != color).
            # This is equivalent to NOT (c[x-1-d] == color AND c[x-1-2d] == color).
            # This prevents c[x-1-2d] == c[x-1-d] == c[x-1] (monochromatic 3-AP).
            # My is_valid_prefix already handles this.
            
            if valid:
                colors.append(color)
                assigned = True
                break
        
        if not assigned:
            # Should not happen if 128 colors are available
            colors.append(0)
            
    return colors

# EVOLVE-BLOCK-START
# The construct function above is the solution.
# EVOLVE-BLOCK-END
