import random

def construct():
    n = 128
    palette = [0, 1, 2]  # Try 3 colors first
    
    def is_valid_partial(colors, x, d):
        # Check if adding color at position x (0-indexed) with the given d
        # violates the constraint for the 4-term AP ending at x with step d
        # Positions: x-3d, x-2d, x-d, x (0-indexed)
        if x - 3*d < 0:
            return True
        c1 = colors[x-3*d]
        c2 = colors[x-2*d]
        c3 = colors[x-d]
        s = {c1, c2, c3}
        if len(s) == 1:
            return False  # Impossible to fix
        # If len(s) == 2, x must not be in s
        # If len(s) == 3, anything works
        return True  # We'll check actual color validity separately
    
    def can_place(colors, x, color):
        # Check if placing 'color' at position x (0-indexed) is valid
        for d in range(1, x//3 + 1):
            if x - 3*d < 0:
                break
            c1 = colors[x-3*d]
            c2 = colors[x-2*d]
            c3 = colors[x-d]
            s = {c1, c2, c3, color}
            if len(s) < 3:
                return False
        return True
    
    def solve(colors, pos):
        if pos == n:
            return colors
        
        # Try colors in random order
        color_order = palette[:]
        random.shuffle(color_order)
        
        for color in color_order:
            if can_place(colors, pos, color):
                colors.append(color)
                result = solve(colors, pos + 1)
                if result is not None:
                    return result
                colors.pop()
        
        return None
    
    random.seed(42)
    result = solve([], 0)
    
    if result is not None:
        return result
    
    # Fallback: try with 4 colors
    palette = [0, 1, 2, 3]
    result = solve([], 0)
    if result is not None:
        return result
    
    # Last resort: return the greedy result from original
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
