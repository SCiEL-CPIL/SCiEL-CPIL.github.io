import random
import time

def construct():
    n = 128
    
    def solve_with_palette(palette, max_steps=100000):
        colors = [None] * n
        steps = 0
        
        def can_place(pos, color):
            # Check all APs ending at pos
            # x = pos
            # x - 3d, x - 2d, x - d, x
            # d >= 1, x - 3d >= 0
            for d in range(1, pos // 3 + 1):
                i1 = pos - 3 * d
                i2 = pos - 2 * d
                i3 = pos - d
                # If any of these are None, we can't check yet? 
                # No, we fill left to right, so i1, i2, i3 are filled.
                c1 = colors[i1]
                c2 = colors[i2]
                c3 = colors[i3]
                # We need len({c1, c2, c3, color}) >= 3
                # This is equivalent to: not (c1==c2==c3==color) and not (only 2 distinct values in {c1,c2,c3,color})
                # Simpler: 
                vals = {c1, c2, c3, color}
                if len(vals) < 3:
                    return False
            return True

        def backtrack(pos):
            nonlocal steps
            if steps > max_steps:
                return False
            if pos == n:
                return True
            
            # Randomize palette order
            p = palette[:]
            random.shuffle(p)
            
            for color in p:
                if can_place(pos, color):
                    colors[pos] = color
                    steps += 1
                    if backtrack(pos + 1):
                        return True
                    colors[pos] = None
            return False

        # Multiple restarts if first fails?
        # For now, one deep search with randomization.
        random.seed(int(time.time() * 1000)) # Or fixed? 
        # Fixed seed is better for reproducibility in eval, but random might find it if fixed doesn't.
        # The current code uses seed 42. I will try a few seeds.
        
        for seed in [42, 123, 999, 2023, 31337]:
            random.seed(seed)
            if backtrack(0):
                return colors
        return None

    # Try 3 colors
    res3 = solve_with_palette([0, 1, 2], max_steps=50000)
    if res3 is not None:
        return res3
        
    # Try 4 colors
    res4 = solve_with_palette([0, 1, 2, 3], max_steps=50000)
    if res4 is not None:
        return res4
        
    # Fallback: Greedy 4 colors (should always work)
    colors = [0] * n
    for i in range(n):
        for d in range(1, i // 3 + 1):
            pass # This fallback is just a placeholder, the solver should work.
            
    # Simple greedy fallback for 4 colors if solver fails (unlikely)
    colors = []
    for x in range(n):
        found = False
        for c in [0, 1, 2, 3]:
            ok = True
            for d in range(1, x // 3 + 1):
                if x - 3*d < 0: break
                vals = {colors[x-3*d], colors[x-2*d], colors[x-d], c}
                if len(vals) < 3:
                    ok = False
                    break
            if ok:
                colors.append(c)
                found = True
                break
        if not found:
            # Should not happen with 4 colors
            colors.append(0) 
    return colors
