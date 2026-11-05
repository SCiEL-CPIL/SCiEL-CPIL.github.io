import random

def construct():
    random.seed(42)
    n = 128
    
    def count_bad(coloring):
        bad = 0
        for a in range(1, n+1):
            for d in range(1, n - a):
                e = a + 3*d
                if e > n:
                    break
                b = a + d
                c = a + 2*d
                cols = {coloring[a-1], coloring[b-1], coloring[c-1], coloring[e-1]}
                if len(cols) < 3:
                    bad += 1
        return bad
    
    # Start with random 3-coloring
    best_coloring = None
    best_bad = float('inf')
    
    for trial in range(50):
        coloring = [random.randint(0, 2) for _ in range(n)]
        bad = count_bad(coloring)
        if bad < best_bad:
            best_bad = bad
            best_coloring = coloring[:]
        
        if bad == 0:
            break
    
    # Local search
    if best_coloring is not None:
        coloring = best_coloring
        bad = best_bad
        improved = True
        while improved and bad > 0:
            improved = False
            for i in range(n):
                old_color = coloring[i]
                for new_color in range(3):
                    if new_color == old_color:
                        continue
                    coloring[i] = new_color
                    new_bad = count_bad(coloring)
                    if new_bad < bad:
                        bad = new_bad
                        improved = True
                    else:
                        coloring[i] = old_color
                if bad == 0:
                    break
        
        if bad == 0:
            return coloring
    
    # Try 4 colors
    for trial in range(50):
        coloring = [random.randint(0, 3) for _ in range(n)]
        bad = count_bad(coloring)
        if bad < best_bad:
            best_bad = bad
            best_coloring = coloring[:]
        
        if bad == 0:
            break
    
    if best_coloring is not None and best_bad == 0:
        return best_coloring
    
    # Fallback: greedy
    colors = []
    for x in range(1, n+1):
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                if x-1-3*d >= 0:
                    cols = {color, colors[x-1-d], colors[x-1-2*d], colors[x-1-3*d]}
                    if len(cols) < 3:
                        valid = False
                        break
            if not valid:
                continue
            # Also check if this creates a 2-color AP with the new color
            for d in range(1, (x-1)//2 + 1):
                if x-1-2*d >= 0:
                    # Check APs where x is the 3rd or 4th term
                    pass  # The above check covers all 4-term APs ending at x
            if valid:
                colors.append(color)
                break
    return colors
