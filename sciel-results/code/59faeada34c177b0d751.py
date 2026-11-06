import time
import itertools

def _check_k4_fast(colors):
    """Check every K4 has at least 5 distinct colours. Returns True if valid."""
    n = 16
    edge = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edge[i][j] = c
            edge[j][i] = c
            idx += 1
    
    for a in range(n):
        ea = edge[a]
        for b in range(a + 1, n):
            eb = edge[b]
            ab = ea[b]
            for c in range(b + 1, n):
                ac = ea[c]
                bc = eb[c]
                ec = edge[c]
                for d in range(c + 1, n):
                    mask = (1 << ab) | (1 << ac) | (1 << ea[d]) | (1 << bc) | (1 << eb[d]) | (1 << ec[d])
                    if mask.bit_count() < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    best_colors = None
    best_count = 999
    
    # Approach 1: Finite geometry over F4^2
    # F4 = {0, 1, alpha, alpha^2} where alpha^2 = alpha + 1
    # Elements: 0, 1, 2(alpha), 3(alpha^2)
    # alpha + 1 = alpha^2, alpha^2 + 1 = alpha, alpha + alpha = 0, etc.
    
    # Define F4 arithmetic
    # Let's use: 0, 1, 2, 3 where 2=alpha, 3=alpha^2
    # Addition table for F4:
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=0, 1+2=3, 1+3=2
    # 2+2=0, 2+3=1
    # 3+3=0
    
    f4_add = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    
    # Multiplication table for F4:
    # 0*x=0
    # 1*x=x
    # 2*2=3, 2*3=1, 3*3=2
    f4_mul = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def f4_inv(x):
        if x == 0:
            return None
        for y in range(4):
            if f4_mul[x][y] == 1:
                return y
        return None
    
    # Map 16 vertices to F4^2: vertex i -> (i // 4, i % 4)
    def vertex_to_point(i):
        return (i // 4, i % 4)
    
    # Compute slope between two points in F4^2
    # Points: (x1,y1), (x2,y2)
    # If x1 != x2: slope = (y2-y1)/(x2-x1)
    # If x1 == x2: vertical, slope = infinity (use special value)
    def compute_slope(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        
        dx = f4_add[x2][f4_add[x1][3]]  # x2 - x1 = x2 + (-x1) = x2 + x1 (since char 2)
        dy = f4_add[y2][f4_add[y1][3]]  # y2 - y1 = y2 + y1
        
        if dx == 0:
            return 4  # vertical line
        else:
            dx_inv = f4_inv(dx)
            slope = f4_mul[dy][dx_inv]
            return slope
    
    # Generate slope-based coloring
    colors_slope = []
    for i in range(n):
        for j in range(i + 1, n):
            p1 = vertex_to_point(i)
            p2 = vertex_to_point(j)
            slope = compute_slope(p1, p2)
            colors_slope.append(slope)
    
    # Check if valid
    if _check_k4_fast(colors_slope):
        count = _num_colours(colors_slope)
        if count < best_count:
            best_count = count
            best_colors = colors_slope[:]
    
    # Approach 2: Additive mod 17 (original)
    colors_add = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    if _check_k4_fast(colors_add):
        count = _num_colours(colors_add)
        if count < best_count:
            best_count = count
            best_colors = colors_add[:]
    
    # Approach 3: Try other prime moduli
    for p in [13, 19, 23, 29, 31]:
        if time.time() > deadline - 30:
            break
        colors_p = [(i + j) % p for i in range(n) for j in range(i + 1, n)]
        if _check_k4_fast(colors_p):
            count = _num_colours(colors_p)
            if count < best_count:
                best_count = count
                best_colors = colors_p[:]
    
    # Approach 4: Try to reduce colors by merging
    if best_colors is not None:
        for _ in range(3):  # Try up to 3 rounds of merging
            if time.time() > deadline - 15:
                break
            
            unique = list(set(best_colors))
            if len(unique) <= best_count:
                break
            
            merged = False
            for c1, c2 in itertools.combinations(unique, 2):
                if time.time() > deadline - 10:
                    break
                
                new_colors = [c1 if c == c2 else c for c in best_colors]
                if _check_k4_fast(new_colors):
                    best_colors = new_colors
                    best_count = _num_colours(best_colors)
                    merged = True
                    break
            
            if not merged:
                break
    
    # If no valid coloring found, fallback
    if best_colors is None:
        import random
        while time.time() < deadline - 5:
            best_colors = [random.randint(0, 16) for _ in range(n * (n - 1) // 2)]
            if _check_k4_fast(best_colors):
                break
    
    return best_colors
