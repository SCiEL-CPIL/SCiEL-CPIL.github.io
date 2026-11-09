"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _verify_c4_free(n, colors):
    """Check no monochromatic C4. Returns True if valid."""
    # Build adjacency per color
    adj = [[] for _ in range(n)]
    color_adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[i].append(j)
            adj[j].append(i)
            color_adj[c].add((i,j))
            idx += 1
    # For each color, check no two vertices share >=2 common neighbors
    for c in range(5):
        # Build neighbor sets for this color
        neigh = [set() for _ in range(n)]
        for (i,j) in color_adj[c]:
            neigh[i].add(j)
            neigh[j].add(i)
        # Check pairs
        for i in range(n):
            for j in range(i+1, n):
                common = neigh[i] & neigh[j]
                if len(common) >= 2:
                    return False
    return True

def _construct_f5x5():
    """F_5 x F_5 construction with 5 colors."""
    n = 25
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            a = (j % 5) - (i % 5)
            b = (j // 5) - (i // 5)
            a %= 5
            b %= 5
            if a == 0:
                c = 0
            elif b == 0:
                c = 1
            elif b == a:
                c = 2
            elif b == 2*a % 5:
                c = 3
            else:
                c = 4
            colors.append(c)
    return n, colors

def _construct_z5x5_v2():
    """Alternative F_5 x F_5: partition non-zero vectors into 5 sets of sizes 4,4,4,4,8."""
    n = 25
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = i % 5, i // 5
            x2, y2 = j % 5, j // 5
            a = (x2 - x1) % 5
            b = (y2 - y1) % 5
            if a == 0 and b != 0:
                c = 0
            elif a != 0 and b == 0:
                c = 1
            elif b == a:
                c = 2
            elif b == (2*a) % 5:
                c = 3
            else:
                c = 4
            colors.append(c)
    return n, colors

def _construct_z5x5_v3():
    """Try different partition: slopes 0, inf, 1, 2, {3,4}"""
    n = 25
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = i % 5, i // 5
            x2, y2 = j % 5, j // 5
            a = (x2 - x1) % 5
            b = (y2 - y1) % 5
            if a == 0:
                c = 0  # vertical
            elif b == 0:
                c = 1  # horizontal
            elif b == a:
                c = 2
            elif b == (2*a) % 5:
                c = 3
            else:
                c = 4
            colors.append(c)
    return n, colors

def _construct_z5x5_v4():
    """Try: group by quadratic residues. a,b non-zero. Color by b^2 * a^(-2) mod 5."""
    n = 25
    # Non-zero squares mod 5: 1, 4
    # b^2/a^2 can be 1/1=1, 1/4=4, 4/1=4, 4/4=1
    # So only 2 values. Not enough.
    # Try: color by (b/a) in {1,2,3,4} plus special cases
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = i % 5, i // 5
            x2, y2 = j % 5, j // 5
            a = (x2 - x1) % 5
            b = (y2 - y1) % 5
            if a == 0:
                c = 0
            elif b == 0:
                c = 1
            else:
                # slope = b * a^(-1) mod 5
                inv_a = pow(a, 3, 5)  # a^3 = a^(-1) mod 5
                slope = (b * inv_a) % 5
                # slope in {1,2,3,4}
                if slope == 1:
                    c = 2
                elif slope == 2:
                    c = 3
                elif slope == 3:
                    c = 4
                else:
                    c = 4  # slope 4 grouped with 3
            colors.append(c)
    return n, colors

def _construct_z5x5_v5():
    """Symmetric partition: 5 groups of 4+4+4+4+8 won't work well.
    Try: use the 5 parallel classes of the affine plane AG(2,5).
    Each parallel class has 5 lines of 5 points each.
    But we need to assign colors to edges, not lines.
    
    Alternative: For each edge, the "direction" is one of 24 non-zero vectors.
    Partition into 5 sets. Let's try making it as symmetric as possible.
    """
    n = 25
    colors = []
    # Precompute: for each (a,b) non-zero, assign a color
    # 24 vectors, 5 colors -> sizes must sum to 24
    # Try: 5,5,5,5,4
    # Group by: for a!=0, b/a mod 5 gives 4 values, plus a=0 (4 vectors), b=0 (4 vectors)
    # That's 4+4+4+4+4+4 = 24 with 6 groups. Need 5.
    # Merge two groups.
    
    color_map = {}
    # a=0, b=1,2,3,4 -> color 0
    for b in range(1,5):
        color_map[(0,b)] = 0
    # a=1,2,3,4, b=0 -> color 1
    for a in range(1,5):
        color_map[(a,0)] = 1
    # slope 1: (a,a) for a=1..4 -> color 2
    for a in range(1,5):
        color_map[(a,a)] = 2
    # slope 2: (a,2a) for a=1..4 -> color 3
    for a in range(1,5):
        color_map[(a, (2*a)%5)] = 3
    # slope 3 and 4: -> color 4
    for a in range(1,5):
        color_map[(a, (3*a)%5)] = 4
        color_map[(a, (4*a)%5)] = 4
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = i % 5, i // 5
            x2, y2 = j % 5, j // 5
            a = (x2 - x1) % 5
            b = (y2 - y1) % 5
            c = color_map[(a,b)]
            colors.append(c)
    return n, colors

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Try constructions in order of expected n
    candidates = []
    
    # Try n=25 constructions
    for fn in [_construct_z5x5_v5, _construct_z5x5_v4, _construct_z5x5_v3, _construct_z5x5_v2, _construct_f5x5]:
        if time.time() > deadline - 30:
            break
        try:
            n, colors = fn()
            if _verify_c4_free(n, colors):
                candidates.append((n, colors))
        except Exception:
            pass
    
    # Sort by n descending
    candidates.sort(key=lambda x: -x[0])
    
    if candidates:
        n, colors = candidates[0]
        return {'n': n, 'colors': colors}
    
    # Fallback: original n=11
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
# EVOLVE-BLOCK-END
