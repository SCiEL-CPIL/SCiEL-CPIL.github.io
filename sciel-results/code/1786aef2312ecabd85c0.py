# EVOLVE-BLOCK-START
import numpy as np
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_7[alpha], alpha^2 = 3).
    
    Uses a Maximum Independent Set approach on the conflict graph.
    The conflict graph has vertices for each point in F_49^2.
    An edge exists between two points if they lie on the same line 
    (vertical or with slope m in F_49).
    We find a large Independent Set S. The Nikodym set N is the complement of S.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    # Field element representation: index 0..48 corresponds to a + b*alpha
    # where index = a * p + b
    # Precompute addition and multiplication tables for speed
    
    def get_ab(idx):
        return idx // p, idx % p
        
    def to_idx(a, b):
        return a * p + b

    # Add table: add[i][j] = i + j
    add_tbl = [[0]*49 for _ in range(49)]
    # Mul table: mul[i][j] = i * j
    mul_tbl = [[0]*49 for _ in range(49)]
    
    for i in range(49):
        ia, ib = get_ab(i)
        for j in range(49):
            ja, jb = get_ab(j)
            # Addition
            ra = (ia + ja) % p
            rb = (ib + jb) % p
            add_tbl[i][j] = to_idx(ra, rb)
            
            # Multiplication: (ia + ib*alpha)(ja + jb*alpha)
            # = ia*ja + ia*jb*alpha + ib*ja*alpha + ib*jb*alpha^2
            # = (ia*ja + ib*jb*k) + (ia*jb + ib*ja)*alpha
            ma = (ia * ja + ib * jb * k) % p
            mb = (ia * jb + ib * ja) % p
            mul_tbl[i][j] = to_idx(ma, mb)

    # Generate all lines
    # A line is defined by direction and a point on it.
    # To avoid duplicates, we can just generate all unique sets of points.
    # Total points N = 2401.
    # Directions: 
    # 1. Vertical: (0, 1). Step: dx=0, dy=1.
    # 2. Slope m: (1, m). Step: dx=1, dy=m.
    
    # We will create a list of lines, where each line is a list of point indices.
    # Point index: P = x * 49 + y, where x, y in 0..48 are field element indices.
    
    lines = []
    
    # Vertical lines: x is constant, y varies
    for x in range(49):
        line_pts = []
        for y in range(49):
            pt_idx = x * 49 + y
            line_pts.append(pt_idx)
        lines.append(line_pts)
        
    # Slope lines: y = m*x + c
    # For each m in 0..48, and c in 0..48
    for m in range(49):
        for c in range(49):
            line_pts = []
            for x in range(49):
                # y = m*x + c
                y = add_tbl[mul_tbl[m][x]][c]
                pt_idx = x * 49 + y
                line_pts.append(pt_idx)
            lines.append(line_pts)
            
    # Build conflict graph (adjacency list)
    N = 2401
    adj = [set() for _ in range(N)]
    
    for line_pts in lines:
        # Add edges between all pairs in line_pts
        for i in range(len(line_pts)):
            u = line_pts[i]
            for j in range(i+1, len(line_pts)):
                v = line_pts[j]
                adj[u].add(v)
                adj[v].add(u)
                
    # Multi-start Greedy Maximum Independent Set
    rng = random.Random(42)
    best_S = set()
    n_restarts = 100 # Increased restarts
    
    for _ in range(n_restarts):
        S = set()
        available = set(range(N))
        
        # Shuffle order for determinism/randomness in selection
        # Instead of picking random from available each time, 
        # we can just iterate through a shuffled list and pick if still available.
        order = list(range(N))
        rng.shuffle(order)
        
        for v in order:
            if v in available:
                S.add(v)
                available.remove(v)
                # Remove neighbors
                for u in adj[v]:
                    if u in available:
                        available.remove(u)
                        
        if len(S) > len(best_S):
            best_S = S.copy()
            
    # Construct N (complement of best_S)
    points = []
    for i in range(N):
        if i not in best_S:
            x = i // 49
            y = i % 49
            xa, xb = get_ab(x)
            ya, yb = get_ab(y)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
