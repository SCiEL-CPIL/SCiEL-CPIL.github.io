import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Strategy: Use the incidence graph of the projective plane of order 5 (31 vertices, bipartite,
    hence C5-free, alpha=6). Add a 32nd vertex connected to all 6 points on one line to try
    to push n to 32 while maintaining C5-freeness and alpha < 9.
    Fallback: 8 disjoint K4s (n=32, alpha=8, C5-free).
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    # Try projective plane approach first
    result = _projective_plane_32(start_time, deadline)
    if result is not None:
        return result
    
    # Fallback: 8 disjoint copies of K4
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}

def _projective_plane_32(start_time, deadline):
    """
    Construct the incidence graph of PG(2,5) with 31 vertices (bipartite, C5-free, alpha=6),
    then add vertex 32 connected to all 6 points on one line.
    Check C5-freeness and alpha < 9.
    """
    # Build projective plane of order 5
    # Points: 31 = (5^2 + 5 + 1)
    # Lines: 31
    # Each line has 6 points, each point on 6 lines
    # Total edges: 31 * 6 = 186
    
    q = 5
    # Enumerate points as equivalence classes of vectors in F_q^3
    # Points: [1,0,0], [0,1,0], [0,0,1], [1,a,0], [0,1,a], [a,0,1], [1,a,b] for a,b in F_q
    # Actually simpler: use standard enumeration
    # Point (i,j) for i,j in 0..q-1 gives q^2 = 25 points
    # Plus q+1 = 6 points at infinity: (inf, k) for k in 0..q-1 and (inf, inf)
    # Total: 25 + 6 = 31
    
    # Lines: 
    # - Vertical lines: x = a for a in 0..q-1 (q lines), each containing points (a, 0..q-1) and (inf, inf)
    # - Horizontal lines: y = a for a in 0..q-1 (q lines), each containing points (0..q-1, a) and (inf, inf)  
    # Wait, that's not right. Let me use the standard construction.
    
    # Standard affine plane of order q has q^2 points and q(q+1) lines
    # Projective plane adds q+1 points and q+1 lines
    # Total: q^2+q+1 points, q^2+q+1 lines
    
    # Let me use a different, cleaner enumeration:
    # Points indexed 0..30
    # Lines: each line is a set of 6 points
    
    # Use the standard construction:
    # Affine points: (x,y) for x,y in {0,...,q-1}, indexed as x*q + y
    # Points at infinity: one for each slope in {0, 1, ..., q-1, infinity}, indexed as q^2 + slope
    # So point indices: 0..24 for affine, 25..30 for infinity (6 points)
    
    # Lines:
    # For each slope m in {0, 1, ..., q-1, inf} and each intercept b in {0, 1, ..., q-1}:
    #   If m = inf: vertical line x = b, contains points (b, 0), (b, 1), ..., (b, q-1), (inf, inf)
    #   If m = k (finite): line y = kx + b, contains points (x, kx+b) for x=0..q-1, (inf, k)
    
    # Total lines: (q+1) * q = 6 * 5 = 30... but we need 31 lines.
    # The missing line is the line at infinity: contains all 6 points at infinity.
    
    # So we have 31 lines total.
    
    def point_index(x, y):
        """Index for affine point (x,y)"""
        return x * q + y
    
    def infinity_index(slope):
        """Index for point at infinity with given slope (0..q-1 or 'inf')"""
        if slope == 'inf':
            return q * q + q  # 25 + 5 = 30
        else:
            return q * q + slope  # 25 + slope
    
    # Build lines
    lines = []
    
    # Vertical lines: x = b for b in 0..q-1
    for b in range(q):
        line = [point_index(b, y) for y in range(q)]
        line.append(infinity_index('inf'))
        lines.append(line)
    
    # Lines with finite slope: y = m*x + b for m in 0..q-1, b in 0..q-1
    for m in range(q):
        for b in range(q):
            line = []
            for x in range(q):
                y = (m * x + b) % q
                line.append(point_index(x, y))
            line.append(infinity_index(m))
            lines.append(line)
    
    # Line at infinity: all 6 points at infinity
    line_inf = [infinity_index(m) for m in range(q)] + [infinity_index('inf')]
    lines.append(line_inf)
    
    # Total: 5 + 25 + 1 = 31 lines. Good.
    
    # Build edges from incidence graph
    # Points: 0..30 (31 vertices)
    # Lines: 31..61 (31 vertices)
    # Edge between point p and line l if p is in line l
    
    n = 62  # 31 points + 31 lines
    edges_set = set()
    
    for li, line in enumerate(lines):
        line_vertex = 31 + li
        for p in line:
            edges_set.add(frozenset((p, line_vertex)))
    
    # Now add vertex 62 (the 32nd vertex)
    # Connect it to all 6 points on one line to help reduce independence number
    # But wait - in a bipartite graph, adding a vertex and connecting it to vertices on one side
    # keeps it bipartite (and thus C5-free) as long as we only connect to one partition.
    # Points are 0-30, Lines are 31-61. If we connect vertex 62 to some points (0-30),
    # the graph remains bipartite with partitions {0-30, 62} and {31-61}.
    # This is still C5-free.
    
    # To minimize independence number, connect vertex 62 to as many points as possible
    # without creating issues. Since it's bipartite, any subset of points we connect to is fine.
    # Connect to all 31 points? That would make vertex 62 adjacent to all points.
    # Then any independent set can contain at most: vertices from {lines} not adjacent to chosen points
    # Actually, let's think about alpha.
    
    # In the base incidence graph (bipartite, parts P={0..30}, L={31..61}):
    # Alpha = max size of independent set. Since it's bipartite, alpha >= max(|P|,|L|) = 31? No.
    # An independent set in a bipartite graph can take vertices from both sides as long as no edge connects them.
    # The maximum independent set in the incidence graph of PG(2,5): 
    # A set of points with no two on the same line, plus a set of lines with no two sharing a point,
    # plus no point-line incidences between the chosen sets.
    # Actually, the simplest independent set is all 31 points (no edges between points) or all 31 lines.
    # So alpha of the base graph is at least 31! That's way too large.
    
    # I was confused. The independence number of the incidence graph is NOT 6.
    # The independence number of the *line graph* or some other derived graph might be small.
    # 
    # Let me reconsider. The incidence graph is bipartite with parts of size 31 each.
    # An independent set can include ALL 31 points (since there are no edges between points).
    # So alpha >= 31. This doesn't satisfy alpha < 9.
    
    # The strategy in the directive was wrong about the independence number.
    # Let me abandon this and use a different approach.
    
    return None

def _has_c5(n, edge_set):
    """Check if graph has a 5-cycle. edge_set is a set of frozensets."""
    adj = [set() for _ in range(n)]
    for e in edge_set:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    from itertools import combinations
    for verts in combinations(range(n), 5):
        vs = set(verts)
        sub_edges = []
        for i in range(5):
            for j in range(i+1, 5):
                if verts[j] in adj[verts[i]]:
                    sub_edges.append((i, j))
        for perm in [(0,1,2,3,4), (0,1,2,4,3), (0,1,3,2,4), (0,1,3,4,2),
                     (0,1,4,2,3), (0,1,4,3,2), (0,2,1,3,4), (0,2,1,4,3),
                     (0,2,3,1,4), (0,2,4,1,3), (0,3,1,2,4), (0,3,1,4,2),
                     (0,3,2,1,4), (0,3,2,4,1), (0,3,4,1,2), (0,3,4,2,1),
                     (0,4,1,2,3), (0,4,1,3,2), (0,4,2,1,3), (0,4,2,3,1),
                     (0,4,3,1,2), (0,4,3,2,1)]:
            ok = True
            for k in range(5):
                a, b = perm[k], perm[(k+1)%5]
                if (a,b) not in sub_edges and (b,a) not in sub_edges:
                    ok = False
                    break
            if ok:
                return True
    return False

def _alpha_ge(n, edge_set, k=9):
    """Check if graph has independent set of size k."""
    adj = [set() for _ in range(n)]
    for e in edge_set:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    from itertools import combinations
    for verts in combinations(range(n), k):
        vs = set(verts)
        independent = True
        for i in range(k):
            for j in range(i+1, k):
                if verts[j] in adj[verts[i]]:
                    independent = False
                    break
            if not independent:
                break
        if independent:
            return True
    return False

def _try_33_vertex(start_time, deadline):
    """Try to construct a 33-vertex C5-free graph with alpha < 9."""
    base_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                base_edges.append((base + i, base + j))
    
    random.seed(42)
    best = None
    
    for attempt in range(200):
        if time.time() > deadline - 2:
            break
        
        edges_set = set()
        for e in base_edges:
            edges_set.add(frozenset(e))
        
        target_k4 = random.randint(0, 7)
        target_verts = [4*target_k4 + i for i in range(4)]
        
        for v in target_verts:
            edges_set.add(frozenset((32, v)))
        
        other_verts = [i for i in range(32) if i not in target_verts]
        num_extra = random.randint(0, 8)
        for v in random.sample(other_verts, num_extra):
            edges_set.add(frozenset((32, v)))
        
        if _has_c5(33, edges_set):
            continue
        
        if _alpha_ge(33, edges_set, 9):
            continue
        
        edges_list = [[u, v] for e in edges_set for u, v in [tuple(e)]]
        return {'n': 33, 'edges': edges_list}
    
    return None
