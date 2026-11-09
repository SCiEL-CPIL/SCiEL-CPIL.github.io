import time
import random

def _has_independent_set_10(adj_mask, n):
    """
    Exact check for independent set of size 10 using branch and bound.
    """
    def popcount(x):
        return x.bit_count()
    
    def dfs(available_mask, size):
        if size >= 10:
            return True
        if available_mask == 0:
            return False
        
        if popcount(available_mask) < 10 - size:
            return False
            
        # Branch on the lowest set bit
        v_bit = available_mask & -available_mask
        v = v_bit.bit_length() - 1
        
        # Include v
        new_avail = available_mask & ~(adj_mask[v] | v_bit)
        if dfs(new_avail, size + 1):
            return True
            
        # Exclude v
        new_avail = available_mask & ~v_bit
        if dfs(new_avail, size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def _greedy_independent_set_size(adj, n):
    """
    Greedy estimate of independence number.
    """
    available = (1 << n) - 1
    count = 0
    while available:
        # Pick vertex with minimum degree in available
        best_v = -1
        min_deg = n + 1
        temp = available
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = (adj[v] & available).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            temp &= temp - 1
        
        if best_v == -1:
            break
        
        count += 1
        available &= ~(adj[best_v] | (1 << best_v))
    
    return count

def _polarity_graph_pg2_q(q):
    """
    Construct the polarity graph of PG(2, q).
    Vertices are points of PG(2, q), which can be represented as:
    - (1, a, b) for a, b in GF(q)  -> q^2 vertices
    - (0, 1, c) for c in GF(q)     -> q vertices
    - (0, 0, 1)                     -> 1 vertex
    Total: q^2 + q + 1 vertices
    
    Two points are adjacent if they are orthogonal under the polarity.
    For the standard polarity in PG(2, q), (x1,y1,z1) ~ (x2,y2,z2) iff
    x1*x2 + y1*y2 + z1*z2 = 0 (for appropriate representation)
    
    Actually, let's use a simpler construction: the graph where vertices are
    elements of GF(q^2) and u~v iff Tr(u*v) = 0, where Tr is the trace
    from GF(q^2) to GF(q). This gives a triangle-free graph on q^2 vertices
    with independence number related to q.
    
    Actually, the polarity graph of PG(2,q) has q^2+q+1 vertices and is
    triangle-free. Let me implement it properly.
    """
    # For simplicity and correctness, let's use a different approach:
    # Cayley graph on Z_n where the connection set S has the property that
    # no three elements sum to 0 (mod n) in a way that creates triangles.
    
    # Actually, let's use the known construction: 
    # The Andrásfai graph And(k) is triangle-free with alpha ~ 2k.
    # For our purposes, let's try to construct graphs using finite field methods.
    
    # Simple approach: use the graph where vertices are 0..q^2-1 (as elements of GF(q^2))
    # and u~v iff Tr(u*v) = 0 in GF(q).
    
    # But implementing GF arithmetic is complex. Let's use a simpler deterministic construction.
    
    # Alternative: Use the blow-up of C5 or other known constructions.
    # For now, return None to indicate this function is a placeholder.
    return None

def _cayley_graph_triangle_free(n):
    """
    Try to construct a triangle-free Cayley graph on Z_n.
    A Cayley graph Cay(Z_n, S) is triangle-free if and only if
    S + S does not intersect S (i.e., no three elements of S sum to 0 mod n
    in a way that creates a triangle, considering the group structure).
    
    More precisely, for an abelian group, Cay(G, S) is triangle-free iff
    there are no a, b, c in S such that a + b + c = 0 (with appropriate
    conditions depending on whether elements are self-inverse).
    
    For Z_n, if we take S to be a set with no 3-term arithmetic progression
    and no solutions to x + y = z for x,y,z in S (with proper handling of
    inverses), we can get triangle-free graphs.
    
    A simpler sufficient condition: if S is a Sidon set (all pairwise sums
    are distinct) and S ∩ (-S) = ∅, then Cay(Z_n, S) is triangle-free.
    """
    adj = [0] * n
    
    # Try to find a connection set S such that:
    # 1. S is symmetric: if s in S, then -s mod n is in S (for undirected)
    # 2. No triangles: no a,b,c in S with a+b+c=0 mod n where the edges
    #    (0,a), (a,a+b), (a+b,0) form a triangle
    
    # Heuristic: try small random symmetric sets and check
    best_adj = None
    best_alpha = n + 1
    
    for attempt in range(100):
        rng = random.Random(attempt * 12345 + n)
        
        # Generate a random symmetric set S
        S = set()
        for i in range(1, n // 2):
            if rng.random() < 0.3:  # 30% chance to include each pair
                S.add(i)
                S.add(n - i)
        
        if not S:
            continue
        
        # Check triangle-freeness
        is_triangle_free = True
        for a in S:
            for b in S:
                c = (-a - b) % n
                if c in S:
                    # Check if this forms a triangle
                    # Vertices 0, a, a+b form a triangle if:
                    # (0,a) in S, (a, a+b) means b in S, (a+b, 0) means -(a+b) = -a-b = c in S
                    # So if a, b, c all in S and c = -a-b, we have a triangle
                    # But we need to be careful about degenerate cases
                    if a != 0 and b != 0 and c != 0:
                        # This is a valid triangle 0-a-(a+b)-0
                        is_triangle_free = False
                        break
            if not is_triangle_free:
                break
        
        if not is_triangle_free:
            continue
        
        # Build adjacency
        adj = [0] * n
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i] |= (1 << j)
        
        # Estimate independence number
        alpha_est = _greedy_independent_set_size(adj, n)
        
        if alpha_est < best_alpha:
            best_alpha = alpha_est
            best_adj = adj[:]
        
        if alpha_est < 10:
            # Verify exactly
            if not _has_independent_set_10(adj, n):
                return adj
    
    return best_adj

def _blow_up_c5(k):
    """
    Blow up C5: replace each vertex of C5 with an independent set of size k.
    Edges between parts if the corresponding vertices in C5 are adjacent.
    This gives a triangle-free graph on 5k vertices.
    Independence number is k (one part is independent).
    For alpha < 10, we need k <= 9, giving n = 45. But wait, the blow-up
    of C5 has alpha = k since each part is independent of size k.
    So for k=9, n=45, alpha=9 < 10. But n must be <= 41.
    For k=8, n=40, alpha=8 < 10. This works!
    """
    n = 5 * k
    adj = [0] * n
    
    # Part i contains vertices i*k to (i+1)*k - 1
    # C5 edges: (0,1), (1,2), (2,3), (3,4), (4,0)
    c5_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    
    for i, j in c5_edges:
        for a in range(k):
            for b in range(k):
                u = i * k + a
                v = j * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # First, try the C5 blow-up construction which is deterministic
    # For k=8, n=40, alpha=8 < 10
    for k in range(8, 3, -1):
        n = 5 * k
        if n > 41:
            continue
        
        adj = _blow_up_c5(k)
        
        # Verify it's triangle-free and has no independent set of size 10
        # Triangle-free: C5 blow-up is triangle-free since C5 is triangle-free
        # and we only add edges between parts corresponding to C5 edges.
        # Independent set: max independent set is one part of size k.
        # So if k < 10, we're good.
        
        if k < 10:
            # Double-check with exact verification
            if not _has_independent_set_10(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Try Cayley graph constructions
    for n in range(40, 29, -1):
        if time.time() > deadline - 10:
            break
        
        adj = _cayley_graph_triangle_free(n)
        if adj is not None:
            if not _has_independent_set_10(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Fallback: return C5 blow-up with k=6 (n=30) as a safe default
    adj = _blow_up_c5(6)
    n = 30
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
