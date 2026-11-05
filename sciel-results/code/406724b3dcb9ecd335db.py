"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: LUW Construction
    def get_luw_graph(d, q_val):
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q_val
            sub = lambda x, y: (x - y) % q_val
            mul = lambda x, y: x * y % q_val

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth(n, edges, min_girth):
        if n == 0: return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    best_n = float('inf')
    best_edges = []
    
    # 1. Try Benson/Group-based construction for k=5, g=9
    if k == 5 and g == 9:
        # Try PSL(2, 11) Cayley graph
        try:
            n_psl, edges_psl = construct_psl_cayley(11, 5, 9, deadline)
            if n_psl < best_n:
                best_n = n_psl
                best_edges = edges_psl
        except Exception:
            pass
            
        # Try PSL(2, 13) Cayley graph
        if best_n > 1000:
            try:
                n_psl, edges_psl = construct_psl_cayley(13, 5, 9, deadline)
                if n_psl < best_n:
                    best_n = n_psl
                    best_edges = edges_psl
            except Exception:
                pass

    # 2. Try LUW as fallback
    if best_n == float('inf') or time.time() < deadline - 20:
        # For k=5, q=5. d=4 gives girth >= 10, n=1250
        # d=3 gives girth >= 8 (invalid for g=9)
        # So use d=4
        try:
            n_luw, edges_luw = get_luw_graph(4, k)
            if n_luw < best_n:
                best_n = n_luw
                best_edges = edges_luw
        except Exception:
            pass

    if best_n == float('inf'):
        # Last resort: Random regular graph with greedy repair (unlikely to work for g=9)
        # Just return a valid large graph if LUW failed
        try:
            n_luw, edges_luw = get_luw_graph(5, k)
            best_n = n_luw
            best_edges = edges_luw
        except:
            pass

    return best_n, best_edges

def construct_psl_cayley(q, k, g, deadline):
    """
    Construct a k-regular Cayley graph of PSL(2, q) with girth >= g.
    Uses random search for generator sets.
    """
    if time.time() > deadline - 10:
        return float('inf'), []
    
    # Generate PSL(2, q) elements
    # Elements are matrices [[a,b],[c,d]] with ad-bc=1, modulo +/- I
    # For q=11, |PSL(2,11)| = 660
    # For q=13, |PSL(2,13)| = 1092
    
    def mat_mul(A, B):
        return [
            [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % q, (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % q],
            [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % q, (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % q]
        ]

    def mat_inv(A):
        # Inverse of [[a,b],[c,d]] is [[d,-b],[-c,a]] for det=1
        return [
            [A[1][1], (-A[0][1]) % q],
            [(-A[1][0]) % q, A[0][0]]
        ]

    # Generate all elements
    elements = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # d is determined by ad - bc = 1 => d = (1+bc)/a if a!=0
                # If a=0, then -bc=1 => c!=0, b!=0, d is free? No, det=1.
                # If a=0, -bc=1 => bc = -1. d can be anything?
                # No, det = ad - bc = 0*d - bc = -bc = 1 => bc = -1.
                # So if a=0, we need bc = q-1.
                # d is free? No, the matrix is [[0,b],[c,d]].
                # We need to iterate d as well.
                for d in range(q):
                    det = (a*d - b*c) % q
                    if det == 1:
                        key = (a, b, c, d)
                        if key not in seen:
                            seen.add(key)
                            elements.append([[a, b], [c, d]])
                    elif det == q - 1:
                        # This is in SL(2,q) but not PSL(2,q) if we quotient by -I
                        # Actually PSL(2,q) = SL(2,q) / {I, -I}
                        # We can just use SL(2,q) and identify M and -M
                        key = (a, b, c, d)
                        if key not in seen:
                            seen.add(key)
                            elements.append([[a, b], [c, d]])
    
    # Identify M and -M
    # Map each element to a canonical form
    # Let's just use the list and handle duplicates in the graph construction
    
    n_group = len(elements)
    
    # We want a k-regular graph.
    # Cayley graph: vertices are group elements.
    # Edges: (g, g*h) for h in S, where S is a set of k generators.
    # S must be symmetric (if h in S, h^-1 in S) and not contain identity.
    
    # Random search for S
    max_attempts = 50
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Pick k random elements
        # Ensure they are not identity
        # Identity is [[1,0],[0,1]]
        identity = (1, 0, 0, 1)
        
        candidates = [e for e in elements if tuple(map(tuple, e)) != identity]
        if len(candidates) < k:
            continue
            
        S = random.sample(candidates, k)
        
        # Check if S generates the group
        # BFS from identity using S
        visited = {identity: 0}
        queue = deque([identity])
        while queue:
            u = queue.popleft()
            for h in S:
                # v = u * h
                u_mat = [[u[0], u[1]], [u[2], u[3]]]
                h_mat = h
                v_mat = mat_mul(u_mat, h_mat)
                v = (v_mat[0][0], v_mat[0][1], v_mat[1][0], v_mat[1][1])
                if v not in visited:
                    visited[v] = visited[u] + 1
                    queue.append(v)
                    
        if len(visited) != n_group:
            continue
            
        # Check if the graph is k-regular and simple
        # Build adjacency list
        adj = {}
        for u in visited:
            u_mat = [[u[0], u[1]], [u[2], u[3]]]
            neighbors = []
            for h in S:
                h_mat = h
                v_mat = mat_mul(u_mat, h_mat)
                v = (v_mat[0][0], v_mat[0][1], v_mat[1][0], v_mat[1][1])
                neighbors.append(v)
            adj[u] = neighbors
            
        # Check for loops (u in adj[u])
        has_loop = False
        for u in adj:
            if u in adj[u]:
                has_loop = True
                break
        if has_loop:
            continue
            
        # Check for multi-edges (duplicates in adj[u])
        has_multi = False
        for u in adj:
            if len(adj[u]) != len(set(adj[u])):
                has_multi = True
                break
        if has_multi:
            continue
            
        # Check girth
        # Convert to integer indices for fast BFS
        index_map = {v: i for i, v in enumerate(visited)}
        n = n_group
        edges = []
        for u in visited:
            i_u = index_map[u]
            for v in adj[u]:
                i_v = index_map[v]
                if i_u < i_v:
                    edges.append((i_u, i_v))
                    
        if check_girth(n, edges, g):
            return n, edges
            
    return float('inf'), []
# EVOLVE-BLOCK-END
