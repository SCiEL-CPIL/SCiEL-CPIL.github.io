import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 60s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Identify edge types
    bottom_edges = []  # edges in bottom cycle
    top_edges = []     # edges in top cycle
    match_edges = []   # matching edges

    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_edges.append(i)
        elif u >= m and v >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)

    # Fast local validity check: build lift adjacency, check 3-regular and connected
    def is_valid_and_radius(signs):
        """
        Check if the two-lift is connected and 3-regular, and compute spectral radius.
        Returns (True, radius) if valid, (False, inf) otherwise.
        """
        # Build the lift: 2n vertices, (v,0) and (v,1)
        # For each edge (u,v) with sign s:
        #   if s == +1: connect (u,0)-(v,0) and (u,1)-(v,1)
        #   if s == -1: connect (u,0)-(v,1) and (u,1)-(v,0)
        # The lift has 2n vertices and 2*E edges, each vertex has degree 3.
        
        # Build adjacency list for the lift
        lift_adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                # equal sides
                lift_adj[2*u].append(2*v)
                lift_adj[2*v].append(2*u)
                lift_adj[2*u+1].append(2*v+1)
                lift_adj[2*v+1].append(2*u+1)
            else:
                # opposite sides
                lift_adj[2*u].append(2*v+1)
                lift_adj[2*v+1].append(2*u)
                lift_adj[2*u+1].append(2*v)
                lift_adj[2*v].append(2*u+1)
        
        # Check 3-regular
        for v in range(2 * n):
            if len(lift_adj[v]) != 3:
                return False, np.inf
        
        # Check connected via BFS
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for nb in lift_adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        
        if count != 2 * n:
            return False, np.inf
        
        # Compute spectral radius of signed adjacency matrix of base graph
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return True, float(np.max(np.abs(eigvals)))

    # Gauge invariance: fix a spanning tree to +1
    # Spanning tree: all m matching edges + first m-1 bottom cycle edges
    # Chords: last bottom cycle edge + all m top cycle edges = m+1 chords
    
    chord_indices = [bottom_edges[m-1]] + top_edges[:]
    num_chords = len(chord_indices)
    num_assignments = 2 ** num_chords

    base_signs = np.ones(E, dtype=np.int8)

    best_signs = None
    best_radius = np.inf

    # Enumerate all 2^(m+1) sign assignments for the chords
    # For m=4: 2^5=32, m=5: 2^6=64, m=7: 2^8=256, m=9: 2^10=1024
    # All are small enough to enumerate exhaustively
    
    for mask in range(num_assignments):
        if time.time() > deadline:
            break
        
        signs = base_signs.copy()
        for j, chord_idx in enumerate(chord_indices):
            if (mask >> j) & 1:
                signs[chord_idx] = -1
            else:
                signs[chord_idx] = 1
        
        valid, radius = is_valid_and_radius(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()

    # Fallback if no valid lift found (shouldn't happen for m=4,5,7,9)
    if best_signs is None:
        # Try all +1
        best_signs = base_signs.copy()
        valid, radius = is_valid_and_radius(best_signs)
        if not valid:
            # Try a few random assignments
            rng = np.random.default_rng(42)
            for _ in range(100):
                signs = rng.choice([-1, 1], size=E).astype(np.int8)
                valid, radius = is_valid_and_radius(signs)
                if valid:
                    best_signs = signs
                    break

    return best_signs
