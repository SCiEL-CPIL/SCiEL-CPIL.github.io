import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency structure
    # For fast connectivity check: we need to check if there exists a non-trivial 
    # subset S where all crossing edges have sign +1
    # This is equivalent to: the graph formed by only +1 edges is not connected 
    # in a specific way... actually no.
    
    # Let me think: the lift is disconnected iff there's a non-trivial S such that
    # all edges between S and V\S have sign +1.
    # This means: in the graph H where we keep only edges with sign +1, 
    # S is a union of connected components of H, and S is non-trivial.
    # Wait no - we need ALL crossing edges to be +1. So in the graph of +1 edges,
    # there are no -1 edges crossing S/V\S. That means S is a union of connected 
    # components of the +1 edge subgraph, AND there are no -1 edges between S and V\S.
    # But if there are no -1 edges between S and V\S, then all edges between S and V\S 
    # are +1, which is what we want.
    
    # So: build graph G+ with only +1 edges. The lift is disconnected iff G+ has 
    # a proper non-empty connected component that has no -1 edges to the rest.
    # But G+ only has +1 edges, so if a connected component C of G+ has no -1 edges 
    # to V\C, then C is a valid disconnecting set.
    # Equivalently: the lift is connected iff every connected component of G+ 
    # has at least one -1 edge to the rest of the graph (unless G+ is connected, 
    # in which case the lift is connected).
    
    # Actually simpler: the lift is connected iff the graph G- (with -1 edges) 
    # connects all the "sides" properly. Let me just use BFS/DFS on the lift directly.
    
    # For n up to 18, building the lift and checking connectivity is very fast.
    
    def build_lift_adjacency(signs):
        """Build adjacency list of the 2-lift."""
        # 2n vertices: (v, 0) -> 2*v, (v, 1) -> 2*v+1
        adj = [[] for _ in range(2 * n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # +1: (u,0)-(v,0) and (u,1)-(v,1)
                adj[2*u].append(2*v)
                adj[2*v].append(2*u)
                adj[2*u+1].append(2*v+1)
                adj[2*v+1].append(2*u+1)
            else:
                # -1: (u,0)-(v,1) and (u,1)-(v,0)
                adj[2*u].append(2*v+1)
                adj[2*v+1].append(2*u)
                adj[2*u+1].append(2*v)
                adj[2*v].append(2*u+1)
        return adj
    
    def is_connected_lift(signs):
        adj = build_lift_adjacency(signs)
        visited = [False] * (2 * n)
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    stack.append(nb)
        return count == 2 * n
    
    def spectral_radius(signs):
        # Build signed adjacency matrix of base graph
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def evaluate(signs):
        """Returns (is_valid, radius)."""
        if not is_connected_lift(signs):
            return False, None
        return True, spectral_radius(signs)
    
    # Local search
    def local_search(initial_signs, max_iter=5000):
        signs = initial_signs.copy()
        valid, radius = evaluate(signs)
        if not valid:
            return None, None
        
        best_radius = radius
        best_signs = signs.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try flipping each edge
            for i in range(num_edges):
                signs[i] = -signs[i]
                valid, r = evaluate(signs)
                if valid and r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()
                    improved = True
                else:
                    signs[i] = -signs[i]  # revert
            if not improved:
                break
        
        return best_signs, best_radius
    
    # Multiple restarts
    rng = np.random.default_rng(3)
    best_overall = None
    best_overall_radius = float('inf')
    
    # First, do the baseline random search
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges)
        valid, r = evaluate(signs)
        if valid and r < best_overall_radius:
            best_overall_radius = r
            best_overall = signs.copy()
    
    if best_overall is None:
        # Find any valid assignment
        for _ in range(1000):
            signs = rng.choice([-1, 1], size=num_edges)
            valid, r = evaluate(signs)
            if valid:
                best_overall = signs.copy()
                best_overall_radius = r
                break
    
    if best_overall is None:
        # Should not happen, but fallback
        return np.ones(num_edges, dtype=np.int8)
    
    # Now do local search from multiple starting points
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            start = best_overall.copy()
        else:
            # Perturb best solution
            start = best_overall.copy()
            num_flips = rng.integers(1, 4)
            flip_idx = rng.choice(num_edges, size=num_flips, replace=False)
            for i in flip_idx:
                start[i] = -start[i]
        
        result_signs, result_radius = local_search(start, max_iter=2000)
        
        if result_signs is not None and result_radius < best_overall_radius:
            best_overall_radius = result_radius
            best_overall = result_signs.copy()
    
    return best_overall.astype(np.int8)
