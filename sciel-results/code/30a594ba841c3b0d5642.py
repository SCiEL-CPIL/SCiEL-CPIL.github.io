"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Identify edge types: cycle edges and matching edges
    # Vertices 0..m-1 form first cycle, m..2m-1 form second cycle
    # Matching edges connect i to i+m
    
    # Build adjacency info for the base graph
    # The signed adjacency matrix A has A[u,v] = A[v,u] = sign of edge (u,v)
    
    # For a 2-lift, the spectrum of the lift's adjacency matrix is related to
    # the signed adjacency matrix of the base graph.
    # The eigenvalues of the lift are the eigenvalues of the signed adjacency matrix.
    
    # Key insight: switching (flipping signs of all edges incident to a vertex)
    # preserves the spectrum. So we can enumerate switching classes.
    
    # For prism graphs, the structure is:
    # - Two m-cycles: (0,1,2,...,m-1,0) and (m,m+1,...,2m-1,m)
    # - Matching: (i, i+m) for i=0..m-1
    
    # Let's identify which edges are which
    cycle_edges_1 = []  # edges in first cycle
    cycle_edges_2 = []  # edges in second cycle
    matching_edges = []  # matching edges
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            cycle_edges_1.append(idx)
        elif u >= m and v >= m:
            cycle_edges_2.append(idx)
        else:
            matching_edges.append(idx)
    
    # The signed adjacency matrix is determined by the signs on all edges.
    # Due to switching symmetry, we can fix the signs on a spanning tree and
    # only vary the signs on the fundamental cycles.
    
    # For a prism graph, there are 2m vertices and 3m edges.
    # A spanning tree has 2m-1 edges, so there are 3m - (2m-1) = m+1 independent cycles.
    # But due to the structure, the relevant degrees of freedom are fewer.
    
    # Strategy: Enumerate all possible sign patterns on the matching edges (2^m possibilities)
    # and for each, determine the optimal signs on cycle edges via local optimization.
    
    # Actually, let's think more carefully. The spectrum of the signed adjacency matrix
    # depends on the product of signs around each cycle.
    
    # For the prism graph, the fundamental cycles are:
    # 1. The two m-cycles (one in each layer)
    # 2. The m "ladder" cycles: (i, i+1, i+1+m, i+m, i) for each i
    
    # Let s_i be the sign on matching edge (i, i+m)
    # Let a_i be the sign on cycle edge (i, i+1) in first cycle (mod m)
    # Let b_i be the sign on cycle edge (i+m, i+1+m) in second cycle (mod m)
    
    # The product around the first m-cycle is prod(a_i)
    # The product around the second m-cycle is prod(b_i)
    # The product around ladder cycle i is a_i * s_i * b_i * s_{i+1} (with appropriate indexing)
    
    # Wait, let me be more careful with the ladder cycles.
    # Ladder cycle i: i -> i+1 -> i+1+m -> i+m -> i
    # Signs: a_i (edge i,i+1), s_{i+1} (edge i+1, i+1+m), b_i (edge i+1+m, i+m), s_i (edge i+m, i)
    # Product: a_i * s_{i+1} * b_i * s_i
    
    # The eigenvalues of the signed adjacency matrix depend on these cycle products.
    
    # Let's use a different approach: directly enumerate switching classes.
    # Fix vertex 0 to have "reference" sign +1. Then the signs on edges incident to
    # vertex 0 determine the switching class. But we need to be more systematic.
    
    # Better approach: Use the fact that for small m (4,5,7,9), we can enumerate
    # all 2^(m+1) or so switching classes and find the best one.
    
    # Let me define the sign vector more carefully.
    # I'll represent signs as +1/-1 values.
    
    # For each possible assignment of signs to the matching edges (2^m possibilities),
    # I'll find the optimal assignment to the cycle edges.
    
    # The objective is to minimize the spectral radius of the signed adjacency matrix.
    # A good proxy is the trace of A^4, which equals sum of squares of eigenvalues^4... 
    # actually trace(A^4) = sum lambda_i^4. Minimizing this helps minimize spectral radius.
    
    # Even better: I can compute the actual spectral radius for each candidate.
    
    # Let's build the signed adjacency matrix for a given sign assignment.
    
    def build_signed_adj(signs):
        """Build the signed adjacency matrix from sign vector."""
        n = 2 * m
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(A):
        """Compute spectral radius of symmetric matrix A."""
        try:
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        except:
            return np.inf
    
    def is_valid_lift(signs):
        """Check if the 2-lift is connected and 3-regular."""
        n = 2 * m
        # Build the lift graph
        # Vertices: (v, 0) and (v, 1) for v in 0..2m-1
        # Total 4m vertices
        # Edge (u,v) with sign +1: connects (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u,v) with sign -1: connects (u,0)-(v,1) and (u,1)-(v,0)
        
        # Build adjacency for the lift
        lift_adj = {}
        for v in range(2 * m):
            for side in range(2):
                lift_adj[(v, side)] = []
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # Equal sides
                lift_adj[(u, 0)].append((v, 0))
                lift_adj[(v, 0)].append((u, 0))
                lift_adj[(u, 1)].append((v, 1))
                lift_adj[(v, 1)].append((u, 1))
            else:
                # Opposite sides
                lift_adj[(u, 0)].append((v, 1))
                lift_adj[(v, 1)].append((u, 0))
                lift_adj[(u, 1)].append((v, 0))
                lift_adj[(v, 0)].append((u, 1))
        
        # Check 3-regular
        for v in range(2 * m):
            for side in range(2):
                if len(lift_adj[(v, side)]) != 3:
                    return False
        
        # Check connected via BFS
        start = (0, 0)
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in lift_adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == 4 * m
    
    # Now, enumerate switching classes.
    # A switching operation flips the sign of all edges incident to a vertex.
    # Two sign assignments are in the same switching class if one can be obtained
    # from the other by a sequence of switchings.
    
    # The number of switching classes is 2^(E - V + C) where C is the number of connected components.
    # For the prism graph (connected), E = 3m, V = 2m, so 2^(3m - 2m + 1) = 2^(m+1).
    
    # For m=9, that's 2^10 = 1024 classes. Very manageable.
    
    # To enumerate, I'll fix the signs on a spanning tree and vary the signs on
    # the non-tree edges (fundamental cycles).
    
    # Let's pick a spanning tree. For the prism graph, a natural spanning tree is:
    # - All matching edges: (i, i+m) for i=0..m-1
    # - Cycle edges in first layer: (i, i+1) for i=0..m-2 (skip the last edge)
    # - No edges in second layer (they're all non-tree)
    
    # Wait, that gives m + (m-1) = 2m-1 edges, which is correct for a spanning tree.
    
    # The non-tree edges are:
    # - The last edge of first cycle: (m-1, 0)
    # - All edges of second cycle: (i+m, i+1+m) for i=0..m-1 (with wrap-around)
    
    # So there are 1 + m = m+1 non-tree edges, giving 2^(m+1) switching classes.
    
    # For each switching class, I need to find the representative that minimizes
    # the spectral radius. But within a switching class, all representatives have
    # the same spectrum! So I just need to evaluate one representative per class.
    
    # Let me fix the signs on the spanning tree edges to +1, and vary the signs
    # on the non-tree edges.
    
    # Identify tree and non-tree edges
    tree_edges = set()
    non_tree_edges = []
    
    # Matching edges are in the tree
    for idx in matching_edges:
        tree_edges.add(idx)
    
    # Cycle edges in first layer except the last one
    # First cycle edges: (0,1), (1,2), ..., (m-2, m-1), (m-1, 0)
    # The last edge (m-1, 0) is non-tree
    for idx in cycle_edges_1:
        u, v = edges[idx]
        # Check if this is the edge (m-1, 0) or (0, m-1)
        if (u == m-1 and v == 0) or (u == 0 and v == m-1):
            non_tree_edges.append(idx)
        else:
            tree_edges.add(idx)
    
    # All second cycle edges are non-tree
    for idx in cycle_edges_2:
        non_tree_edges.append(idx)
    
    n_non_tree = len(non_tree_edges)
    # Should be m+1
    
    # Enumerate all 2^(m+1) assignments of signs to non-tree edges
    # For each, set tree edges to +1, compute spectral radius, check validity.
    
    best_radius = np.inf
    best_signs = None
    
    n_assignments = 1 << n_non_tree
    for mask in range(n_assignments):
        if time.time() > deadline:
            break
        
        signs = np.ones(n_edges, dtype=np.int8)
        
        # Set non-tree edge signs based on mask
        for i, idx in enumerate(non_tree_edges):
            if mask & (1 << i):
                signs[idx] = -1
        
        # Tree edges are already +1
        
        # Check validity
        if not is_valid_lift(signs):
            continue
        
        # Compute spectral radius
        A = build_signed_adj(signs)
        radius = spectral_radius(A)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # If we didn't find a valid lift, try a different approach
    if best_signs is None:
        # Fallback: try all +1
        signs = np.ones(n_edges, dtype=np.int8)
        if is_valid_lift(signs):
            best_signs = signs
        else:
            # Try random search
            rng = np.random.default_rng(3)
            for _ in range(10000):
                if time.time() > deadline:
                    break
                signs = rng.choice([-1, 1], size=n_edges)
                if is_valid_lift(signs):
                    A = build_signed_adj(signs)
                    radius = spectral_radius(A)
                    if radius < best_radius:
                        best_radius = radius
                        best_signs = signs.copy()
    
    if best_signs is None:
        # Last resort: return all +1
        best_signs = np.ones(n_edges, dtype=np.int8)
    
    return best_signs
# EVOLVE-BLOCK-END
