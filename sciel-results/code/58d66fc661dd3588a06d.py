import time
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)
    
    # Identify edge types
    bottom_edges = []  # edges in bottom cycle (u < m, v < m)
    top_edges = []     # edges in top cycle (u >= m, v >= m)
    match_edges = []   # matching edges (one < m, one >= m)
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_edges.append(i)
        elif u >= m and v >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)
    
    # Build adjacency structure
    # For the prism graph:
    # Bottom cycle: vertices 0..m-1, edges (i, i+1 mod m)
    # Top cycle: vertices m..2m-1, edges (m+i, m+i+1 mod m)
    # Matching: edges (i, m+i)
    
    # Key insight: Use gauge invariance of 2-lifts
    # A 2-lift is determined by signs on a spanning tree + signs on non-tree edges
    # We can fix all spanning tree edges to +1 without loss of generality
    # The remaining edges (chords) determine the "twist" of the lift
    
    # For a prism graph (two m-cycles + matching), a spanning tree can be:
    # - The matching edges (m edges) + m-1 edges from one cycle = 2m-1 edges
    # Total edges = 3m, so non-tree edges = 3m - (2m-1) = m+1
    
    # Actually, let's think more carefully. The base graph has:
    # - m edges in bottom cycle
    # - m edges in top cycle  
    # - m matching edges
    # Total: 3m edges
    
    # A spanning tree of the base graph has 2m-1 edges
    # So there are 3m - (2m-1) = m+1 non-tree edges (chords)
    
    # The gauge freedom means we can fix the signs of any spanning tree to +1
    # and only need to enumerate the 2^(m+1) possible sign assignments for chords
    
    # However, we need to be careful: the "signs" here are the edge signs in the
    # signed adjacency matrix, which directly define the lift.
    
    # Let's use a different approach: directly enumerate all valid sign patterns
    # For m <= 9, 3m <= 27, so 2^27 is too large. But with gauge fixing, 2^(m+1) <= 2^10 = 1024
    
    # Strategy: 
    # 1. Choose a spanning tree T of the base graph
    # 2. Fix all edges in T to sign +1
    # 3. Enumerate all 2^(m+1) sign assignments for the non-tree edges
    # 4. For each assignment, check if the lift is valid and compute spectral radius
    # 5. Return the best valid assignment
    
    # But wait - the gauge transformation changes the actual signs. We need to
    # understand how the signs transform under gauge changes.
    
    # Actually, for 2-lifts, the edge signs define a 1-cochain. Two sign assignments
    # that differ by a coboundary (i.e., signs[i,j] * signs[j,k] / signs[i,k] pattern)
    # give isomorphic lifts. The gauge freedom corresponds to choosing vertex signs.
    
    # Let me use a simpler direct approach: since m is small (4,5,7,9), let's
    # enumerate smartly.
    
    # For m=9: 3m=27 edges. 2^27 is too many. But we can use the structure.
    
    # Better approach: The prism graph is a "circular ladder". Let's exploit its symmetry.
    
    # Actually, let me just do exhaustive search over the non-tree edges after gauge fixing.
    
    # Step 1: Build a spanning tree
    # Use the matching edges + bottom cycle edges (except one) as the spanning tree
    # Tree edges: all m matching edges + (m-1) bottom cycle edges = 2m-1 edges
    # Non-tree edges: 1 bottom edge + m top edges = m+1 edges
    
    # Let's identify which specific edges are in the tree and which are not
    
    # Bottom cycle edges are in order: (0,1), (1,2), ..., (m-2, m-1), (m-1, 0)
    # Let's verify the actual edge ordering
    
    # For now, let's use a robust approach: 
    # 1. Find a spanning tree using BFS/DFS
    # 2. Identify non-tree edges
    # 3. Enumerate 2^(m+1) assignments
    
    # Build adjacency list for the base graph
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        edge_set.add((min(u,v), max(u,v)))
    
    # BFS to find spanning tree
    from collections import deque
    visited = [False] * n
    tree_edges = set()
    queue = deque([0])
    visited[0] = True
    
    while queue:
        u = queue.popleft()
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(ei)
                queue.append(v)
    
    # Non-tree edges
    non_tree_edges = [i for i in range(E) if i not in tree_edges]
    num_non_tree = len(non_tree_edges)
    
    # num_non_tree should be m+1 for a prism graph (3m edges, 2m-1 tree edges)
    # Let's verify: 3m - (2m-1) = m+1. Yes.
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        try:
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        except:
            return np.inf
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, np.inf
    
    best_signs = None
    best_radius = np.inf
    
    # Enumerate all 2^(num_non_tree) sign assignments for non-tree edges
    # Tree edges are fixed to +1
    
    if num_non_tree <= 20:  # Safe to enumerate
        for combo in product([-1, 1], repeat=num_non_tree):
            if time.time() > deadline - 2.0:
                break
            
            signs = np.ones(E, dtype=np.int8)
            for j, ei in enumerate(non_tree_edges):
                signs[ei] = combo[j]
            
            valid, radius = is_valid_lift(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    else:
        # Fallback: random search if enumeration is too large
        rng = np.random.default_rng(42)
        for _ in range(10000):
            if time.time() > deadline - 2.0:
                break
            
            signs = rng.choice([-1, 1], size=E).astype(np.int8)
            valid, radius = is_valid_lift(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    # If no valid lift found via enumeration, try some structured patterns
    if best_signs is None:
        # Try all +1
        signs = np.ones(E, dtype=np.int8)
        valid, radius = is_valid_lift(signs)
        if valid:
            best_radius = radius
            best_signs = signs.copy()
        else:
            # Try random
            rng = np.random.default_rng(123)
            for _ in range(1000):
                signs = rng.choice([-1, 1], size=E).astype(np.int8)
                valid, radius = is_valid_lift(signs)
                if valid:
                    best_radius = radius
                    best_signs = signs.copy()
                    break
    
    # If still nothing, use all +1 as fallback
    if best_signs is None:
        best_signs = np.ones(E, dtype=np.int8)
    
    return best_signs
