"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge types
    top_cycle = []    # edges in top cycle (0..m-1)
    bottom_cycle = [] # edges in bottom cycle (m..2m-1)
    matching = []     # matching edges (i, i+m)
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle.append(i)
        elif u >= m and v >= m:
            bottom_cycle.append(i)
        else:
            matching.append(i)
    
    # Build signed adjacency matrix efficiently
    def build_signed_adj(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        # Use union-find for efficiency
        parent = list(range(2 * n))
        rank = [0] * (2 * n)
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                union(u, v)
                union(u + n, v + n)
            else:
                union(u, v + n)
                union(u + n, v)
        
        # Check all vertices are in the same component
        root = find(0)
        for i in range(1, 2 * n):
            if find(i) != root:
                return False
        return True
    
    # Key insight: For prism graphs, we can exploit the structure.
    # The signed adjacency matrix has a block structure.
    # Let's use a more efficient local search with better initialization.
    
    rng = np.random.default_rng(42)
    
    best_radius = float('inf')
    best_signs = np.ones(num_edges, dtype=np.int8)
    
    # Try multiple strategies
    # Strategy 1: Start with patterns that are likely to have low spectral radius
    # For a 3-regular graph, the minimum possible spectral radius is related to the 
    # smallest eigenvalue of certain matrices.
    
    # Known good patterns for prism graphs:
    # - Alternating signs on cycles
    # - Signs that make the signed graph have eigenvalues close to 0
    
    # Let's try structured initializations
    initializations = []
    
    # All -1 (works for odd m)
    if m % 2 == 1:
        initializations.append(np.full(num_edges, -1, dtype=np.int8))
    
    # Alternating on top cycle
    for pattern_name in ['alt_top', 'alt_bottom', 'alt_match', 'mixed']:
        signs = np.ones(num_edges, dtype=np.int8)
        if pattern_name == 'alt_top':
            for j, idx in enumerate(top_cycle):
                signs[idx] = -1 if j % 2 == 0 else 1
        elif pattern_name == 'alt_bottom':
            for j, idx in enumerate(bottom_cycle):
                signs[idx] = -1 if j % 2 == 0 else 1
        elif pattern_name == 'alt_match':
            for j, idx in enumerate(matching):
                signs[idx] = -1 if j % 2 == 0 else 1
        elif pattern_name == 'mixed':
            # Mix of alternating patterns
            for j, idx in enumerate(top_cycle):
                signs[idx] = -1 if j % 2 == 0 else 1
            for j, idx in enumerate(bottom_cycle):
                signs[idx] = 1 if j % 2 == 0 else -1
            for j, idx in enumerate(matching):
                signs[idx] = -1 if j % 3 == 0 else 1
        initializations.append(signs)
    
    # Add random initializations
    for _ in range(10):
        initializations.append(rng.choice([-1, 1], size=num_edges).astype(np.int8))
    
    # Local search with simulated annealing
    for init_idx, init_signs in enumerate(initializations):
        if time.time() > deadline:
            break
        
        curr = init_signs.copy()
        
        # Ensure connectivity
        if not is_connected_lift(curr):
            # Try to fix by flipping edges
            fixed = False
            for _ in range(50):
                if is_connected_lift(curr):
                    fixed = True
                    break
                idx = rng.integers(0, num_edges)
                curr[idx] *= -1
            if not fixed:
                continue
        
        curr_rad = spectral_radius(curr)
        
        # Simulated annealing
        T = 0.5
        cooling = 0.995
        
        for step in range(2000):
            if time.time() > deadline:
                break
            
            if curr_rad < best_radius:
                best_radius = curr_rad
                best_signs = curr.copy()
            
            # Pick a random edge to flip
            idx = rng.integers(0, num_edges)
            new_signs = curr.copy()
            new_signs[idx] *= -1
            
            if not is_connected_lift(new_signs):
                continue
            
            new_rad = spectral_radius(new_signs)
            delta = new_rad - curr_rad
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or rng.random() < np.exp(-delta / T):
                curr = new_signs
                curr_rad = new_rad
            
            T *= cooling
    
    # Final verification and fallback
    if not is_connected_lift(best_signs):
        # Fallback: try simple patterns
        for i in range(num_edges):
            signs = np.full(num_edges, -1, dtype=np.int8)
            signs[i] = 1
            if is_connected_lift(signs):
                best_signs = signs
                break
    
    return best_signs
# EVOLVE-BLOCK-END
