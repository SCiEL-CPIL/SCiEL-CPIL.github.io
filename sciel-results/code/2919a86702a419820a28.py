import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge types: top cycle, bottom cycle, matching
    top_edges = []
    bottom_edges = []
    match_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bottom_edges.append(idx)
        else:
            match_edges.append(idx)
            
    top_edges.sort()
    bottom_edges.sort()
    match_edges.sort()
    
    # Pre-allocate the full sign vector
    # We will search over all signs for all edges, but we can exploit symmetry.
    # The spectrum is invariant under global sign flip (eigenvalues -> -eigenvalues, radius same).
    # Also, if we flip signs of all edges incident to a vertex v, the spectrum is invariant?
    # Let's check: A' = D A D where D is diagonal with +/-1.
    # Eigenvalues of D A D are same as A.
    # So we have gauge freedom.
    # We can fix signs of all matching edges to +1.
    # Then we only need to search signs of top and bottom cycle edges.
    
    signs = np.ones(num_edges, dtype=int)
    for idx in match_edges:
        signs[idx] = 1
        
    # Variable edges: top and bottom cycles
    cycle_edges = top_edges + bottom_edges
    num_cycle = len(cycle_edges)
    
    # Fix first cycle edge to +1 to break global symmetry?
    # If we fix matching to +1, global symmetry is broken?
    # If we flip all signs, matching becomes -1. But we fixed them to +1.
    # So the space of gauge-equivalent classes is represented by fixing matching to +1.
    # Within this space, is there any remaining symmetry?
    # If we flip signs of all edges incident to a top vertex i, 
    # the matching edge (i, i+m) changes sign. So matching signs are no longer all +1.
    # So fixing matching to +1 breaks all gauge symmetries except the global one?
    # No, the global flip changes matching to -1, which is not in our space.
    # So we are searching over a fundamental domain.
    
    # We need to check connectivity.
    # Condition: NOT (prod_top == 1 AND prod_bottom == 1).
    
    # To make the search fast, we iterate over masks for the cycle edges.
    # For m=9, num_cycle = 18. 2^18 = 262,144 masks.
    # For each mask, we compute spectral radius.
    
    # Optimization: Pre-compute the structure.
    # We will build A and compute eigvalsh.
    
    # Pre-compute edge endpoints for fast access
    edge_u = np.array([u for u, v in edges])
    edge_v = np.array([v for u, v in edges])
    
    # Map variable edges to indices
    var_edge_indices = cycle_edges
    num_var = len(var_edge_indices)
    
    # Pre-compute bit positions for top and bottom to check connectivity quickly
    # top_var_bits: list of bit positions in the mask corresponding to top edges
    # bottom_var_bits: list of bit positions in the mask corresponding to bottom edges
    top_var_bits = []
    bottom_var_bits = []
    for i, idx in enumerate(cycle_edges):
        if idx in top_edges:
            top_var_bits.append(i)
        else:
            bottom_var_bits.append(i)
    
    best_radius = float('inf')
    best_signs = None
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Iterate over all masks
    for mask in range(1 << num_var):
        if time.time() > deadline:
            break
            
        # Assign signs to cycle edges based on mask
        # Bit i is 1 -> sign -1, bit i is 0 -> sign +1
        # We update the signs array for variable edges
        for i, idx in enumerate(var_edge_indices):
            if (mask >> i) & 1:
                signs[idx] = -1
            else:
                signs[idx] = 1
                
        # Check connectivity
        # prod_top = (-1)^(sum of bits in top_var_bits)
        # prod_bottom = (-1)^(sum of bits in bottom_var_bits)
        # Disconnected if sum_top is even AND sum_bottom is even.
        
        sum_top = 0
        for b in top_var_bits:
            sum_top += (mask >> b) & 1
            
        sum_bottom = 0
        for b in bottom_var_bits:
            sum_bottom += (mask >> b) & 1
            
        if (sum_top % 2 == 0) and (sum_bottom % 2 == 0):
            continue
            
        # Build adjacency matrix A
        # A is symmetric, size n x n
        # We can construct it using np.zeros and fill
        # Or use a pre-allocated matrix and update only changed entries?
        # Since we change many entries, it's better to rebuild or use a fast method.
        # np.zeros is fast.
        
        A = np.zeros((n, n), dtype=float)
        
        # Fill A with signs
        # A[u, v] = signs[idx] and A[v, u] = signs[idx]
        # Vectorized assignment?
        # A[edge_u, edge_v] = signs
        # A[edge_v, edge_u] = signs
        # This is very fast in NumPy.
        
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        
        # Compute spectral radius
        # eigvalsh returns sorted eigenvalues
        eigvals = np.linalg.eigvalsh(A)
        radius = max(abs(eigvals[0]), abs(eigvals[-1]))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # Fallback if no valid sign found (should not happen)
    if best_signs is None:
        # Try to find any valid sign by random search
        rng = np.random.default_rng(3)
        for _ in range(1000):
            signs = rng.choice([-1, 1], size=num_edges)
            # Check connectivity
            prod_top = 1
            prod_bottom = 1
            for idx in top_edges:
                prod_top *= signs[idx]
            for idx in bottom_edges:
                prod_bottom *= signs[idx]
            if not (prod_top == 1 and prod_bottom == 1):
                # Valid
                A = np.zeros((n, n), dtype=float)
                A[edge_u, edge_v] = signs
                A[edge_v, edge_u] = signs
                eigvals = np.linalg.eigvalsh(A)
                radius = max(abs(eigvals[0]), abs(eigvals[-1]))
                best_radius = radius
                best_signs = signs.copy()
                break
                
        if best_signs is None:
            # Emergency: return all +1 with one -1
            signs = np.ones(num_edges, dtype=int)
            if num_edges > 0:
                signs[0] = -1
            best_signs = signs
            
    return best_signs
