"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Identify edge types
    # Prism graph structure:
    # Top cycle: (0, 1), (1, 2), ..., (m-1, 0)
    # Bottom cycle: (m, m+1), ..., (2m-1, m)
    # Matching: (i, i+m)
    
    top_edges = []
    bottom_edges = []
    match_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Check if it's a cycle edge in top
            # In a simple cycle 0-1-...-m-1-0, edges are (i, i+1) and (m-1, 0)
            if (v == (u + 1) % m):
                top_edges.append(idx)
            else:
                # Should not happen for standard prism definition, but handle robustly
                top_edges.append(idx)
        elif u >= m and v >= m:
            # Bottom cycle
            u_prime = u - m
            v_prime = v - m
            if (v_prime == (u_prime + 1) % m):
                bottom_edges.append(idx)
            else:
                bottom_edges.append(idx)
        else:
            # Matching
            match_edges.append(idx)
            
    # Sort indices to maintain consistent order if needed, though we just use them as sets
    top_edges.sort()
    bottom_edges.sort()
    match_edges.sort()
    
    num_top = len(top_edges)
    num_bottom = len(bottom_edges)
    
    # We fix matching signs to +1.
    # We search over signs for top and bottom cycle edges.
    # Total free signs: num_top + num_bottom = 2m.
    
    best_signs = None
    best_radius = float('inf')
    
    # Pre-allocate sign array
    signs = np.ones(num_edges, dtype=int)
    for idx in match_edges:
        signs[idx] = 1
        
    # Iterate over all 2^(2m) combinations
    # For m=9, 2^18 = 262144 iterations.
    # We can use bit manipulation or itertools.product.
    # itertools.product is clean.
    
    from itertools import product
    
    # Generate all sign combinations for top and bottom edges
    # Signs are -1 or 1
    
    # Optimization: 
    # Connectivity condition: Product of top signs == -1 OR Product of bottom signs == -1.
    # Let's just generate and check.
    
    # To speed up, we can compute the spectral radius directly without check_lift
    # if we are confident in the connectivity check.
    # But check_lift is safer. Let's try to use it.
    # If check_lift is too slow, we switch to direct eigenvalue calc.
    
    # Let's estimate time.
    # For m=9, 262k iterations.
    # If each takes 0.1ms, that's 26 seconds. Might be tight.
    # Let's implement a fast spectral radius calculator.
    
    # The signed adjacency matrix A is symmetric.
    # We can build it once and update it, or just compute eigenvalues.
    # Building A from scratch each time is O(n^2) = 18^2 = 324 ops. Negligible.
    # Eigvalsh is O(n^3) = 18^3 = 5832 ops.
    # 262k * 5832 ~ 1.5 billion flops. NumPy is fast, but Python loop overhead is high.
    
    # Better approach:
    # Use NumPy vectorization? Hard for eigenvalues.
    # Use a compiled loop? Not available.
    # We can prune.
    
    # Let's try the direct check_lift first for smaller m, and optimized loop for larger m.
    # Actually, let's just write a tight loop.
    
    # Helper to compute radius quickly
    def get_radius_fast(signs_vec):
        # Build signed adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for idx, (u, v) in enumerate(edges):
            s = signs_vec[idx]
            A[u, v] = s
            A[v, u] = s
        # Spectral radius is max abs eigenvalue
        # For symmetric matrix, eigenvalues are real.
        # We only need the largest magnitude.
        # np.linalg.eigvalsh returns sorted eigenvalues.
        # max abs is max(abs(eig[0]), abs(eig[-1]))
        try:
            eigvals = np.linalg.eigvalsh(A)
            return max(abs(eigvals[0]), abs(eigvals[-1]))
        except Exception:
            return float('inf')

    # Connectivity check:
    # Product of signs on top cycle
    # Product of signs on bottom cycle
    # Valid if prod_top == -1 or prod_bottom == -1
    
    # We iterate over all 2^(2m) sign assignments for top and bottom edges.
    
    # For m=9, 2^18 = 262,144.
    # For m=7, 2^14 = 16,384.
    # For m=5, 2^10 = 1,024.
    # For m=4, 2^8 = 256.
    
    # Total iterations across all 4 graphs (if called separately) is manageable.
    # But sign_edges is called once per graph.
    
    # Let's use a single loop over bitmasks.
    
    total_bits = num_top + num_bottom
    
    # We will iterate from 0 to 2^total_bits - 1
    # Bit i corresponds to the i-th edge in the combined list of top and bottom edges.
    # 0 -> +1, 1 -> -1
    
    combined_edge_indices = top_edges + bottom_edges
    
    # Pre-compute products for each bit? No, just compute on the fly.
    
    # Optimization: 
    # If we flip a sign, the product changes sign.
    # We can update the product incrementally?
    # Or just compute product for each config.
    # Product of 9 or 18 elements is fast.
    
    # Let's run the search.
    
    # To handle the 60s limit, we break if time runs out.
    
    # We'll use a simple loop.
    
    # For very large m (not in this task, but good practice), we might need heuristics.
    # Here m <= 9, so exhaustive is fine.
    
    # Initialize best with all +1 (which is invalid if m>1 usually, but check_lift will reject)
    # We start with best_radius = inf.
    
    # We can speed up by noting that the spectrum depends on the "holonomies".
    # But exhaustive search is simpler and guaranteed optimal.
    
    # Let's do it.
    
    # One optimization: 
    # The problem is symmetric under flipping all signs.
    # So we can fix the first top edge sign to +1, reducing search space by half.
    # Let's do that.
    
    if num_top > 0:
        # Fix first top edge to +1
        fixed_first = combined_edge_indices[0]
        signs[fixed_first] = 1
        
        # Iterate over the remaining bits
        remaining_bits = total_bits - 1
        # The remaining edges are combined_edge_indices[1:]
        
        # We still need to check connectivity.
        
        for mask in range(1 << remaining_bits):
            if time.time() > deadline - 2:
                break
            
            # Set signs based on mask
            # Bit 0 of mask corresponds to combined_edge_indices[1]
            # ...
            # Bit remaining_bits-1 corresponds to combined_edge_indices[total_bits-1]
            
            # We can update signs in place
            
            # Calculate product of top and bottom signs
            # We need to know which bits belong to top and which to bottom.
            
            # This bit manipulation is getting complex. 
            # Given the small size, let's just use itertools.product for clarity and sufficient speed.
            # 262k iterations with itertools might be slower than bit ops, but likely < 5s.
            
            pass # We'll replace this block with the itertools version below for clarity
            
    # Let's use itertools.product directly on the combined edges, with the first one fixed.
    
    if total_bits == 0:
        # No cycle edges? Impossible for m>=4.
        return np.ones(num_edges, dtype=int)
        
    # Fix first edge in combined list to +1
    first_edge_idx = combined_edge_indices[0]
    remaining_edge_indices = combined_edge_indices[1:]
    num_remaining = len(remaining_edge_indices)
    
    # Pre-fill signs with +1 for remaining edges initially
    for idx in remaining_edge_indices:
        signs[idx] = 1
    signs[first_edge_idx] = 1
    for idx in match_edges:
        signs[idx] = 1
        
    # We need to track which of the remaining edges are in top vs bottom to compute products efficiently?
    # Or just compute product from the signs array.
    
    # To speed up product calculation, we can maintain running products.
    # But changing one bit flips the product.
    
    # Let's just compute product each time. It's O(m).
    
    # Map edge index to 'top' or 'bottom' for product calc?
    # We can just iterate through top_edges and bottom_edges.
    
    # Let's try the loop.
    
    for mask in range(1 << num_remaining):
        if time.time() > deadline - 2:
            break
            
        # Apply mask to signs
        # We can do this by iterating through bits of mask
        # But that's O(num_remaining) per iteration.
        # Total O(2^N * N). For N=17, 131k * 17 ~ 2M ops. Fast.
        
        # Actually, we can update signs incrementally.
        # But for simplicity, let's just set them.
        
        # Optimization: 
        # Instead of setting all signs, we can just compute the products and radius.
        # But we need the full signs array for radius calc.
        
        # Let's just update the signs array.
        
        # To avoid O(N) update, we can use a Gray code sequence?
        # No, let's keep it simple.
        
        # Update signs for remaining edges
        for i, idx in enumerate(remaining_edge_indices):
            if (mask >> i) & 1:
                signs[idx] = -1
            else:
                signs[idx] = 1
        
        # Check connectivity
        # Product of top signs
        prod_top = 1
        for idx in top_edges:
            prod_top *= signs[idx]
            
        prod_bot = 1
        for idx in bottom_edges:
            prod_bot *= signs[idx]
            
        if prod_top == 1 and prod_bot == 1:
            continue # Disconnected
            
        # Compute radius
        # Using fast eigvalsh
        A = np.zeros((n, n), dtype=float)
        # Fill A
        # We can pre-fill the static parts?
        # Matching edges are always +1.
        # Cycle edges change.
        
        # Let's just build A from scratch.
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
            
        try:
            eigvals = np.linalg.eigvalsh(A)
            radius = max(abs(eigvals[0]), abs(eigvals[-1]))
        except Exception:
            continue
            
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback: return all +1 (might be invalid, but better than error)
        # Or try a random valid one
        return np.ones(num_edges, dtype=int)
        
    return best_signs

# EVOLVE-BLOCK-END
