import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge types
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
            
    # Sort to ensure consistent ordering for cycle products
    top_cycle_edges.sort()
    bottom_cycle_edges.sort()
    matching_edges.sort()
    
    # Gauge fixing:
    # Set all matching edges to +1.
    # Set the first top cycle edge to +1.
    # This is valid because spectral radius is invariant under gauge transformations.
    # We must ensure the resulting lift is connected.
    
    fixed_signs = np.ones(num_edges)
    for i in matching_edges:
        fixed_signs[i] = 1
    if top_cycle_edges:
        fixed_signs[top_cycle_edges[0]] = 1
        
    # Free edges:
    # Remaining top cycle edges (m-1)
    # All bottom cycle edges (m)
    free_top_edges = top_cycle_edges[1:]
    free_bot_edges = bottom_cycle_edges
    
    free_edges = free_top_edges + free_bot_edges
    num_free = len(free_edges)
    
    best_signs = None
    best_radius = float('inf')
    
    # Pre-compute indices for fast sign assignment
    # We will iterate through masks
    
    # Optimization: 
    # Instead of checking connectivity via cycle products every time, 
    # we can just build the lift and check, but cycle product is O(m).
    # m is small (<=9).
    
    # We can also pre-compute the "base" matrix S_base with zeros, 
    # and just set the values.
    
    # Let's try to vectorize or optimize the inner loop.
    # For m=9, num_free = 17. 2^17 = 131072 iterations.
    # Each iteration:
    # 1. Update signs (O(num_free) or just unpack mask)
    # 2. Check connectivity (O(m))
    # 3. Build S (O(n^2) or O(E))
    # 4. eigvalsh (O(n^3) ~ 18^3 = 5832 ops)
    
    # 131072 * (small overhead + 5832 ops) is very fast in Python if we use NumPy for eigvalsh.
    # The overhead of Python loops might be significant.
    # 131072 iterations in Python ~ 0.1-0.5s per million? No, Python is slower.
    # 10^5 iterations with NumPy calls inside might take 1-2 seconds.
    # 4 graphs * 2s = 8s. Safe.
    
    for mask in range(1 << num_free):
        if time.time() > deadline:
            break
            
        # Construct signs
        signs = fixed_signs.copy()
        
        # Assign free signs
        # We can do this more efficiently by just tracking the bits
        # But for simplicity and safety, we copy and set.
        
        # To speed up, we can pre-allocate the S matrix and update it?
        # No, S depends on signs.
        
        # Let's just build S directly from mask bits.
        # This avoids the intermediate `signs` array creation if we are careful,
        # but `signs` is needed for the return value and connectivity check.
        
        # Faster approach:
        # 1. Determine signs for free edges from mask.
        # 2. Check connectivity.
        # 3. Build S.
        
        # Extract signs for free edges
        # free_edges list contains indices into the `edges` list.
        
        # We can pre-compute the "contribution" of each free edge to the cycle products.
        # But simpler: just compute the product from the constructed signs.
        
        # Let's try to minimize Python overhead.
        
        # We'll create the signs array
        # This is O(num_edges)
        
        # If we want to be very fast, we can pre-compute the S matrix for each mask? No.
        
        # Let's just do it straightforwardly.
        
        # Assign signs
        for j, idx in enumerate(free_edges):
            if (mask >> j) & 1:
                signs[idx] = -1
            else:
                signs[idx] = 1
                
        # Check Connectivity
        # Product of top cycle
        prod_top = 1
        for idx in top_cycle_edges:
            prod_top *= signs[idx]
            
        # Product of bottom cycle
        prod_bot = 1
        for idx in bottom_cycle_edges:
            prod_bot *= signs[idx]
            
        if prod_top == 1 and prod_bot == 1:
            continue # Disconnected
            
        # Build S
        S = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            S[u, v] = s
            S[v, u] = s
            
        # Compute Spectral Radius
        # S is symmetric
        eigvals = np.linalg.eigvalsh(S)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback: Random search
        rng = np.random.default_rng(3)
        for _ in range(budget):
            signs = rng.choice([-1, 1], size=num_edges)
            # Check connectivity
            # Re-use logic?
            # For fallback, just try and catch?
            # Or just ensure at least one cycle is -1.
            # Simple random might hit disconnected.
            # We can force one edge to -1 to ensure connectivity?
            # If we flip one edge in a cycle, the product becomes -1 (if it was +1).
            # So if we pick a random sign vector, and it's disconnected, we can flip a random edge to fix it?
            # Not necessarily, but likely.
            
            # Let's just do a quick connectivity check similar to above.
            prod_top = 1
            for idx in top_cycle_edges:
                prod_top *= signs[idx]
            prod_bot = 1
            for idx in bottom_cycle_edges:
                prod_bot *= signs[idx]
                
            if prod_top == 1 and prod_bot == 1:
                # Flip a random top edge to force connectivity
                if top_cycle_edges:
                    idx = top_cycle_edges[0]
                    signs[idx] *= -1
                    prod_top *= -1
                    # Now prod_top is -1, so connected.
                    
            S = np.zeros((n, n))
            for i, (u, v) in enumerate(edges):
                s = signs[i]
                S[u, v] = s
                S[v, u] = s
            eigvals = np.linalg.eigvalsh(S)
            radius = np.max(np.abs(eigvals))
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                
    return best_signs
