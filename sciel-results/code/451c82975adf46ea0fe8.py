import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n=26 first, then 25, etc.
    # The current program scores 2.0 (n=22). We aim for n=26 (score ~2.36).
    # We will use a combination of:
    # 1. Spectral initialization (from inspiration strategy) to provide a better starting point.
    # 2. Optimized local search with delta updates (from current program) for fast convergence.
    
    for n in [26, 25, 24, 23, 22]:
        if time.time() > deadline - 10:
            break
        
        # Attempt multiple restarts with different initializations
        found = False
        attempts = 0
        max_attempts = 5
        
        while not found and attempts < max_attempts and time.time() < deadline - 5:
            # Choose initialization strategy
            if attempts == 0:
                # Try spectral initialization
                colors = spectral_init(n)
                if colors is None:
                    colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            else:
                colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            
            result = local_search_optimized(n, colors, time_limit=max(2.0, (deadline - time.time()) / (max_attempts - attempts)))
            if result is not None:
                return result
            attempts += 1
        
        # If we didn't find a valid coloring for this n, try the next lower n
        # Note: local_search_optimized returns None if it doesn't reach 0 violations within time limit
        # But it returns the best it found? No, the current code returns None if not 0.
        # We should modify local_search_optimized to return the best result found if time runs out,
        # or just return None and let the loop try the next n.
        # The task says "Maximize n". If we can't get 26, we want 25.
        # So if local_search fails to find 0 violations for n=26, we should try n=25.
        
    # Fallback to seed
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def spectral_init(n):
    """
    Initialize colors using a spectral approach.
    We generate a random symmetric matrix, find its top 5 eigenvectors,
    and use the sign/magnitude of components to assign colors.
    This might create structures closer to C4-free than pure random.
    """
    try:
        import numpy as np
    except ImportError:
        return None
        
    # Generate a random symmetric matrix
    # Use a sparse-like structure or just dense random
    # For n up to 26, dense is fine.
    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    
    # Compute eigenvalues and eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
    except Exception:
        return None
        
    # Take the top 5 eigenvectors (largest absolute eigenvalues? or just largest?)
    # Let's take the 5 eigenvectors corresponding to the largest absolute eigenvalues
    idx = np.argsort(np.abs(eigvals))[::-1][:5]
    V = eigvecs[:, idx] # Shape (n, 5)
    
    # For each edge (i, j), assign a color based on the vector difference V[i] - V[j]
    # or similar.
    # A simple heuristic: project the edge vector onto the 5 dimensions and pick the max component?
    # Or use the sign pattern?
    # Let's try: for edge (i,j), compute d = V[i] - V[j].
    # Assign color = argmax |d_k| ? Or argmax d_k?
    # If we use argmax d_k, we are essentially doing a nearest-neighbor classification in the eigenspace.
    
    num_edges = n * (n - 1) // 2
    colors = [0] * num_edges
    
    edge_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = V[i] - V[j]
            # Normalize?
            # Pick the dimension with the largest absolute value
            c = np.argmax(np.abs(d))
            colors[edge_idx] = int(c)
            edge_idx += 1
            
    return colors

def local_search_optimized(n, initial_colors, time_limit=15):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize colors
    if initial_colors is None:
        colors = [random.randint(0, 4) for _ in range(num_edges)]
    else:
        colors = initial_colors[:]
        
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    # Precompute common neighbor counts for each color and pair of vertices
    # k[c][u][v] = number of common neighbors of u and v in color c
    k = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    def calc_k_for_color(c):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors[c][u] & neighbors[c][v]
                kc = common.bit_count()
                k[c][u][v] = kc
                k[c][v][u] = kc
                
    for c in range(5):
        calc_k_for_color(c)
        
    def total_violations():
        v = 0
        for c in range(5):
            for u in range(n):
                for v_idx in range(u + 1, n):
                    kc = k[c][u][v_idx]
                    if kc >= 2:
                        v += kc * (kc - 1) // 2
        return v
        
    current_violations = total_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    # Local search loop
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        i, j = edges[e_idx]
        
        # Calculate delta violations
        delta = 0
        
        # Effect on old_color
        for w in range(n):
            if w == i or w == j:
                continue
            # Update k for old_color
            # Check if j is in N_old(w)
            if (neighbors[old_color][w] >> j) & 1:
                old_k = k[old_color][i][w]
                new_k = old_k - 1
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][i][w] = new_k
                k[old_color][w][i] = new_k
                
            if (neighbors[old_color][w] >> i) & 1:
                old_k = k[old_color][j][w]
                new_k = old_k - 1
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][j][w] = new_k
                k[old_color][w][j] = new_k
                
        # Effect on new_color
        for w in range(n):
            if w == i or w == j:
                continue
            if (neighbors[new_color][w] >> j) & 1:
                old_k = k[new_color][i][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][i][w] = new_k
                k[new_color][w][i] = new_k
                
            if (neighbors[new_color][w] >> i) & 1:
                old_k = k[new_color][j][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][j][w] = new_k
                k[new_color][w][j] = new_k
                
        # Apply change if accepted
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations += delta
            colors[e_idx] = new_color
            update_neighbors(e_idx, old_color, new_color)
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert k changes
            for w in range(n):
                if w == i or w == j:
                    continue
                common = neighbors[old_color][i] & neighbors[old_color][w]
                k[old_color][i][w] = common.bit_count()
                k[old_color][w][i] = common.bit_count()
                
                common = neighbors[old_color][j] & neighbors[old_color][w]
                k[old_color][j][w] = common.bit_count()
                k[old_color][w][j] = common.bit_count()
                
                common = neighbors[new_color][i] & neighbors[new_color][w]
                k[new_color][i][w] = common.bit_count()
                k[new_color][w][i] = common.bit_count()
                
                common = neighbors[new_color][j] & neighbors[new_color][w]
                k[new_color][j][w] = common.bit_count()
                k[new_color][w][j] = common.bit_count()
            
            # Revert colors and neighbors
            colors[e_idx] = old_color
            update_neighbors(e_idx, new_color, old_color)
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    # If we didn't find a solution, return None to signal failure for this n
    # The caller will try a smaller n
    return None
