import numpy as np
import time

def _greedy_decode(adjacency, scores):
    """Fixed greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    picked = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if forbidden[v]:
            continue
        picked[v] = True
        count += 1
        # Mark neighbours as forbidden
        neighbors = np.where(adjacency[v])[0]
        forbidden[neighbors] = True
    return count

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # 1. Compute Degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # 2. Compute Average Neighbor Degree
    # Sum of degrees of neighbors
    nbr_deg_sums = adj @ degrees
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)
        
    # 3. Reaction-Diffusion Score Computation
    # Model vertex priorities as a continuous-time mean-field game where each vertex's
    # score evolves via a reaction-diffusion process on the graph. The source term favors
    # low-degree nodes (they're better candidates for independent set), and the diffusion
    # term smooths scores across the graph to capture global structural independence.
    #
    # The PDE: ∂u/∂t = D * Δu + r * (S - u)
    # where Δ is the graph Laplacian, S is the source term (favoring low degree),
    # D is diffusion coefficient, r is reaction rate.
    
    # Initialize with source term: favor low-degree vertices
    # Normalize degrees to [0, 1]
    deg_min = degrees.min()
    deg_max = degrees.max()
    if deg_max > deg_min:
        norm_deg = (degrees - deg_min) / (deg_max - deg_min)
    else:
        norm_deg = np.zeros(n)
    
    # Source term: high score for low-degree vertices
    source = 1.0 - norm_deg
    
    # Graph Laplacian: L = D - A
    # For diffusion, we use the normalized Laplacian to avoid scale issues
    # L_sym = I - D^{-1/2} A D^{-1/2}
    # But for simplicity and stability, use L = D - A
    
    D_mat = np.diag(degrees)
    L = D_mat - adj.astype(float)
    
    # Run reaction-diffusion for a fixed number of steps
    # u_{t+1} = u_t + dt * (D * L @ u_t + r * (source - u_t))
    # With D=0.1, r=0.5, dt=0.1, we get stable convergence
    
    D_coeff = 0.1  # Diffusion coefficient
    r_coeff = 0.5  # Reaction rate
    dt = 0.1       # Time step
    num_steps = 50  # Number of diffusion steps
    
    u = source.copy()
    try:
        if time.time() < deadline - 10:
            for _ in range(num_steps):
                if time.time() > deadline - 5:
                    break
                # Diffusion term: D * L @ u
                # Note: L @ u = (D - A) @ u = D*u - A@u
                # So D_coeff * L @ u = D_coeff * (D*u - A@u)
                diffusion = D_coeff * (degrees * u - adj.astype(float) @ u)
                # Reaction term: r * (source - u)
                reaction = r_coeff * (source - u)
                # Update
                u = u + dt * (diffusion + reaction)
        else:
            u = source
    except Exception:
        u = source
    
    # Normalize u for stability
    u_max = np.max(np.abs(u))
    if u_max > 0:
        v1 = u / u_max
    else:
        v1 = np.zeros(n)
        
    # 4. Combine Scores
    # Base: Low degree is good (standard greedy heuristic)
    # Bonus: High avg neighbor degree is good (removes hubs by prioritizing their low-degree neighbors? 
    #        Actually, high avg_nbr_deg means the vertex is connected to hubs. 
    #        In independent set, we usually want to avoid hubs. 
    #        If we pick a vertex with high avg_nbr_deg, we block many hubs. This is good.)
    # Reaction-Diffusion: Global structural independence from PDE solution
    
    # Weights
    w_deg = 1.0
    w_nbr = 1.0
    w_rd = 0.8  # Weight for reaction-diffusion component
    
    scores = -w_deg * degrees + w_nbr * avg_nbr_deg + w_rd * v1
    
    # 5. Local Polish: try swapping a few vertices to improve the greedy result
    # We can't change the decoder, but we can adjust scores to change the order.
    # Let's evaluate the current greedy result.
    best_count = _greedy_decode(adj, scores)
    best_scores = scores.copy()
    
    # Try small perturbations: for each vertex, try increasing/decreasing its score
    # slightly to see if it improves the greedy result.
    # This is expensive for 200 vertices, so do a limited number of iterations.
    max_polish_iters = 15
    for _ in range(max_polish_iters):
        if time.time() > deadline - 2:
            break
        improved = False
        # Try swapping pairs of vertices in the score order
        # This is O(n^2) per iteration, which is 40000 for n=200. Acceptable.
        for i in range(n):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 2:
                    break
                # Swap scores of i and j
                scores[i], scores[j] = scores[j], scores[i]
                count = _greedy_decode(adj, scores)
                if count > best_count:
                    best_count = count
                    best_scores = scores.copy()
                    improved = True
                else:
                    # Revert
                    scores[i], scores[j] = scores[j], scores[i]
        if not improved:
            break
    
    return best_scores
