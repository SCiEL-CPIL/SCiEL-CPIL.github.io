import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    n = 24
    # Precompute roots of unity for evaluation
    # We use a dense grid for the search phase
    num_points = 2048
    angles = 2 * np.pi * np.arange(num_points) / num_points
    z = np.exp(1j * angles)
    
    # Function to compute sup norm on the grid
    def sup_norm(c):
        # c is a numpy array of 1s and -1s
        # P(z) = sum c_k z^k
        # We can compute this via convolution or direct matrix multiplication
        # Direct matrix multiplication is fast enough for 24x2048
        # Z_matrix[i, k] = z_i^k
        # We can precompute Z_matrix
        pass

    # Precompute the evaluation matrix Z
    # Z[i, k] = exp(i * theta_i * k)
    # theta_i = 2 * pi * i / num_points
    # Z = exp(1j * np.outer(angles, np.arange(n)))
    Z = np.exp(1j * np.outer(angles, np.arange(n)))
    
    def eval_sup_norm(c):
        # c: np.array of shape (n,) with values +/-1
        vals = Z @ c
        return np.max(np.abs(vals))
        
    def high_precision_sup_norm(c):
        # For final polish, use more points or local refinement
        # The task says "dense sampling and golden-section refinement"
        # For now, a very dense grid (e.g., 100k points) is a good proxy
        # 100k points * 24 coeffs = 2.4M ops, very fast in numpy
        num_hp = 100000
        angles_hp = 2 * np.pi * np.arange(num_hp) / num_hp
        z_hp = np.exp(1j * angles_hp)
        Z_hp = np.exp(1j * np.outer(angles_hp, np.arange(n)))
        vals = Z_hp @ c
        return np.max(np.abs(vals))

    best_c = None
    best_score = float('inf')
    
    # Initial candidates
    # 1. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    c_golay = np.array(p[:n])
    
    # 2. All +1 (worst case, but valid start)
    c_all_plus = np.ones(n)
    
    # 3. Random starts
    candidates = [c_golay, c_all_plus]
    for _ in range(50):
        candidates.append(np.random.choice([1, -1], size=n))
        
    # Local Search (Hill Climbing with Random Restarts)
    # We will iterate through candidates and improve them
    
    for c0 in candidates:
        c = c0.copy()
        current_score = eval_sup_norm(c)
        
        # Simple Hill Climbing
        improved = True
        while improved:
            improved = False
            # Try flipping each bit
            for k in range(n):
                c_flip = c.copy()
                c_flip[k] *= -1
                score_flip = eval_sup_norm(c_flip)
                if score_flip < current_score - 1e-12:
                    c = c_flip
                    current_score = score_flip
                    improved = True
                    # Early exit if we hit a very good score? 
                    # The optimum is ~6.152. Let's just keep going until local min.
                    
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()
            
        if time.time() > deadline - 5.0:
            break
            
    # If we have time, do a more sophisticated search like Simulated Annealing or Tabu
    # But for N=24, exhaustive search of neighborhoods is often enough if we have enough restarts.
    # Let's add a random walk phase if time permits.
    
    if best_c is None:
        best_c = c_golay
        
    # Final Polish:
    # The grid sup norm might be slightly higher than the true sup norm.
    # We should check the high precision score of the best candidate.
    # Also, we might want to try flipping bits again with high precision evaluation
    # to see if the "local minimum" on the coarse grid is actually not a minimum on the fine grid.
    
    # Refine best_c using high precision
    hp_score = high_precision_sup_norm(best_c)
    
    # Local search with high precision
    c_hp = best_c.copy()
    improved_hp = True
    while improved_hp and time.time() < deadline - 2.0:
        improved_hp = False
        for k in range(n):
            c_flip = c_hp.copy()
            c_flip[k] *= -1
            score_flip = high_precision_sup_norm(c_flip)
            if score_flip < hp_score - 1e-12:
                c_hp = c_flip
                hp_score = score_flip
                improved_hp = True
                if time.time() > deadline - 2.0:
                    break
        if time.time() > deadline - 2.0:
            break
            
    best_c = c_hp
    
    return list(best_c)
