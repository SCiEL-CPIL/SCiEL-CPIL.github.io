# EVOLVE-BLOCK-START
import numpy as np

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    This implementation uses a greedy local search algorithm to iteratively improve
    the distribution by transferring small amounts of probability mass between cells.
    This avoids the constraints of parametric models.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Precompute bin indices for each projection
    # r = -1: x - y in [-9, 9] -> offset 9 -> [0, 18]
    # r = 0: x in [0, 9]
    # r = 1: x + y in [0, 18]
    # r = 2: x + 2y in [0, 27]
    
    def get_bin_indices(proj_func, min_val, max_val):
        indices = {}
        for val in range(min_val, max_val + 1):
            mask = (proj_func(X, Y) == val)
            indices[val] = np.argwhere(mask)
        return indices

    idx_neg1 = get_bin_indices(lambda X, Y: X - Y, -9, 9)
    idx_0 = get_bin_indices(lambda X, Y: X, 0, 9)
    idx_1 = get_bin_indices(lambda X, Y: X + Y, 0, 18)
    idx_2 = get_bin_indices(lambda X, Y: X + 2*Y, 0, 27)

    def compute_entropies(P):
        # P is 10x10
        # Calculate H for each projection
        # We sum P over the indices for each bin
        
        # H_{-1}
        probs_neg1 = np.zeros(19)
        for val, indices in idx_neg1.items():
            if len(indices) > 0:
                probs_neg1[val + 9] = np.sum(P[indices[:, 0], indices[:, 1]])
        
        # H_0
        probs_0 = np.zeros(10)
        for val, indices in idx_0.items():
            if len(indices) > 0:
                probs_0[val] = np.sum(P[indices[:, 0], indices[:, 1]])
        
        # H_1
        probs_1 = np.zeros(19)
        for val, indices in idx_1.items():
            if len(indices) > 0:
                probs_1[val] = np.sum(P[indices[:, 0], indices[:, 1]])
        
        # H_2
        probs_2 = np.zeros(28)
        for val, indices in idx_2.items():
            if len(indices) > 0:
                probs_2[val] = np.sum(P[indices[:, 0], indices[:, 1]])
        
        def ent(p):
            p = p[p > 1e-15]
            if len(p) == 0: return 0.0
            return -np.sum(p * np.log2(p))
            
        D = ent(probs_neg1)
        M = max(ent(probs_0), ent(probs_1), ent(probs_2))
        
        if M < 1e-9:
            return 0.0, 0.0
        return D, M

    # Initialize P with a Gaussian that has negative correlation
    # This provides a good starting point for the greedy search
    mu_x, mu_y = 4.5, 4.5
    sigma_x, sigma_y = 1.5, 0.75
    rho = -0.9
    cov = np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])
    inv_cov = np.linalg.inv(cov)
    dx = X - mu_x
    dy = Y - mu_y
    z = np.stack([dx, dy], axis=-1)
    d2 = np.einsum('...i,ij,...j->...', z, inv_cov, z)
    P = np.exp(-0.5 * d2)
    P = P / P.sum()
    
    best_D, best_M = compute_entropies(P)
    best_ratio = best_D / best_M if best_M > 0 else 0
    best_P = P.copy()
    
    # Greedy Search Parameters
    num_iterations = 20000
    step_size = 0.01
    
    for it in range(num_iterations):
        # Randomly select two cells to swap mass
        # Flatten P for easy indexing
        P_flat = P.flatten()
        
        i = np.random.randint(0, n * n)
        j = np.random.randint(0, n * n)
        
        if i == j:
            continue
            
        # Try moving mass from i to j
        # Ensure i has enough mass
        if P_flat[i] < step_size:
            continue
            
        P_new = P.copy()
        P_flat_new = P_new.flatten()
        
        # Move mass
        P_flat_new[i] -= step_size
        P_flat_new[j] += step_size
        
        P_new = P_flat_new.reshape((n, n))
        
        D_new, M_new = compute_entropies(P_new)
        ratio_new = D_new / M_new if M_new > 0 else 0
        
        if ratio_new > best_ratio:
            best_ratio = ratio_new
            best_P = P_new
            P = P_new
            best_D = D_new
            best_M = M_new
        else:
            # Optional: Try reverse move? Or just discard.
            # Discarding is standard hill climbing.
            pass
            
    return best_P

# EVOLVE-BLOCK-END
