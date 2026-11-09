# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    D = H_{x-y}, M = max(H_x, H_{x+y}, H_{x+2y}).
    Uses Adam optimization on logits to find a non-uniform distribution.
    """
    n = 10
    # Pre-compute projection indices
    # P is shape (10, 10), P[y, x]
    # x in 0..9, y in 0..9
    X = np.arange(n).reshape(1, -1)
    Y = np.arange(n).reshape(-1, 1)
    
    # Projection values
    # D: x - y. Range: -9 to 9. 19 bins.
    proj_D = X - Y
    # M components:
    # H0: x. Range 0..9. 10 bins.
    proj_H0 = X
    # H1: x + y. Range 0..18. 19 bins.
    proj_H1 = X + Y
    # H2: x + 2y. Range 0..27. 28 bins.
    proj_H2 = X + 2 * Y
    
    # Flatten indices for bincount
    # We need to map each cell to a bin index for each projection.
    # Using np.unique to get consistent binning
    D_vals, D_inv = np.unique(proj_D, return_inverse=True)
    H0_vals, H0_inv = np.unique(proj_H0, return_inverse=True)
    H1_vals, H1_inv = np.unique(proj_H1, return_inverse=True)
    H2_vals, H2_inv = np.unique(proj_H2, return_inverse=True)
    
    D_inv_flat = D_inv.flatten()
    H0_inv_flat = H0_inv.flatten()
    H1_inv_flat = H1_inv.flatten()
    H2_inv_flat = H2_inv.flatten()
    
    n_D = len(D_vals)
    n_H0 = len(H0_vals)
    n_H1 = len(H1_vals)
    n_H2 = len(H2_vals)
    
    # Initial state: Uniform distribution (logits = 0)
    # We optimize logits z, P = softmax(z)
    # But softmax over 100 elements can be numerically tricky if not careful.
    # Alternatively, optimize P directly with projection back to simplex?
    # Adam on logits is standard.
    
    z = np.zeros((n, n), dtype=np.float64)
    
    # Adam parameters
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    t = 0
    
    start_time = time.time()
    max_time = 590.0  # Safety margin
    
    best_P = None
    best_score = -np.inf
    
    # Helper to compute score and gradients
    def compute_score_and_grad(z):
        # Compute probabilities
        # Shift z for numerical stability
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        P = exp_z / np.sum(exp_z)
        
        # Flatten P
        P_flat = P.flatten()
        
        # Compute marginal distributions for projections
        # H_r is entropy of the distribution of values.
        # If multiple cells map to the same value, their probabilities sum.
        
        # D distribution
        p_D = np.zeros(n_D)
        np.add.at(p_D, D_inv_flat, P_flat)
        
        # H0 distribution
        p_H0 = np.zeros(n_H0)
        np.add.at(p_H0, H0_inv_flat, P_flat)
        
        # H1 distribution
        p_H1 = np.zeros(n_H1)
        np.add.at(p_H1, H1_inv_flat, P_flat)
        
        # H2 distribution
        p_H2 = np.zeros(n_H2)
        np.add.at(p_H2, H2_inv_flat, P_flat)
        
        # Compute Entropies (Shannon, bits)
        # H = - sum p log2 p, ignoring p=0
        
        def shannon_entropy(p):
            p_nonzero = p[p > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_D = shannon_entropy(p_D)
        H_H0 = shannon_entropy(p_H0)
        H_H1 = shannon_entropy(p_H1)
        H_H2 = shannon_entropy(p_H2)
        
        M = max(H_H0, H_H1, H_H2)
        
        if M < 1e-9:
            return -np.inf, np.zeros_like(z)
            
        score = H_D / M
        
        # Gradient computation
        # We want d(score)/d(z).
        # score = H_D / M
        # d(score)/d(z) = (1/M) * d(H_D)/d(z) - (H_D / M^2) * d(M)/d(z)
        
        # d(H_D)/d(z)
        # H_D = - sum_k p_D_k log2 p_D_k
        # d(H_D)/d(p_D_k) = -log2(p_D_k) - 1/ln(2)
        # d(p_D_k)/d(P_i) = 1 if cell i maps to bin k, else 0
        # d(P_i)/d(z_j) = P_i * (delta_ij - P_j)
        
        # This gradient is expensive to compute naively for every step if we do it in pure Python loops.
        # Let's use vectorized numpy operations.
        
        # Gradient of Entropy H wrt P_flat
        # dH/dP_i = -log2(p_bin(i)) - 1/ln(2)
        
        # For D:
        # p_D is the sum of P_flat over cells in bin k.
        # dH_D/dP_i = -log2(p_D[bin(i)]) - 1/ln(2)
        # Note: This assumes p_D[bin(i)] > 0. If 0, gradient is infinite, but in practice we avoid 0.
        # To be safe, we can clip p_D for log calculation.
        
        log2_inv = 1.0 / np.log(2.0)
        
        # Calculate dH/dP for each projection
        # We need the value of the marginal probability for the bin that each cell belongs to.
        
        # For D:
        # p_D_vals_per_cell = p_D[D_inv_flat]
        # dH_D_dP_flat = -np.log2(p_D_vals_per_cell) - log2_inv
        # But if p_D_vals_per_cell is 0, log2 is -inf.
        # We assume P is strictly positive due to softmax.
        
        p_D_cell = p_D[D_inv_flat]
        # Clip to avoid log(0)
        p_D_cell_safe = np.maximum(p_D_cell, 1e-15)
        dH_D_dP_flat = -np.log2(p_D_cell_safe) - log2_inv
        
        p_H0_cell = p_H0[H0_inv_flat]
        p_H0_cell_safe = np.maximum(p_H0_cell, 1e-15)
        dH_H0_dP_flat = -np.log2(p_H0_cell_safe) - log2_inv
        
        p_H1_cell = p_H1[H1_inv_flat]
        p_H1_cell_safe = np.maximum(p_H1_cell, 1e-15)
        dH_H1_dP_flat = -np.log2(p_H1_cell_safe) - log2_inv
        
        p_H2_cell = p_H2[H2_inv_flat]
        p_H2_cell_safe = np.maximum(p_H2_cell, 1e-15)
        dH_H2_dP_flat = -np.log2(p_H2_cell_safe) - log2_inv
        
        # Determine which H is the max (M)
        # If multiple are max, M is non-differentiable, but subgradient can be used.
        # We'll pick the one with the maximum value. If tie, average gradients? 
        # For simplicity, pick the argmax.
        
        # Let's compute dM/dP_flat
        # M = max(H_H0, H_H1, H_H2)
        
        # Gradient of M wrt P_flat is the gradient of the max component.
        # Let idx_max be index of max H.
        # dM_dP_flat = dH_idx_max_dP_flat
        
        # However, if there are ties, the gradient of max is a convex combination.
        # Let's find all maxes.
        H_vals = np.array([H_H0, H_H1, H_H2])
        max_H = np.max(H_vals)
        # Indices where H == max_H (with tolerance)
        is_max = np.abs(H_vals - max_H) < 1e-9
        
        # dM_dP_flat is sum of dH_k_dP_flat for k in is_max?
        # Actually, d(max(f, g))/dx = df/dx if f>g, dg/dx if g>f.
        # If f=g, it's in the subdifferential.
        # Let's just use the gradient of the single max. If tie, pick first.
        # This is a common heuristic in such optimizations.
        
        idx_max = np.argmax(H_vals)
        
        if idx_max == 0:
            dM_dP_flat = dH_H0_dP_flat
        elif idx_max == 1:
            dM_dP_flat = dH_H1_dP_flat
        else:
            dM_dP_flat = dH_H2_dP_flat
            
        # d(score)/d(P_flat)
        # score = H_D / M
        # d(score)/dP = (1/M) * dH_D_dP - (H_D / M^2) * dM_dP
        
        coeff1 = 1.0 / M
        coeff2 = H_D / (M * M)
        
        d_score_dP_flat = coeff1 * dH_D_dP_flat - coeff2 * dM_dP_flat
        
        # Now chain rule to z
        # dP_i/dz_j = P_i (delta_ij - P_j)
        # d_score/dz_j = sum_i d_score/dP_i * dP_i/dz_j
        # = sum_i d_score_dP_i * P_i * (delta_ij - P_j)
        # = P_j * (sum_i d_score_dP_i * P_i - P_j * sum_i d_score_dP_i * P_i) ... wait.
        
        # Let G_i = d_score_dP_i
        # d_score/dz_j = sum_i G_i P_i (delta_ij - P_j)
        # = G_j P_j - P_j sum_i G_i P_i
        # = P_j (G_j - sum_k G_k P_k)
        
        # Vectorized:
        # S = sum_k G_k P_k
        # d_score_dz_flat = P_flat * (d_score_dP_flat - S)
        
        S = np.sum(d_score_dP_flat * P_flat)
        d_score_dz_flat = P_flat * (d_score_dP_flat - S)
        
        # Reshape to (n, n)
        d_score_dz = d_score_dz_flat.reshape(n, n)
        
        return score, d_score_dz

    # Optimization loop
    while time.time() - start_time < max_time:
        score, grad = compute_score_and_grad(z)
        
        if not np.isfinite(score):
            # Reset or break?
            # If score is -inf, it means M was 0. Unlikely with softmax.
            break
            
        if score > best_score:
            best_score = score
            # Reconstruct P for best score
            # We need P corresponding to current z
            z_max = np.max(z)
            exp_z = np.exp(z - z_max)
            best_P = exp_z / np.sum(exp_z)
            
        t += 1
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        z -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Optional: Decay learning rate
        # lr = lr * 0.999 (very slow decay)
        
    # If best_P is None (e.g., first step failed), return uniform
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
        
    # Ensure valid output
    # Clip negatives (should be 0 from softmax, but just in case)
    best_P = np.clip(best_P, 0, None)
    
    # Normalize
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    else:
        # Fallback to uniform
        best_P = np.ones((n, n)) / (n * n)
        
    return best_P

# EVOLVE-BLOCK-END
