# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    deadline = time.time() + 580  # 600s limit, leave 20s margin

    # Build index arrays for each projection
    X, Y = np.meshgrid(np.arange(n), np.arange(n))  # X[y,x]=x, Y[y,x]=y
    # Flatten indices for each projection
    idx_x = X.ravel().astype(np.int64)              # x: 0..9
    idx_xy = (X + Y).ravel().astype(np.int64)       # x+y: 0..18
    idx_xm = (X - Y + 9).ravel().astype(np.int64)   # x-y+9: 0..18
    idx_x2y = (X + 2 * Y).ravel().astype(np.int64)  # x+2y: 0..27

    # Number of bins for each projection
    n_bins_x = 10
    n_bins_xy = 19
    n_bins_xm = 19
    n_bins_x2y = 28

    def softmax(z):
        z_shifted = z - np.max(z)
        e = np.exp(z_shifted)
        return e / e.sum()

    def compute_entropy(q):
        """Compute Shannon entropy in bits, ignoring zero probabilities."""
        mask = q > 1e-300
        if not np.any(mask):
            return 0.0
        return -np.sum(q[mask] * np.log2(q[mask]))

    def project_entropy(P_flat, idx, n_bins):
        """Compute entropy of projection onto given index."""
        q = np.zeros(n_bins)
        np.add.at(q, idx, P_flat)
        return compute_entropy(q)

    def objective(z):
        """Negative of D/M to be minimized."""
        P_flat = softmax(z)
        H_xm = project_entropy(P_flat, idx_xm, n_bins_xm)
        H_x = project_entropy(P_flat, idx_x, n_bins_x)
        H_xy = project_entropy(P_flat, idx_xy, n_bins_xy)
        H_x2y = project_entropy(P_flat, idx_x2y, n_bins_x2y)
        M = max(H_x, H_xy, H_x2y)
        if M <= 1e-9:
            return 0.0
        return -H_xm / M

    def objective_grad(z):
        """Gradient of negative D/M."""
        P_flat = softmax(z)
        H_xm = project_entropy(P_flat, idx_xm, n_bins_xm)
        H_x = project_entropy(P_flat, idx_x, n_bins_x)
        H_xy = project_entropy(P_flat, idx_xy, n_bins_xy)
        H_x2y = project_entropy(P_flat, idx_x2y, n_bins_x2y)
        M = max(H_x, H_xy, H_x2y)
        if M <= 1e-9:
            return 0.0, np.zeros_like(z)

        # Gradient of entropy w.r.t. P_flat
        # dH/dP_i = -(log2(P_i) + 1/ln(2)) for P_i > 0, else 0
        def dH_dP(P_flat, idx, n_bins):
            q = np.zeros(n_bins)
            np.add.at(q, idx, P_flat)
            dq_dP = np.zeros_like(P_flat)
            mask = q > 1e-300
            if np.any(mask):
                dH_dq = np.zeros(n_bins)
                dH_dq[mask] = -(np.log2(q[mask]) + 1.0 / np.log(2.0))
                # dq_dP: for each cell j with index idx[j], dq[idx[j]]/dP[j] = 1
                # So dH/dP[j] = dH_dq[idx[j]]
                dq_dP[:] = dH_dq[idx]
            return dq_dP

        dHxm_dP = dH_dP(P_flat, idx_xm, n_bins_xm)
        dHx_dP = dH_dP(P_flat, idx_x, n_bins_x)
        dHxy_dP = dH_dP(P_flat, idx_xy, n_bins_xy)
        dHx2y_dP = dH_dP(P_flat, idx_x2y, n_bins_x2y)

        # Determine which projection achieves M
        Hs = [H_x, H_xy, H_x2y]
        dHs = [dHx_dP, dHxy_dP, dHx2y_dP]
        max_idx = np.argmax(Hs)
        dM_dP = dHs[max_idx]

        # d(-D/M)/dP = -(dD/M - D*dM/M^2) = (-dD*M + D*dM) / M^2
        D = H_xm
        dobj_dP = (-dHxm_dP * M + D * dM_dP) / (M * M)

        # Gradient w.r.t. z: P_j = exp(z_j)/sum(exp(z))
        # dP_j/dz_k = P_j*(delta_jk - P_k)
        # So dobj/dz_k = sum_j dobj_dP_j * dP_j/dz_k
        # = sum_j dobj_dP_j * P_j * (delta_jk - P_k)
        # = P_k * (sum_j dobj_dP_j * P_j - P_k * sum_j dobj_dP_j * P_j)
        # Wait, let me redo: dobj/dz_k = sum_j dobj_dP_j * P_j * (delta_jk - P_k)
        # = P_k * sum_j dobj_dP_j * P_j - P_k * sum_j dobj_dP_j * P_j
        # That's not right. Let me be more careful.
        # dP_j/dz_k = P_j if j==k, -P_j*P_k if j!=k
        # = P_j*(delta_jk - P_k)
        # dobj/dz_k = sum_j dobj_dP_j * P_j*(delta_jk - P_k)
        # = sum_j dobj_dP_j * P_j * delta_jk - P_k * sum_j dobj_dP_j * P_j
        # = dobj_dP_k * P_k - P_k * sum_j dobj_dP_j * P_j
        # = P_k * (dobj_dP_k - sum_j dobj_dP_j * P_j)
        
        dot = np.sum(dobj_dP * P_flat)
        dobj_dz = P_flat * (dobj_dP - dot)

        return objective(z), dobj_dz

    # Start from uniform distribution (z=0 gives uniform after softmax)
    z0 = np.zeros(n * n)
    
    best_z = z0.copy()
    best_val = objective(z0)
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            z_init = z0.copy()
        else:
            # Random initialization with small perturbation
            z_init = rng.normal(0, 0.5, size=n * n)
        
        try:
            result = minimize(
                objective_grad,
                z_init,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 5000, 'gtol': 1e-8}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_z = result.x.copy()
        except Exception:
            continue
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Convert best_z to P
    P_flat = softmax(best_z)
    P = P_flat.reshape(n, n)
    
    # Ensure validity: clip and normalize
    P = np.clip(P, 0, None)
    total = P.sum()
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / (n * n)
    
    return P


# EVOLVE-BLOCK-END
