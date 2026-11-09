import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _compute_entropies(P):
    # P is 10x10
    # x, y in 0..9
    # x - y: -9..9 (19 bins)
    # x: 0..9 (10 bins)
    # x + y: 0..18 (19 bins)
    # x + 2y: 0..27 (28 bins)
    
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    
    # Helper to get entropy from projected values
    def proj_entropy(vals, n_bins, offset=0):
        # vals is 10x10 array of integer indices
        # counts is 1D array of size n_bins
        counts = np.zeros(n_bins)
        # np.add.at is slow, use bincount if possible? 
        # vals are integers.
        # Flatten P and vals
        p_flat = P.flatten()
        v_flat = (vals + offset).flatten()
        # bincount requires non-negative ints
        counts = np.bincount(v_flat, weights=p_flat, minlength=n_bins)
        # Entropy
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        if len(probs) == 0: return 0.0
        return -np.sum(probs * np.log2(probs))
        
    H_d = proj_entropy(X - Y, 19, 9)
    H_0 = proj_entropy(X, 10, 0)
    H_1 = proj_entropy(X + Y, 19, 0)
    H_2 = proj_entropy(X + 2*Y, 28, 0)
    
    return H_d, H_0, H_1, H_2

def _compute_grads(P):
    # Returns gradients of H_d, H_0, H_1, H_2 w.r.t P
    # grad_H_r[y,x] = -1/ln(2) * (log2(Q_r[x+ry]) + 1)
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    
    def proj_grad(vals, n_bins, offset=0):
        counts = np.zeros(n_bins)
        p_flat = P.flatten()
        v_flat = (vals + offset).flatten()
        counts = np.bincount(v_flat, weights=p_flat, minlength=n_bins)
        # Avoid log(0)
        counts_safe = np.where(counts > 0, counts, 1.0)
        log_probs = np.log2(counts_safe)
        # grad for each cell (y,x) is -1/ln2 * (log2(count[x+ry]) + 1)
        # We need to map back to 10x10
        # cell (y,x) has value v = vals[y,x]
        # index in counts is v + offset
        # grad[y,x] = -1/ln2 * (log_probs[v+offset] + 1)
        idx = (vals + offset)
        grad_vals = -1.0/np.log(2) * (log_probs[idx] + 1.0)
        return grad_vals

    G_d = proj_grad(X - Y, 19, 9)
    G_0 = proj_grad(X, 10, 0)
    G_1 = proj_grad(X + Y, 19, 0)
    G_2 = proj_grad(X + 2*Y, 28, 0)
    
    return G_d, G_0, G_1, G_2

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600 # Safety margin
    
    # 1. Differential Evolution
    # Variables: z (100 values)
    # P = softmax(z)
    
    def softmax(z):
        z = z - np.max(z) # for stability
        e = np.exp(z)
        return e / np.sum(e)

    def objective_de(z):
        P = softmax(z).reshape(10, 10)
        H_d, H_0, H_1, H_2 = _compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        return H_d / M

    # DE on z in [-10, 10]
    # Uniform P corresponds to z=0.
    bounds = [(-10, 10)] * 100
    
    try:
        # Use DE to find a good starting point
        res_de = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-4,
            disp=False
        )
        z_init = res_de.x
    except Exception:
        z_init = np.zeros(100)
        
    # Check time
    if time.time() > deadline - 60:
        P = softmax(z_init).reshape(10, 10)
        P = np.clip(P, 0, None)
        P /= P.sum()
        return P

    # 2. L-BFGS-B Refinement
    def objective_lbgfs(z):
        P = softmax(z).reshape(10, 10)
        H_d, H_0, H_1, H_2 = _compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        return - (H_d / M) # Minimize negative ratio

    def gradient_lbgfs(z):
        P = softmax(z).reshape(10, 10)
        H_d, H_0, H_1, H_2 = _compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return np.zeros(100)
        
        G_d, G_0, G_1, G_2 = _compute_grads(P)
        
        # Determine which H_k is max
        # If tie, pick first? Or average?
        # For sub-gradient, any is valid.
        Hs = [H_0, H_1, H_2]
        Gs = [G_0, G_1, G_2]
        max_idx = np.argmax(Hs)
        G_m = Gs[max_idx]
        
        # grad_P f = (M * grad_P D - D * grad_P M) / M^2
        # f = D/M
        # grad f = (M * grad D - D * grad M) / M^2
        grad_P_f = (M * G_d - H_d * G_m) / (M * M)
        
        # Chain rule to z
        # P = softmax(z)
        # dP_i/dz_j = P_i (delta_ij - P_j)
        # grad_z f = sum_i (dP_i/dz_j * d f / d P_i)
        # = sum_i P_i (delta_ij - P_j) grad_P_f[i]
        # = P_j grad_P_f[j] - P_j sum_i P_i grad_P_f[i]
        # = P (grad_P_f - sum(P * grad_P_f))
        
        P_flat = P.flatten()
        grad_P_flat = grad_P_f.flatten()
        
        dot = np.dot(P_flat, grad_P_flat)
        grad_z = P_flat * (grad_P_flat - dot)
        
        return grad_z

    try:
        res_lbgfs = minimize(
            objective_lbgfs, 
            z_init, 
            method='L-BFGS-B', 
            jac=gradient_lbgfs,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        z_final = res_lbgfs.x
    except Exception:
        z_final = z_init

    # Final Check
    if time.time() > deadline - 10:
        # Fallback to best found so far
        pass

    P = softmax(z_final).reshape(10, 10)
    P = np.clip(P, 0, None)
    total = P.sum()
    if total <= 1e-9:
        return np.ones((10,10))/100.0
    P /= total
    
    # Verify validity
    if not np.all(np.isfinite(P)):
        return np.ones((10,10))/100.0
    if np.any(P < -1e-9):
        P = np.clip(P, 0, None)
        P /= P.sum()
        
    return P
