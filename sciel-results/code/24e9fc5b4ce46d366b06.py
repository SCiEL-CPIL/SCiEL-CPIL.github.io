import numpy as np
import scipy.special

def rudin_problem10() -> np.ndarray:
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in cos(theta)
    t, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(t)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    T = np.cos(Theta)
    
    W = np.tile(w[:, np.newaxis], (1, n_phi)) * (2 * np.pi / n_phi)
    W_flat = W.flatten()
    
    m_vals = np.arange(-k, k + 1)
    A = np.zeros((n_theta * n_phi, 2 * k + 1), dtype=np.float64)
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21) - np.log(4 * np.pi) + scipy.special.gammaln(11 - abs_m) - scipy.special.gammaln(11 + abs_m))
        N_m = np.exp(log_N)
        
        P_lm = scipy.special.lpmv(abs_m, k, T)
        
        if m == 0:
            Y_m = N_m * P_lm
        elif m > 0:
            Y_m = np.sqrt(2) * N_m * P_lm * np.cos(m * Phi)
        else:
            Y_m = np.sqrt(2) * N_m * P_lm * np.sin(abs_m * Phi)
            
        A[:, idx] = Y_m.flatten()
        
    dim = 2 * k + 1
    
    def objective(c):
        y = A.dot(c)
        return np.sum(W_flat * y**4)
        
    def gradient(c):
        y = A.dot(c)
        return 4.0 * A.T.dot(W_flat * y**3)
        
    def normalize(c):
        n = np.linalg.norm(c)
        if n == 0:
            return c
        return c / n
        
    # Generate candidates
    candidates = []
    # Basis vectors
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        candidates.append(e)
        candidates.append(-e)
        
    # Pairs
    for i in range(dim):
        for j in range(i + 1, dim):
            e = np.zeros(dim)
            e[i] = 1.0
            e[j] = 1.0
            candidates.append(normalize(e))
            e[i] = 1.0
            e[j] = -1.0
            candidates.append(normalize(e))
            
    # Random vectors
    rng = np.random.default_rng(42)
    for _ in range(50):
        c = rng.normal(size=dim)
        candidates.append(normalize(c))
        
    best_c = None
    best_val = np.inf
    
    # Gradient Descent with Backtracking Line Search
    for c0 in candidates:
        c = c0.copy()
        val = objective(c)
        if val < best_val:
            best_val = val
            best_c = c.copy()
            
        # Optimize from this start
        alpha = 1e-3
        for step in range(100):
            grad = gradient(c)
            c_new = c - alpha * grad
            c_new = normalize(c_new)
            
            val_new = objective(c_new)
            
            # Check if we improved
            if val_new < val:
                c = c_new
                val = val_new
                alpha = min(alpha * 1.5, 0.1) # Increase step size
                if val < best_val:
                    best_val = val
                    best_c = c.copy()
            else:
                alpha *= 0.5
                if alpha < 1e-8:
                    break
                    
    if best_c is None:
        best_c = np.ones(dim)
        best_c /= np.linalg.norm(best_c)
        
    return best_c
