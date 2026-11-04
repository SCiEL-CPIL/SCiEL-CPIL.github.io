import numpy as np
import math

N = 150

def construct():
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def random_on_sphere(n, rng):
        v = rng.normal(0, 1, size=(n, 3))
        norms = np.linalg.norm(v, axis=1)
        v /= norms[:, None]
        return v

    def compute_energy(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        d_iu = np.sqrt(d2_iu)
        return -np.sum(np.log(d_iu))

    def compute_grad(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        
        inv_d2 = np.zeros_like(d2)
        mask = d2 > 1e-15
        inv_d2[mask] = 1.0 / d2[mask]
        np.fill_diagonal(inv_d2, 0.0)
        
        G_euclidean = inv_d2 @ P
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        G_euclidean -= sum_inv_d2[:, None] * P
        
        dots_GP = np.sum(G_euclidean * P, axis=1)
        G_riemann = G_euclidean - dots_GP[:, None] * P
        return G_riemann

    def riemannian_exp(P, V):
        v_norm = np.linalg.norm(V, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_t = np.cos(v_norm_safe)
        sin_t = np.sin(v_norm_safe) / v_norm_safe
        P_new = cos_t[:, None] * P + sin_t[:, None] * V
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    def optimize_riemannian_adam(P_init, max_iters=1000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        P = P_init.copy()
        E = compute_energy(P)
        G = compute_grad(P)
        
        # Adam state
        m = np.zeros_like(P)
        v = np.zeros_like(P)
        
        best_E = E
        best_P = P.copy()
        
        for t in range(1, max_iters + 1):
            m = beta1 * m + (1 - beta1) * G
            v = beta2 * v + (1 - beta2) * (G * G)
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Adam step in tangent space
            step = -lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Clip step norm to avoid explosion
            step_norms = np.linalg.norm(step, axis=1)
            max_norm = 0.1
            for i in range(N):
                if step_norms[i] > max_norm:
                    step[i] *= max_norm / step_norms[i]
                    
            P_new = riemannian_exp(P, step)
            E_new = compute_energy(P_new)
            
            # If energy increases significantly, we might need to adjust, 
            # but Adam is generally stable. 
            # However, for energy minimization, if it goes up, we can skip or backtrack.
            # Let's just accept it for now, Adam is self-correcting.
            # To be safe, we can check if E_new is much worse and revert.
            if E_new < E + 1e-4: # Allow small increases
                P = P_new
                E = E_new
                G = compute_grad(P)
                
                if E < best_E:
                    best_E = E
                    best_P = P.copy()
            else:
                # If it got worse, keep old P but update momentum? 
                # Standard Adam doesn't revert. 
                # Let's try a simple backtracking if it fails.
                alpha = 1.0
                while alpha > 1e-6:
                    P_trial = riemannian_exp(P, alpha * step)
                    E_trial = compute_energy(P_trial)
                    if E_trial < E:
                        P = P_trial
                        E = E_trial
                        G = compute_grad(P)
                        if E < best_E:
                            best_E = E
                            best_P = P.copy()
                        break
                    alpha *= 0.5
                # If alpha loop finishes without success, keep P as is
                # G remains the same
                pass
                
            if np.linalg.norm(G) < 1e-8:
                break
                
        return best_P, best_E

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    # Use more initializations
    initializations = [fibonacci_spiral(N)]
    for _ in range(10):
        initializations.append(random_on_sphere(N, rng))
        
    for P_init in initializations:
        P_res, E_res = optimize_riemannian_adam(P_init)
        if E_res < best_energy:
            best_energy = E_res
            P_best = P_res
            
    return P_best
