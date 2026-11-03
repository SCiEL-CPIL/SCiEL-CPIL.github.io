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
        return -np.sum(np.log(np.sqrt(d2_iu)))
    
    def compute_grad(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2_safe = np.where(d2 > 1e-15, d2, 1e-15)
        
        diff = P[None, :, :] - P[:, None, :]
        terms = diff / d2_safe[:, :, None]
        G = np.sum(terms, axis=1)
        
        dots_GP = np.einsum('ij,ij->i', G, P)
        G_proj = G - dots_GP[:, None] * P
        return G_proj

    def riemannian_exp(P, V):
        v_norm = np.linalg.norm(V, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_t = np.cos(v_norm_safe)
        sin_t = np.sin(v_norm_safe) / v_norm_safe
        P_new = cos_t[:, None] * P + sin_t[:, None] * V
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    def riemannian_inner(A, B):
        return np.sum(A * B, axis=1)

    def riemannian_trans(X, Y, V):
        # Transport V from X to Y
        # For sphere, use the formula:
        # V_trans = (1 - (1-cos(theta))/theta^2) V + ((1-cos(theta))/theta) (X x Y) x V ? 
        # Simpler: use the exponential map derivative approximation or just project.
        # For small steps, parallel transport is close to projection.
        # Let's use the standard Riemannian CG transport formula:
        # T_{X->Y}(V) = V - <V, X>Y + <V, Y>X ... no that's not right.
        # Let's use the retraction-based transport: T_{X->Y}(V) = (I - YY^T) V + Y <V, X> ? 
        # Actually, a common simple transport is:
        # V_new = V - <V, X> Y + <V, Y> X is incorrect.
        # Correct transport on sphere:
        # If Y = exp_X(V), then T_{X->Y}(W) = W + (1 - 1/sin(theta)) <W, V> V ? 
        # Let's use a simpler approximation for CG: 
        # The "standard" RCG uses a transport map. 
        # For small steps, we can approximate transport by parallel transport.
        # Parallel transport of V from X to Y:
        # Let u = X x Y. theta = angle(X,Y).
        # T(V) = V + (1 - cos(theta)/theta) (u x V) + (1 - 1/sin(theta)) <V, X> Y ... this is complex.
        # Given the risk of bugs, let's stick to a robust Riemannian Gradient Descent with Adaptive Step Size (Barzilai-Borwein).
        # BB step size: alpha_k = <s_{k-1}, y_{k-1}> / <s_{k-1}, s_{k-1}> or <s_{k-1}, y_{k-1}> / <y_{k-1}, y_{k-1}>.
        # This is much safer to implement correctly than RCG transport.
        pass

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(12345)
    
    initializations = []
    initializations.append(fibonacci_spiral(N))
    for r in range(4):
        initializations.append(random_on_sphere(N, rng))
    
    for init_idx, P_init in enumerate(initializations):
        P_cur = P_init.copy()
        E_cur = compute_energy(P_cur)
        G_cur = compute_grad(P_cur)
        
        if E_cur < best_energy:
            best_energy = E_cur
            P_best = P_cur.copy()
        
        eta = 1e-2
        # Barzilai-Borwein state
        s_prev = None
        y_prev = None
        
        max_iters = 5000
        min_eta = 1e-10
        max_eta = 1.0
        patience = 0
        best_in_restart = E_cur
        P_best_restart = P_cur.copy()
        
        for it in range(max_iters):
            G_cur = compute_grad(P_cur)
            grad_norm = np.linalg.norm(G_cur)
            
            if grad_norm < 1e-10:
                break
            
            # Determine step size
            if s_prev is not None and y_prev is not None:
                # BB step sizes
                # s = P_k - P_{k-1} (in tangent space approx)
                # y = G_k - G_{k-1} (in tangent space approx)
                # We need to transport gradients to a common point to compute y properly.
                # For simplicity and robustness, we can use the Euclidean difference of projected gradients as an approximation for BB.
                # Or just use a fixed adaptive rule.
                # Let's use a simple adaptive rule: if energy decreased, increase eta, else decrease.
                pass
            
            # Simple adaptive step size with line search
            improved = False
            for ls_iter in range(30):
                V = -eta * G_cur
                P_trial = riemannian_exp(P_cur, V)
                E_trial = compute_energy(P_trial)
                
                if E_trial < E_cur:
                    best_in_restart = min(best_in_restart, E_trial)
                    if E_trial < best_in_restart:
                         P_best_restart = P_trial.copy()
                    P_cur = P_trial
                    E_cur = E_trial
                    eta = min(eta * 1.1, max_eta)
                    improved = True
                    patience = 0
                    break
                else:
                    eta *= 0.8
            
            if not improved:
                patience += 1
                if patience > 50:
                    break
            
            if E_cur < best_energy:
                best_energy = E_cur
                P_best = P_cur.copy()
        
        if best_in_restart < best_energy:
            best_energy = best_in_restart
            P_best = P_best_restart.copy()
    
    return P_best
