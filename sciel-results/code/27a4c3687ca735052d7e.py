"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Precompute pair indices
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)

    def _energy_and_grad(X):
        """Compute log energy and Euclidean gradient (not yet projected)."""
        diff = X[idx_i] - X[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-12)
        energy = -np.sum(np.log(dist))
        
        grad_euclidean = np.zeros((N, 3))
        inv_d2 = 1.0 / (dist * dist)
        np.add.at(grad_euclidean, idx_i, diff * inv_d2[:, np.newaxis])
        np.add.at(grad_euclidean, idx_j, -diff * inv_d2[:, np.newaxis])
        
        return energy, grad_euclidean

    def _tangent_grad(X, grad_euclidean):
        """Project Euclidean gradient to tangent space."""
        dot = np.sum(grad_euclidean * X, axis=1, keepdims=True)
        return grad_euclidean - dot * X

    def _hessian_vp(X, V):
        """
        Riemannian Hessian-vector product via finite differences in tangent space.
        V: (N, 3) tangent vector field.
        Returns: (N, 3) tangent vector (Hessian applied to V).
        """
        eps = 1e-5
        # Move V by +eps and -eps
        X_plus = _exp_map(X, V, eps)
        X_minus = _exp_map(X, -V, eps)
        
        _, g_plus = _energy_and_grad(X_plus)
        _, g_minus = _energy_and_grad(X_minus)
        
        g_plus_tan = _tangent_grad(X_plus, g_plus)
        g_minus_tan = _tangent_grad(X_minus, g_minus)
        
        # Parallel transport g_plus_tan and g_minus_tan back to X
        g_plus_back = _parallel_transport(X_plus, X, g_plus_tan)
        g_minus_back = _parallel_transport(X_minus, X, g_minus_tan)
        
        hvp = (g_plus_back - g_minus_back) / (2 * eps)
        return hvp

    def _parallel_transport(X_from, X_to, V):
        """
        Parallel transport V from tangent space at X_from to tangent space at X_to.
        On the sphere: rotate V so that it remains orthogonal to X_to.
        """
        # Simple approach: project V onto tangent space at X_to
        # This is not true parallel transport but a reasonable approximation
        # For better accuracy, use rotation
        # Find rotation that maps X_from to X_to
        # Then apply same rotation to V
        
        # Project V to tangent at X_to (approximation)
        dot = np.sum(V * X_to, axis=1, keepdims=True)
        V_proj = V - dot * X_to
        
        # Re-orthogonalize
        norm = np.linalg.norm(V_proj, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        return V_proj / norm

    def _exp_map(X, V, step):
        """Exponential map on the sphere."""
        norm_V = np.linalg.norm(V, axis=1, keepdims=True)
        norm_V = np.maximum(norm_V, 1e-12)
        theta = step * norm_V
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        V_unit = V / norm_V
        X_new = cos_theta * X + sin_theta * V_unit
        X_new_norm = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new_norm = np.maximum(X_new_norm, 1e-12)
        return X_new / X_new_norm

    def _repulsive_energy(X, power=2):
        """
        Repulsive energy: sum 1/d^power, used for pre-conditioning.
        """
        diff = X[idx_i] - X[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-12)
        return np.sum(1.0 / (dist ** power))

    def _repulsive_grad(X, power=2):
        """Gradient of repulsive energy."""
        diff = X[idx_i] - X[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-12)
        
        grad_euclidean = np.zeros((N, 3))
        # d/dX_i [1/d^power] = -power * (X_i - X_j) / d^(power+2)
        factor = -power / (dist ** (power + 2))
        np.add.at(grad_euclidean, idx_i, diff * factor[:, np.newaxis])
        np.add.at(grad_euclidean, idx_j, -diff * factor[:, np.newaxis])
        
        return grad_euclidean

    def _local_minimize_log(X0, max_iter=200, initial_step=0.1):
        """
        Damped Newton method on the sphere for log energy.
        Uses Hessian-vector products via finite differences.
        """
        X = X0.copy()
        step = initial_step
        energy, grad_euclidean = _energy_and_grad(X)
        grad_tan = _tangent_grad(X, grad_euclidean)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            grad_norm = np.linalg.norm(grad_tan)
            if grad_norm < 1e-9:
                break
            
            # Newton direction: solve H * d = -grad_tan
            # Use conjugate gradient on the Hessian (approximate)
            # For simplicity, use a few CG iterations with HVP
            d = np.zeros_like(grad_tan)
            r = -grad_tan
            p = r.copy()
            rs_old = np.sum(r * r)
            
            for cg_iter in range(30):
                Hp = _hessian_vp(X, p)
                # Ensure Hp is in tangent space
                dot_p = np.sum(Hp * X, axis=1, keepdims=True)
                Hp = Hp - dot_p * X
                
                p_norm = np.sum(p * p)
                if p_norm < 1e-15:
                    break
                alpha = rs_old / p_norm
                d_new = d + alpha * p
                r_new = r - alpha * Hp
                rs_new = np.sum(r_new * r_new)
                
                if rs_new < 1e-12:
                    d = d_new
                    r = r_new
                    break
                
                beta = rs_new / rs_old
                p = r_new + beta * p
                rs_old = rs_new
                d = d_new
            
            # Ensure d is in tangent space
            dot_d = np.sum(d * X, axis=1, keepdims=True)
            d = d - dot_d * X
            
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-12:
                # Fallback to gradient direction
                d = -grad_tan
                d_norm = np.linalg.norm(d)
            
            # Scale direction to unit norm for line search
            d_unit = d / np.maximum(d_norm, 1e-12)
            
            # Backtracking line search
            alpha = step
            best_energy = energy
            best_X = X
            best_grad_tan = grad_tan
            
            for ls in range(20):
                X_trial = _exp_map(X, d_unit, alpha)
                e_trial, g_trial_euclidean = _energy_and_grad(X_trial)
                g_trial_tan = _tangent_grad(X_trial, g_trial_euclidean)
                
                if e_trial < best_energy - 1e-4 * alpha * grad_norm:
                    best_energy = e_trial
                    best_X = X_trial
                    best_grad_tan = g_trial_tan
                    break
                else:
                    alpha *= 0.5
            
            if best_energy < energy:
                X = best_X
                energy = best_energy
                grad_tan = best_grad_tan
            else:
                step *= 0.5
                if step < 1e-10:
                    break
        
        return X, energy

    def _local_minimize_repulsive(X0, max_iter=300, initial_step=0.05):
        """
        Gradient descent on repulsive energy for pre-conditioning.
        """
        X = X0.copy()
        step = initial_step
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            grad_euclidean = _repulsive_grad(X, power=2)
            grad_tan = _tangent_grad(X, grad_euclidean)
            grad_norm = np.linalg.norm(grad_tan)
            if grad_norm < 1e-9:
                break
            
            direction = -grad_tan / grad_norm
            
            alpha = step
            best_X = X
            best_energy = _repulsive_energy(X)
            
            for ls in range(15):
                X_trial = _exp_map(X, direction, alpha)
                e_trial = _repulsive_energy(X_trial)
                if e_trial < best_energy:
                    best_X = X_trial
                    best_energy = e_trial
                    break
                else:
                    alpha *= 0.5
            
            X = best_X
        
        return X

    def _fibonacci_init():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def _random_init(seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((N, 3))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms

    # Phase 1: Pre-condition with repulsive energy from multiple starts
    initializations = [
        _fibonacci_init(),
        _random_init(42),
        _random_init(123),
        _random_init(456),
    ]
    
    best_pre_X = initializations[0].copy()
    best_pre_energy = float('inf')
    
    for init in initializations:
        if time.time() > deadline - 30.0:
            break
        X_pre = _local_minimize_repulsive(init)
        e_pre = _repulsive_energy(X_pre)
        if e_pre < best_pre_energy:
            best_pre_energy = e_pre
            best_pre_X = X_pre
    
    # Phase 2: Optimize log energy from the best pre-conditioned start
    # Also try a few direct log-energy optimizations from other starts
    candidates = [best_pre_X]
    for init in initializations[1:]:
        candidates.append(init)
    
    best_X = candidates[0].copy()
    best_energy = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 10.0:
            break
        X_opt, e_opt = _local_minimize_log(cand)
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt
    
    # Final normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    return best_X
# EVOLVE-BLOCK-END
