import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8 using Broyden's quasi-Newton method
    with adaptive damping and a polish phase.
    """
    x = np.array(x0, dtype=float)
    tol = 1e-8
    deadline = time.time() + 105.0  # Safety margin for 120s limit (evaluator kills at 150s)
    
    call_count = [0]
    
    def Fc(x):
        call_count[0] += 1
        return np.array(F(x), dtype=float)
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fc(x))
    
    if best_norm <= tol:
        return best_x
    
    max_total_calls = 480
    max_restarts = 20
    
    def compute_initial_jacobian(x):
        """Compute Jacobian via forward differences. Returns J and F(x)."""
        Fx = Fc(x)
        h = 1e-6 * max(1.0, np.linalg.norm(x))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (Fc(x + e) - Fx) / h
        return J, Fx
    
    def broyden_phase(x_start, J_init, F_init, max_iters):
        """
        Broyden's good method with adaptive damping.
        Returns (x, Fx, converged).
        """
        x_cur = x_start.copy()
        J = J_init.copy()
        Fx = F_init.copy()
        n = 2
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            if call_count[0] >= max_total_calls - 5:
                break
            
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x_cur.copy()
            
            if norm_Fx <= tol:
                return x_cur, Fx, True
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Fallback: least squares
                dx, _, _, _ = np.linalg.lstsq(J, -Fx, rcond=None)
            
            # Adaptive damping
            alpha = 1.0
            accepted = False
            for _ in range(6):
                x_try = x_cur + alpha * dx
                F_try = Fc(x_try)
                norm_try = np.linalg.norm(F_try)
                
                if norm_try < norm_Fx * 0.999:
                    # Broyden update
                    s = x_try - x_cur
                    y = F_try - Fx
                    denom = np.dot(y, s)
                    if abs(denom) > 1e-15:
                        J = J + (y - J @ s) @ s.T / denom
                    
                    x_cur = x_try
                    Fx = F_try
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
        
        return x_cur, Fx, np.linalg.norm(Fx) <= tol
    
    def polish_phase(x_start, max_iters):
        """
        Finite-difference Newton polish for final convergence.
        Returns (x, Fx, converged).
        """
        x_cur = x_start.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 1.0:
                break
            if call_count[0] >= max_total_calls - 3:
                break
            
            Fx = Fc(x_cur)
            norm_Fx = np.linalg.norm(Fx)
            
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x_cur.copy()
            
            if norm_Fx <= tol:
                return x_cur, Fx, True
            
            # Finite difference Jacobian
            h = 1e-6 * max(1.0, np.linalg.norm(x_cur))
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                J[:, j] = (Fc(x_cur + e) - Fx) / h
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx, _, _, _ = np.linalg.lstsq(J, -Fx, rcond=None)
            
            x_try = x_cur + dx
            F_try = Fc(x_try)
            if np.linalg.norm(F_try) < norm_Fx:
                x_cur = x_try
            else:
                # Try damped
                x_try = x_cur + 0.5 * dx
                F_try = Fc(x_try)
                if np.linalg.norm(F_try) < norm_Fx:
                    x_cur = x_try
                else:
                    break
        
        return x_cur, Fx, np.linalg.norm(Fx) <= tol
    
    # Main loop: try Broyden from various starting points
    start_points = [x.copy()]
    
    # Generate diverse restart points
    for i in range(1, max_restarts):
        if time.time() > deadline - 10.0:
            break
        if call_count[0] >= max_total_calls - 20:
            break
        
        if i == 1:
            sp = x0 + np.random.uniform(-1, 1, size=2)
        elif i <= 4:
            sp = x0 + np.random.uniform(-3, 3, size=2)
        elif i <= 8:
            sp = best_x + np.random.uniform(-2, 2, size=2)
        else:
            radius = 1.0 + 0.5 * (i - 8)
            angle = np.random.uniform(0, 2 * np.pi)
            sp = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        
        start_points.append(sp)
    
    for sp in start_points:
        if time.time() > deadline - 5.0:
            break
        if call_count[0] >= max_total_calls - 10:
            break
        
        # Check if already good
        F_sp = Fc(sp)
        n_sp = np.linalg.norm(F_sp)
        if n_sp < best_norm:
            best_norm = n_sp
            best_x = sp.copy()
        if n_sp <= tol:
            return best_x
        
        # Compute initial Jacobian
        J_init, F_init = compute_initial_jacobian(sp)
        if call_count[0] >= max_total_calls - 10:
            break
        
        # Run Broyden phase
        x_b, F_b, conv = broyden_phase(sp, J_init, F_init, max_iters=60)
        
        if conv:
            return x_b
        
        # If Broyden got close, try polish
        n_b = np.linalg.norm(F_b)
        if n_b < best_norm:
            best_norm = n_b
            best_x = x_b.copy()
        
        if n_b < 1e-4:
            x_p, F_p, conv_p = polish_phase(x_b, max_iters=30)
            if conv_p:
                return x_p
            n_p = np.linalg.norm(F_p)
            if n_p < best_norm:
                best_norm = n_p
                best_x = x_p.copy()
    
    return best_x
