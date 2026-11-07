import numpy as np
import time


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    calls = 0
    max_calls = 490 # Safety margin

    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= max_calls:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None

    def broyden(F, x_init, max_iter, call_limit):
        x = np.array(x_init, dtype=float)
        Fx = safe_F(x)
        if Fx is None:
            return x, float('inf')
        
        norm = np.linalg.norm(Fx)
        if norm <= 1e-8:
            return x, norm

        # Compute initial Jacobian
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            if calls >= call_limit:
                break
            e = np.zeros(2)
            e[j] = h
            Fx_j = safe_F(x + e)
            if Fx_j is None:
                J = np.eye(2)
                break
            J[:, j] = (Fx_j - Fx) / h
        else:
            # If loop finished without break, J is valid
            pass
            
        if calls >= call_limit:
            return x, norm

        for _ in range(max_iter):
            if calls >= call_limit or time.time() > deadline:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break

            if not np.all(np.isfinite(dx)):
                break

            # Backtracking line search
            alpha = 1.0
            Fx_old_norm = norm
            step_accepted = False

            for _ in range(10):
                if calls >= call_limit:
                    break
                x_try = x + alpha * dx
                Fx_try = safe_F(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue

                norm_try = np.linalg.norm(Fx_try)

                if norm_try < Fx_old_norm:
                    s = alpha * dx
                    y = Fx_try - Fx
                    J_s = J @ s
                    diff = y - J_s
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(diff, s) / denom
                    else:
                        J = np.eye(2)

                    x = x_try
                    Fx = Fx_try
                    norm = norm_try
                    step_accepted = True
                    break
                else:
                    alpha *= 0.5

            if not step_accepted:
                break

            if norm <= 1e-8:
                return x, norm

        return x, norm

    # Strategy: Multi-start
    # 1. Try x0 with a small budget to see if it's easy
    # 2. If not, try other points with remaining budget
    
    # Phase 1: Try x0 with limited calls (e.g., 20 calls)
    # This allows quick success for easy cases without wasting budget if it fails
    x1, n1 = broyden(F, x, 20, 20)
    if n1 < best_norm:
        best_norm = n1
        best_x = x1.copy()
    
    if best_norm <= 1e-8:
        return best_x

    # Phase 2: Generate diverse starting points
    # We have ~470 calls left. 
    # Let's try 3 diverse starts.
    # Start 2: Perturbed x0
    # Start 3: Random point in [-10, 10]
    # Start 4: Random point in [-1, 1]
    
    starts = []
    # Perturbation
    starts.append(x + np.array([0.1, -0.1]))
    starts.append(x + np.array([-0.1, 0.1]))
    # Random near origin
    starts.append(np.array([0.5, 0.5]))
    starts.append(np.array([-0.5, -0.5]))
    # Random far
    starts.append(np.array([5.0, 5.0]))
    starts.append(np.array([-5.0, -5.0]))
    
    # Allocate remaining calls to these starts
    remaining_calls = max_calls - calls
    if remaining_calls <= 0:
        return best_x
    
    # We will try each start with a portion of the remaining budget
    # Or just run one after another, breaking if we find a root or run out of calls
    # Since Broyden is fast, we can just run them sequentially with a high max_iter
    # but bounded by global call limit
    
    for s in starts:
        if calls >= max_calls or time.time() > deadline:
            break
        
        # Give each start a reasonable budget, e.g., 50 calls
        # But we must not exceed global limit
        # We'll just let broyden run until it converges or hits global limit
        # To avoid one start eating all budget, we can limit iterations per start
        # But Broyden converges in < 10 steps usually. 
        # Let's limit iterations to 30 per start to ensure we try multiple starts if possible
        
        x_try, n_try = broyden(F, s, 30, max_calls)
        
        if n_try < best_norm:
            best_norm = n_try
            best_x = x_try.copy()
            
        if best_norm <= 1e-8:
            return best_x

    return best_x
