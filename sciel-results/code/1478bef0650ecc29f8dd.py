"""Root finder for 3 fixed 2-D systems: Robust Broyden's method with multi-start."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with robust line search and multi-start fallbacks.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    x0 = np.array(x0, dtype=float)
    call_count = 0
    best_x = x0.copy()
    best_norm = float('inf')

    def safe_F(x_in):
        nonlocal call_count
        if time.time() > deadline:
            raise TimeoutError()
        if call_count >= 490:
            raise TimeoutError()
        call_count += 1
        result = F(x_in)
        result = np.asarray(result, dtype=float).flatten()[:2]
        if not np.all(np.isfinite(result)):
            return np.array([1e10, 1e10])
        return result

    def update_best(x, Fx):
        nonlocal best_x, best_norm
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_norm = norm
            best_x = x.copy()
        return norm

    def try_broyden(x_start, max_iters=50):
        nonlocal call_count
        x = x_start.copy()
        try:
            Fx = safe_F(x)
        except TimeoutError:
            return False
        
        norm = update_best(x, Fx)
        if norm <= 1e-8:
            return True

        # Initial Jacobian via finite differences
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            try:
                Fx_e = safe_F(x + e)
            except TimeoutError:
                return False
            J[:, j] = (Fx_e - Fx) / h

        for _ in range(max_iters):
            if time.time() > deadline or call_count >= 490:
                return False
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    return False
            
            # Line Search
            alpha = 1.0
            current_norm = norm
            success = False
            for _ in range(10):
                x_try = x + alpha * dx
                try:
                    Fx_try = safe_F(x_try)
                except TimeoutError:
                    return False
                norm_try = np.linalg.norm(Fx_try)
                
                if norm_try < current_norm * 0.9: # 10% decrease required
                    # Update Broyden
                    s = x_try - x
                    y = Fx_try - Fx
                    if s @ s > 1e-14:
                        J = J + np.outer(y - J @ s, s) / (s @ s)
                    
                    x = x_try
                    Fx = Fx_try
                    norm = norm_try
                    update_best(x, Fx)
                    success = True
                    break
                else:
                    alpha *= 0.5
            
            if not success:
                # Stagnation or failure
                return False
                
            if norm <= 1e-8:
                return True

        return False

    # Phase 1: Try from x0
    if not try_broyden(x0, 50):
        # Phase 2: Local Grid Search for better start
        offsets = [
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, -1], [1, -1], [-1, 1],
            [2, 0], [-2, 0], [0, 2], [0, -2]
        ]
        candidates = [x0]
        for off in offsets:
            candidates.append(x0 + np.array(off, dtype=float))
        
        best_start = x0
        best_start_norm = best_norm
        
        # We already have best_norm from x0, but let's check neighbors
        for c in candidates[1:]:
            try:
                Fc = safe_F(c)
            except TimeoutError:
                continue
            nc = np.linalg.norm(Fc)
            if nc < best_start_norm:
                best_start_norm = nc
                best_start = c
        
        # If we found a better start, try Broyden from there
        if best_start is not x0:
            if try_broyden(best_start, 50):
                pass # Success

        # Phase 3: Random Restarts if still failing
        if best_norm > 1e-8:
            # Generate random points in a box around x0
            # Scale: try 1, 10, 100
            for scale in [1.0, 10.0, 100.0]:
                for _ in range(5):
                    if time.time() > deadline or call_count >= 450:
                        break
                    offset = np.random.uniform(-scale, scale, size=2)
                    x_rand = x0 + offset
                    try:
                        Frand = safe_F(x_rand)
                    except TimeoutError:
                        continue
                    nr = np.linalg.norm(Frand)
                    if nr < best_start_norm:
                        best_start_norm = nr
                        best_start = x_rand
                        # Try Broyden from this random point
                        if try_broyden(best_start, 30):
                            break
                if best_norm <= 1e-8:
                    break

    return best_x
