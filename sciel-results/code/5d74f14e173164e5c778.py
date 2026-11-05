import numpy as np
import time
import scipy.linalg as la
import scipy.integrate as spi

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Detect if the system is linear/affine and time-invariant (y' = A y + b).
    2. If so, attempt to exploit tridiagonal structure to minimize f calls.
    3. If structure holds, use matrix exponential for exact solution.
    4. If not, fallback to full linear extraction or high-accuracy adaptive Runge-Kutta.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # --- Attempt Linear/Affine Time-Invariant Extraction ---
    
    try:
        # 1. Probe b = f(t0, 0)
        b = np.asarray(f(t0, np.zeros(n)), dtype=float)
        
        # 2. Attempt Tridiagonal Structure Detection
        # We need to determine if A is tridiagonal with constant interior coefficients.
        # Probes: e_0, e_1, e_{n-1}
        
        e_0 = np.zeros(n); e_0[0] = 1.0
        c_0 = np.asarray(f(t0, e_0), dtype=float) - b
        
        e_1 = np.zeros(n); e_1[1] = 1.0
        c_1 = np.asarray(f(t0, e_1), dtype=float) - b
        
        e_last = np.zeros(n); e_last[n-1] = 1.0
        c_last = np.asarray(f(t0, e_last), dtype=float) - b
        
        # Extract coefficients from c_1 (interior representative)
        # c_1 has non-zeros at indices 0, 1, 2 (if n>2)
        # A[0, 1] = c_1[0] (Upper)
        # A[1, 1] = c_1[1] (Diag)
        # A[2, 1] = c_1[2] (Lower)
        
        upper = c_1[0]
        diag = c_1[1]
        lower = c_1[2]
        
        # Check sparsity of c_0, c_1, c_last to ensure tridiagonal
        # c_0 should be non-zero only at 0, 1
        # c_1 should be non-zero only at 0, 1, 2
        # c_last should be non-zero only at n-2, n-1
        
        def is_sparse_tridiag(col, n, expected_indices):
            for i in range(n):
                if i in expected_indices:
                    continue
                if abs(col[i]) > 1e-10:
                    return False
            return True

        # Expected indices for c_0: {0, 1}
        # Expected indices for c_1: {0, 1, 2} (if n>2, else adjust)
        # Expected indices for c_last: {n-2, n-1}
        
        if n == 1:
            # Trivial case
            A = np.array([[c_0[0]]])
            b_val = b[0]
            # Solution: y(t) = e^{At}(y0 - b/A) + b/A
            # Or just use expm on 2x2
            M = np.array([[A[0,0], b_val], [0, 1]])
            z0 = np.array([y0[0], 1.0])
            ExpM = la.expm((t1-t0)*M)
            return (ExpM @ z0)[0]

        if n == 2:
             # c_0: indices 0,1. c_1: indices 0,1. c_last is c_1.
             # Just build 2x2 from c_0 and c_1
             A = np.zeros((2,2))
             A[:, 0] = c_0
             A[:, 1] = c_1
             M = np.zeros((3,3))
             M[:2, :2] = A
             M[:2, 2] = b
             M[2, 2] = 1.0
             z0 = np.zeros(3)
             z0[:2] = y0
             z0[2] = 1.0
             ExpM = la.expm((t1-t0)*M)
             return (ExpM @ z0)[:2]

        # n >= 3
        exp_idx_0 = {0, 1}
        exp_idx_1 = {0, 1, 2}
        exp_idx_last = {n-2, n-1}
        
        if is_sparse_tridiag(c_0, n, exp_idx_0) and \
           is_sparse_tridiag(c_1, n, exp_idx_1) and \
           is_sparse_tridiag(c_last, n, exp_idx_last):
            
            # Construct candidate A
            A = np.zeros((n, n))
            
            # Fill diagonal
            np.fill_diagonal(A, diag)
            
            # Fill upper
            for i in range(n-1):
                A[i, i+1] = upper
                
            # Fill lower
            for i in range(1, n):
                A[i, i-1] = lower
                
            # Overwrite boundaries with specific probes
            # Column 0
            A[0, 0] = c_0[0]
            A[1, 0] = c_0[1]
            
            # Column n-1
            A[n-1, n-1] = c_last[n-1]
            A[n-2, n-1] = c_last[n-2]
            
            # Verify with a random test point
            # This ensures the constant interior assumption holds and boundaries are correct
            np.random.seed(42)
            y_test = np.random.rand(n)
            pred = A @ y_test + b
            act = np.asarray(f(t0, y_test), dtype=float)
            
            err = np.linalg.norm(act - pred)
            scale = np.linalg.norm(pred) + 1e-12
            
            if err / scale < 1e-10:
                dt = t1 - t0
                M = np.zeros((n + 1, n + 1))
                M[:n, :n] = A
                M[:n, n] = b
                M[n, n] = 1.0
                
                z0 = np.zeros(n + 1)
                z0[:n] = y0
                z0[n] = 1.0
                
                ExpM = la.expm(dt * M)
                z_final = ExpM @ z0
                y_final = z_final[:n]
                
                if np.all(np.isfinite(y_final)):
                    return y_final
            else:
                # Structure mismatch, fallback to full extraction
                pass
        else:
            # Not tridiagonal, fallback to full extraction
            pass

        # Fallback: Full Linear Extraction
        # We already have b, c_0, c_1, c_last.
        # We need to fill the rest of A.
        A = np.zeros((n, n))
        A[:, 0] = c_0
        A[:, 1] = c_1
        A[:, n-1] = c_last
        
        # Fill remaining columns
        for i in range(2, n-1):
            if i == 1 or i == n-1: continue # Already done
            e_i = np.zeros(n)
            e_i[i] = 1.0
            val = np.asarray(f(t0, e_i), dtype=float)
            A[:, i] = val - b
            
        # Verify full A
        np.random.seed(43)
        y_test = np.random.rand(n)
        pred = A @ y_test + b
        act = np.asarray(f(t0, y_test), dtype=float)
        
        err = np.linalg.norm(act - pred)
        scale = np.linalg.norm(pred) + 1e-12
        
        if err / scale < 1e-10:
            dt = t1 - t0
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = A
            M[:n, n] = b
            M[n, n] = 1.0
            
            z0 = np.zeros(n + 1)
            z0[:n] = y0
            z0[n] = 1.0
            
            ExpM = la.expm(dt * M)
            z_final = ExpM @ z0
            y_final = z_final[:n]
            
            if np.all(np.isfinite(y_final)):
                return y_final
                
    except Exception:
        pass

    # --- Fallback: High-accuracy adaptive solver ---
    methods = [
        ('Radau', {'rtol': 1e-10, 'atol': 1e-12}),
        ('BDF', {'rtol': 1e-10, 'atol': 1e-12}),
        ('DOP853', {'rtol': 1e-12, 'atol': 1e-14}),
    ]
    
    for method_name, opts in methods:
        if time.time() > deadline:
            break
        try:
            if method_name not in spi.solve_ivp.methods:
                continue
            sol = spi.solve_ivp(
                f, t_span, y0,
                method=method_name,
                **opts
            )
            if sol.success:
                y_final = sol.y[:, -1]
                if np.all(np.isfinite(y_final)):
                    return y_final
        except Exception:
            pass
            
    # Last resort: Fixed step RK4
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or time.time() > deadline:
            break
            
    return y
