import numpy as np
import scipy.optimize
import time

def coefficients(stages):
    start_time = time.time()
    
    # 1. Build Hamiltonians
    # 6-site XXZ chain, open boundary
    # S^x, S^y, S^z on each site
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    
    def kron_all(*mats):
        res = mats[0]
        for m in mats[1:]:
            res = np.kron(res, m)
        return res
        
    # Identity
    I2 = np.eye(2, dtype=complex)
    
    # Operators on 6 sites
    # A: Odd bonds (1-2, 3-4, 5-6) + half field
    # B: Even bonds (2-3, 4-5) + half field
    
    # Term helper
    def term(ops):
        # ops is list of 6 matrices
        return kron_all(*ops)
        
    A = 0.0
    B = 0.0
    
    # Bonds
    # 1-2 (odd)
    A += term([sx, sx, I2, I2, I2, I2])
    A += term([sy, sy, I2, I2, I2, I2])
    A += 0.8 * term([sz, sz, I2, I2, I2, I2])
    
    # 2-3 (even)
    B += term([I2, sx, sx, I2, I2, I2])
    B += term([I2, sy, sy, I2, I2, I2])
    B += 0.8 * term([I2, sz, sz, I2, I2, I2])
    
    # 3-4 (odd)
    A += term([I2, I2, sx, sx, I2, I2])
    A += term([I2, I2, sy, sy, I2, I2])
    A += 0.8 * term([I2, I2, sz, sz, I2, I2])
    
    # 4-5 (even)
    B += term([I2, I2, I2, sx, sx, I2])
    B += term([I2, I2, I2, sy, sy, I2])
    B += 0.8 * term([I2, I2, I2, sz, sz, I2])
    
    # 5-6 (odd)
    A += term([I2, I2, I2, I2, sx, sx])
    A += term([I2, I2, I2, I2, sy, sy])
    A += 0.8 * term([I2, I2, I2, I2, sz, sz])
    
    # Field
    # h = 0.3
    # A gets half, B gets half
    field_term = 0.0
    for i in range(6):
        ops = [I2]*6
        ops[i] = sz
        field_term += term(ops)
        
    A += 0.15 * field_term
    B += 0.15 * field_term
    
    H = A + B
    
    # 2. Precompute Eigendecomps
    wA, vA = np.linalg.eigh(A)
    wB, vB = np.linalg.eigh(B)
    wH, vH = np.linalg.eigh(H)
    
    ts = [0.25, 0.5, 1.0]
    U_exact = [vH @ np.diag(np.exp(-1j * t * wH)) @ vH.conj().T for t in ts]
    
    # 3. Optimization
    # Symmetric 12-stage:
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # Sum a = 2*(a1+...+a6) = 1 => sum(a1..a6) = 0.5
    # Sum b = 2*(b1+...+b6) = 1 => sum(b1..b6) = 0.5
    
    def get_full(a_half, b_half):
        a = np.concatenate([a_half, a_half[::-1]])
        b = np.concatenate([b_half, b_half[::-1]])
        return a, b
        
    def objective(x):
        # x[0:5] = a1..a5, a6 = 0.5 - sum(a1..a5)
        # x[5:10] = b1..b5, b6 = 0.5 - sum(b1..b5)
        a_half = np.zeros(6)
        b_half = np.zeros(6)
        a_half[:5] = x[:5]
        a_half[5] = 0.5 - np.sum(a_half[:5])
        b_half[:5] = x[5:10]
        b_half[5] = 0.5 - np.sum(b_half[:5])
        
        a, b = get_full(a_half, b_half)
        
        # Optional: Check bounds
        # If any value outside [-50, 50], return large penalty
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
            
        err = 0.0
        for t, Ue in zip(ts, U_exact):
            U = np.eye(64, dtype=complex)
            for k in range(12):
                # exp(-i b_k t B)
                expB = vB @ np.diag(np.exp(-1j * b[k] * t * wB)) @ vB.conj().T
                U = expB @ U
                # exp(-i a_k t A)
                expA = vA @ np.diag(np.exp(-1j * a[k] * t * wA)) @ vA.conj().T
                U = expA @ U
            err += np.linalg.norm(U - Ue, 'fro')**2
        return err

    # Initial guess: Symmetric Strang-like
    # a: [1/12]*12 -> a_half = [1/12]*6
    # b: [1/6, 0, 1/6, 0, 1/6, 0, 0, 1/6, 0, 1/6, 0, 1/6]? 
    # Seed: substeps=6.
    # k=1: b=1/6, a=1/12
    # k=2: b=0, a=1/12
    # ...
    # This is not symmetric in the way defined above (a_k = a_{13-k}).
    # Seed a is symmetric: 1/12 everywhere.
    # Seed b: 1/6, 0, 1/6, 0, 1/6, 0, 0, 1/6, 0, 1/6, 0, 1/6.
    # b_1=1/6, b_2=0, b_3=1/6, b_4=0, b_5=1/6, b_6=0.
    # b_7=0, b_8=1/6, b_9=0, b_10=1/6, b_11=0, b_12=1/6.
    # b_1 != b_12 (1/6 vs 1/6, ok). b_2 != b_11 (0 vs 0, ok).
    # It IS symmetric.
    
    a_init_half = np.array([1/12]*6)
    b_init_half = np.array([1/6, 0, 1/6, 0, 1/6, 0])
    
    x0 = np.concatenate([a_init_half[:5], b_init_half[:5]])
    
    # Run Nelder-Mead
    best_x = x0
    best_err = objective(x0)
    
    # Multiple restarts if time permits?
    # 150s budget.
    # One run of Nelder-Mead on 10 dims is fast.
    
    try:
        res = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                                      options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
        if res.success or res.fun < best_err:
            best_x = res.x
            best_err = res.fun
    except Exception:
        pass
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed < 140:
        # Try a different initial guess?
        # Random symmetric guess
        np.random.seed(42)
        x_rand = np.random.uniform(-0.1, 0.1, 10)
        # Ensure sums are close
        # Just let optimizer handle it
        
        try:
            res2 = scipy.optimize.minimize(objective, x_rand, method='Nelder-Mead', 
                                           options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
            if res2.fun < best_err:
                best_x = res2.x
                best_err = res2.fun
        except Exception:
            pass
            
    a_half = np.zeros(6)
    b_half = np.zeros(6)
    a_half[:5] = best_x[:5]
    a_half[5] = 0.5 - np.sum(a_half[:5])
    b_half[:5] = best_x[5:10]
    b_half[5] = 0.5 - np.sum(b_half[:5])
    
    a, b = get_full(a_half, b_half)
    
    # Final safety checks
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    a = a / np.sum(a)
    b = b / np.sum(b)
    
    return a.tolist(), b.tolist()
