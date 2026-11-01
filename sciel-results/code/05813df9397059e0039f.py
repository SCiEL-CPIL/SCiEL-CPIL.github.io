import numpy as np
import time

DIMENSIONS = [2, 4, 7]
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    def build_strassen_222(i0, j0, k0, U, V, W, col):
        """Add a 2x2x2 Strassen block. Returns new col index."""
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        return col
    
    def build_schoolbook_221(i0, j0, k0, U, V, W, col):
        """Add a 2x2x1 schoolbook block (4 mults)."""
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        return col
    
    def verify_decomposition(U, V, W, rank):
        """Check that the decomposition is exact using multiple random tests."""
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            A = np.random.randint(-5, 6, size=(n, m))
            B = np.random.randint(-5, 6, size=(m, p))
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                U_r = U[:, r]
                V_r = V[:, r]
                W_r = W[:, r]
                if np.all(U_r == 0) or np.all(V_r == 0) or np.all(W_r == 0):
                    continue
                UA = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        if U_r[i * m + j] != 0:
                            UA[i] += U_r[i * m + j] * A[i, j]
                VB = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        if V_r[j * p + k] != 0:
                            VB[k] += V_r[j * p + k] * B[j, k]
                for i in range(n):
                    for k in range(p):
                        if W_r[k * n + i] != 0:
                            C_computed[i, k] += W_r[k * n + i] * UA[i] * VB[k]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    def try_find_lower_rank(start_rank, max_attempts=50):
        """Try to find a lower-rank decomposition by perturbing and optimizing."""
        # Build the known rank-50 solution first
        U50 = np.zeros((n * m, 50), dtype=np.float64)
        V50 = np.zeros((m * p, 50), dtype=np.float64)
        W50 = np.zeros((p * n, 50), dtype=np.float64)
        col = 0
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                col = build_strassen_222(0, j_start, k_start, U50, V50, W50, col)
            col = build_schoolbook_221(0, j_start, 6, U50, V50, W50, col)
        
        # Try to reduce rank by zeroing out some columns and adjusting others
        # This is a heuristic approach
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE_SECONDS - 30:
                break
            
            target_rank = start_rank
            np.random.seed(attempt * 1000 + 42)
            
            # Start from rank-50 and try to nullify some terms
            U = U50[:, :target_rank].copy()
            V = V50[:, :target_rank].copy()
            W = W50[:, :target_rank].copy()
            
            # Add random perturbations to try to absorb the removed terms
            noise_scale = 0.1
            U += np.random.randn(n * m, target_rank) * noise_scale
            V += np.random.randn(m * p, target_rank) * noise_scale
            W += np.random.randn(p * n, target_rank) * noise_scale
            
            # Simple gradient-free optimization: try to minimize error
            # by adjusting the last few columns
            best_err = np.inf
            for inner in range(100):
                if time.time() - start_time > DEADLINE_SECONDS - 30:
                    break
                
                # Compute error
                np.random.seed(42)
                A = np.random.randint(-5, 6, size=(n, m))
                B = np.random.randint(-5, 6, size=(m, p))
                C_expected = A @ B
                
                C_computed = np.zeros((n, p))
                for r in range(target_rank):
                    U_r = U[:, r]
                    V_r = V[:, r]
                    W_r = W[:, r]
                    if np.all(U_r == 0) or np.all(V_r == 0) or np.all(W_r == 0):
                        continue
                    UA = np.zeros(n)
                    for i in range(n):
                        for j in range(m):
                            if U_r[i * m + j] != 0:
                                UA[i] += U_r[i * m + j] * A[i, j]
                    VB = np.zeros(p)
                    for k in range(p):
                        for j in range(m):
                            if V_r[j * p + k] != 0:
                                VB[k] += V_r[j * p + k] * B[j, k]
                    for i in range(n):
                        for k in range(p):
                            if W_r[k * n + i] != 0:
                                C_computed[i, k] += W_r[k * n + i] * UA[i] * VB[k]
                
                err = np.linalg.norm(C_computed - C_expected)
                if err < best_err:
                    best_err = err
                
                # Perturb
                U += np.random.randn(n * m, target_rank) * 0.01
                V += np.random.randn(m * p, target_rank) * 0.01
                W += np.random.randn(p * n, target_rank) * 0.01
            
            if best_err < 1e-8:
                # Round to nearest integers or half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                if verify_decomposition(U_r, V_r, W_r, target_rank):
                    return U_r, V_r, W_r, target_rank
        
        return None
    
    # Build the known rank-50 solution
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            col = build_strassen_222(0, j_start, k_start, U, V, W, col)
        col = build_schoolbook_221(0, j_start, 6, U, V, W, col)
    assert col == 50
    
    best_result = None
    best_rank = 56
    
    if verify_decomposition(U, V, W, 50):
        best_result = (U.copy(), V.copy(), W.copy(), 50)
        best_rank = 50
    
    # Try to find lower rank
    if time.time() - start_time < DEADLINE_SECONDS - 60:
        for target_rank in [49, 48, 47]:
            if time.time() - start_time > DEADLINE_SECONDS - 30:
                break
            result = try_find_lower_rank(target_rank, max_attempts=30)
            if result is not None:
                U_r, V_r, W_r, r = result
                if r < best_rank:
                    best_result = (U_r, V_r, W_r, r)
                    best_rank = r
                break
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_result = (U, V, W, rank)
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
