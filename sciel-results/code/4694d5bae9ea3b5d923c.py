"""Combined program: Strassen block decomposition + cross-column optimization search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115  # safety margin before 150s kill

    def verify_decomp(U, V, W, trials=30):
        """Verify that the decomposition exactly computes matrix multiplication using trace convention W[k*n+i, r]."""
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        for _ in range(trials):
            A = np.array([[random.randint(-10, 10) for _ in range(m)] for _ in range(n)])
            B = np.array([[random.randint(-10, 10) for _ in range(p)] for _ in range(m)])
            C_true = A @ B

            Ua = np.array(U) if isinstance(U, list) else U
            Va = np.array(V) if isinstance(V, list) else V
            Wa = np.array(W) if isinstance(W, list) else W

            C_calc = np.zeros((n, p))
            for r in range(rank):
                u_val = sum(Ua[i*m+j, r] * A[i, j] for i in range(n) for j in range(m))
                v_val = sum(Va[j*p+k, r] * B[j, k] for j in range(m) for k in range(p))
                prod = u_val * v_val
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += Wa[k*n+i, r] * prod

            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    def build_rank25():
        """Build the known rank-25 Strassen-based decomposition for 2x2 * 2x7."""
        rank = 25
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_strassen(col_start, k_start):
            # Strassen for 2x2 * 2x2 block starting at column k_start
            # M1 = (A00+A11)*(B0,ks + B1,ks+1)
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[7*0+k_start, col_start+0] += 1; V[7*1+k_start+1, col_start+0] += 1
            # Output: C[0,ks] += M1, C[1,ks+1] += M1
            W[k_start*n+0, col_start+0] += 1; W[(k_start+1)*n+1, col_start+0] += 1
            # M2 = (A10+A11)*B0,ks
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[7*0+k_start, col_start+1] += 1
            # Output: C[1,ks] += M2, C[1,ks+1] -= M2
            W[k_start*n+1, col_start+1] += 1; W[(k_start+1)*n+1, col_start+1] -= 1
            # M3 = A00*(B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0+k_start+1, col_start+2] += 1; V[7*1+k_start+1, col_start+2] -= 1
            # Output: C[0,ks+1] += M3, C[1,ks+1] += M3
            W[(k_start+1)*n+0, col_start+2] += 1; W[(k_start+1)*n+1, col_start+2] += 1
            # M4 = A11*(B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
            # Output: C[0,ks] += M4, C[1,ks] += M4
            W[k_start*n+0, col_start+3] += 1; W[k_start*n+1, col_start+3] += 1
            # M5 = (A00+A01)*B1,ks+1
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[7*1+k_start+1, col_start+4] += 1
            # Output: C[0,ks] -= M5, C[0,ks+1] += M5
            W[k_start*n+0, col_start+4] -= 1; W[(k_start+1)*n+0, col_start+4] += 1
            # M6 = (A10-A00)*(B0,ks+B0,ks+1)
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[7*0+k_start, col_start+5] += 1; V[7*0+k_start+1, col_start+5] += 1
            # Output: C[1,ks+1] += M6
            W[(k_start+1)*n+1, col_start+5] += 1
            # M7 = (A01-A11)*(B1,ks+B1,ks+1)
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[7*1+k_start, col_start+6] += 1; V[7*1+k_start+1, col_start+6] += 1
            # Output: C[0,ks] += M7
            W[k_start*n+0, col_start+6] += 1

        add_strassen(0, 0)
        add_strassen(7, 2)
        add_strassen(14, 4)
        # Schoolbook for column 6
        cs = 21
        U[0, cs+0] += 1; V[0*7+6, cs+0] += 1; W[6*n+0, cs+0] += 1
        U[1, cs+1] += 1; V[1*7+6, cs+1] += 1; W[6*n+0, cs+1] += 1
        U[2, cs+2] += 1; V[0*7+6, cs+2] += 1; W[6*n+1, cs+2] += 1
        U[3, cs+3] += 1; V[1*7+6, cs+3] += 1; W[6*n+1, cs+3] += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def search_rank24():
        """
        Attempt to find a rank-24 decomposition by merging two columns from the rank-25 solution.
        We assume the rank-25 solution is block diagonal. We try to replace two columns (i, j)
        with a single optimized column k, effectively reducing the rank by 1.
        """
        import scipy.optimize as opt
        
        # Get the rank-25 base solution
        base = build_rank25()
        U25 = np.array(base["U"])
        V25 = np.array(base["V"])
        W25 = np.array(base["W"])
        rank25 = U25.shape[1]
        
        # Pre-compute random test matrices for objective function
        # We use fixed matrices to ensure deterministic objective and faster convergence
        np.random.seed(42)
        A_tests = [np.random.randint(-5, 5, (n, m)).astype(float) for _ in range(5)]
        B_tests = [np.random.randint(-5, 5, (m, p)).astype(float) for _ in range(5)]
        
        def calculate_error(U, V, W):
            """Calculate sum of squared errors for the decomposition."""
            total_err = 0.0
            rank = U.shape[1]
            for A in A_tests:
                for B in B_tests:
                    C_true = A @ B
                    C_calc = np.zeros((n, p))
                    for r in range(rank):
                        u_val = np.dot(U[:, r], A.flatten())
                        v_val = np.dot(V[:, r], B.flatten())
                        C_calc += W[:, r] * (u_val * v_val)
                    total_err += np.sum((C_true - C_calc)**2)
            return total_err

        # Try merging each pair of columns
        # We prioritize merging columns from different blocks if possible, 
        # but any pair is a valid candidate.
        # To save time, we limit the number of pairs we try if it takes too long.
        pairs = [(i, j) for i in range(rank25) for j in range(i+1, rank25)]
        # Shuffle to avoid bias towards low indices if one fails
        random.shuffle(pairs)
        
        for i, j in pairs:
            if time.time() > deadline - 10:
                break
                
            # Construct initial guess for the new column
            # Simple average
            U_init_col = (U25[:, i] + U25[:, j]) / 2
            V_init_col = (V25[:, i] + V25[:, j]) / 2
            W_init_col = (W25[:, i] + W25[:, j]) / 2
            
            x0 = np.concatenate([U_init_col, V_init_col, W_init_col])
            
            def objective_merge(x):
                U_new = x[:n*m].reshape(n*m)
                V_new = x[n*m:n*m+m*p].reshape(m*p)
                W_new = x[n*m+m*p:].reshape(p*n)
                
                # Construct full U, V, W with merged column
                # We remove columns i and j, and insert new column at position i
                # To keep it simple, we just build the list of columns
                U_cols = []
                V_cols = []
                W_cols = []
                for k in range(rank25):
                    if k == i:
                        continue
                    if k == j:
                        continue
                    U_cols.append(U25[:, k])
                    V_cols.append(V25[:, k])
                    W_cols.append(W25[:, k])
                
                # Insert new column at position i (which is now the correct index in the reduced list)
                # Note: if i > j, the index shifts. Let's just append for simplicity in objective,
                # but the position doesn't matter for the sum.
                U_cols.append(U_new)
                V_cols.append(V_new)
                W_cols.append(W_new)
                
                U_full = np.column_stack(U_cols)
                V_full = np.column_stack(V_cols)
                W_full = np.column_stack(W_cols)
                
                return calculate_error(U_full, V_full, W_full)

            try:
                res = opt.minimize(objective_merge, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15})
                
                if res.fun < 1e-12:
                    U_new = res.x[:n*m].reshape(n*m)
                    V_new = res.x[n*m:n*m+m*p].reshape(m*p)
                    W_new = res.x[n*m+m*p:].reshape(p*n)
                    
                    U_cols = []
                    V_cols = []
                    W_cols = []
                    for k in range(rank25):
                        if k == i or k == j:
                            continue
                        U_cols.append(U25[:, k])
                        V_cols.append(V25[:, k])
                        W_cols.append(W25[:, k])
                    
                    U_cols.append(U_new)
                    V_cols.append(V_new)
                    W_cols.append(W_new)
                    
                    U_final = np.column_stack(U_cols)
                    V_final = np.column_stack(V_cols)
                    W_final = np.column_stack(W_cols)
                    
                    # Verify exactness
                    if verify_decomp(U_final.tolist(), V_final.tolist(), W_final.tolist(), trials=50):
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            except Exception:
                continue
        
        return None

    def verify_decomp_block(U, V, W, n, m, p, trials=50):
        rank = U.shape[1]
        for _ in range(trials):
            A = np.array([[random.randint(-10, 10) for _ in range(m)] for _ in range(n)])
            B = np.array([[random.randint(-10, 10) for _ in range(p)] for _ in range(m)])
            C_true = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(rank):
                u_val = sum(U[i*m+j, r] * A[i, j] for i in range(n) for j in range(m))
                v_val = sum(V[j*p+k, r] * B[j, k] for j in range(m) for k in range(p))
                prod = u_val * v_val
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += W[k*n+i, r] * prod
                        
            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    # Attempt to find rank-24
    result_24 = search_rank24()
    if result_24 is not None:
        if verify_decomp(result_24["U"], result_24["V"], result_24["W"], trials=50):
            return result_24

    # Build and verify the rank-25 decomposition
    result = build_rank25()
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result

    # Fallback: schoolbook rank 28
    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
