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
        Attempt to find a rank-24 decomposition by finding a rank-13 decomposition for 2x2x4
        and combining it with a rank-11 decomposition for 2x2x3.
        """
        import scipy.optimize as opt
        
        n, m, p4 = 2, 2, 4
        rank4 = 13
        
        # Build initial rank-14 Strassen solution for 2x2x4
        U_init = np.zeros((n*m, 14))
        V_init = np.zeros((m*p4, 14))
        W_init = np.zeros((p4*n, 14))
        
        def add_strassen_4(U, V, W, col_start, k_start):
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[4*0+k_start, col_start+0] += 1; V[4*1+k_start+1, col_start+0] += 1
            W[k_start*n+0, col_start+0] += 1; W[(k_start+1)*n+1, col_start+0] += 1
            
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[4*0+k_start, col_start+1] += 1
            W[k_start*n+1, col_start+1] += 1; W[(k_start+1)*n+1, col_start+1] -= 1
            
            U[0, col_start+2] += 1
            V[4*0+k_start+1, col_start+2] += 1; V[4*1+k_start+1, col_start+2] -= 1
            W[(k_start+1)*n+0, col_start+2] += 1; W[(k_start+1)*n+1, col_start+2] += 1
            
            U[3, col_start+3] += 1
            V[4*1+k_start, col_start+3] += 1; V[4*0+k_start, col_start+3] -= 1
            W[k_start*n+0, col_start+3] += 1; W[k_start*n+1, col_start+3] += 1
            
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[4*1+k_start+1, col_start+4] += 1
            W[k_start*n+0, col_start+4] -= 1; W[(k_start+1)*n+0, col_start+4] += 1
            
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[4*0+k_start, col_start+5] += 1; V[4*0+k_start+1, col_start+5] += 1
            W[(k_start+1)*n+1, col_start+5] += 1
            
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[4*1+k_start, col_start+6] += 1; V[4*1+k_start+1, col_start+6] += 1
            W[k_start*n+0, col_start+6] += 1

        add_strassen_4(U_init, V_init, W_init, 0, 0)
        add_strassen_4(U_init, V_init, W_init, 7, 2)
        
        # Zero out last column to start optimization from rank 13
        U_init[:, 13] = 0
        V_init[:, 13] = 0
        W_init[:, 13] = 0
        
        def flatten_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten_params(x):
            U_size = n*m*rank4
            V_size = m*p4*rank4
            W_size = p4*n*rank4
            U = x[:U_size].reshape(n*m, rank4)
            V = x[U_size:U_size+V_size].reshape(m*p4, rank4)
            W = x[U_size+V_size:].reshape(p4*n, rank4)
            return U, V, W
            
        def objective(x):
            U, V, W = unflatten_params(x)
            errors = 0
            for _ in range(10):
                A = np.random.randn(n, m)
                B = np.random.randn(m, p4)
                C_true = A @ B
                
                C_calc = np.zeros((n, p4))
                for r in range(rank4):
                    u_val = np.dot(U[:, r], A.flatten())
                    v_val = np.dot(V[:, r], B.flatten())
                    C_calc += W[:, r] * (u_val * v_val)
                    
                errors += np.sum((C_true - C_calc)**2)
            
            return errors / 10

        x0 = flatten_params(U_init, V_init, W_init)
        
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < 1e-9:
                U_opt, V_opt, W_opt = unflatten_params(res.x)
                if verify_decomp_block(U_opt, V_opt, W_opt, n, m, p4, trials=50):
                    U_full = np.zeros((n*m, 24))
                    V_full = np.zeros((m*7, 24))
                    W_full = np.zeros((7*n, 24))
                    
                    U_full[:, :13] = U_opt
                    V_full[:m*4, :13] = V_opt
                    W_full[:4*n, :13] = W_opt
                    
                    cs = 13
                    add_strassen_7(U_full, V_full, W_full, cs, 4)
                    cs2 = cs + 7
                    U_full[0, cs2+0] += 1; V_full[0*7+6, cs2+0] += 1; W_full[6*n+0, cs2+0] += 1
                    U_full[1, cs2+1] += 1; V_full[1*7+6, cs2+1] += 1; W_full[6*n+0, cs2+1] += 1
                    U_full[2, cs2+2] += 1; V_full[0*7+6, cs2+2] += 1; W_full[6*n+1, cs2+2] += 1
                    U_full[3, cs2+3] += 1; V_full[1*7+6, cs2+3] += 1; W_full[6*n+1, cs2+3] += 1
                    
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        except Exception:
            pass
            
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
