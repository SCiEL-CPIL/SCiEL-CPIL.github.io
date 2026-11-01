import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # n=2, m=2, p=9

    # Trivial decomposition fallback (Rank 36)
    def trivial_decomposition():
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m+jj][c] = 1
                    V_t[jj*p+kk][c] = 1
                    W_t[kk*n+ii][c] = 1
                    c += 1
        return {"U": U_t, "V": V_t, "W": W_t}

    # Build rank-32 decomposition
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1; V[k0][col] = 1; V[p+k1][col] = 1; W[2*k0][col] = 1; W[2*k1+1][col] = 1; col += 1
            U[2][col] = 1; U[3][col] = 1; V[k0][col] = 1; W[2*k0+1][col] = 1; W[2*k1+1][col] = -1; col += 1
            U[0][col] = 1; V[k1][col] = 1; V[p+k1][col] = -1; W[2*k1][col] = 1; W[2*k1+1][col] = 1; col += 1
            U[3][col] = 1; V[p+k0][col] = 1; V[k0][col] = -1; W[2*k0][col] = 1; W[2*k0+1][col] = 1; col += 1
            U[0][col] = 1; U[1][col] = 1; V[p+k1][col] = 1; W[2*k0][col] = -1; W[2*k1][col] = 1; col += 1
            U[2][col] = 1; U[0][col] = -1; V[k0][col] = 1; V[k1][col] = 1; W[2*k1+1][col] = 1; col += 1
            U[1][col] = 1; U[3][col] = -1; V[p+k0][col] = 1; V[p+k1][col] = 1; W[2*k0][col] = 1; col += 1

        def add_schoolbook(k):
            nonlocal col
            U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
            U[1][col] = 1; V[p+k][col] = 1; W[2*k][col] = 1; col += 1
            U[2][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
            U[3][col] = 1; V[p+k][col] = 1; W[2*k+1][col] = 1; col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_schoolbook(8)
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    U32, V32, W32 = build_rank32()

    # Verify
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify(U32, V32, W32):
        return trivial_decomposition()

    # Try rank-31: merge the last schoolbook term into the last Strassen block
    # The last 7+4=11 terms cover Strassen(6,7) + schoolbook(8)
    # Try to represent these 11 terms as 10 terms
    
    def try_rank31():
        if time.time() > deadline - 10:
            return None
        
        rank = 31
        # Initialize: take first 31 columns, but the 32nd column (index 31) is A[1,1]*B[1,8]->C[1,8]
        # We'll optimize all 31 columns starting from the rank-32 init with col 31 zeroed
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Use first 31 columns as init
        U[:, :] = U32[:, :31]
        V[:, :] = V32[:, :31]
        W[:, :] = W32[:, :31]
        
        def objective_and_grad(params):
            U_f = params[:n*m*rank].reshape(n*m, rank)
            V_f = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
            W_f = params[2*n*m*rank:].reshape(p*n, rank)
            
            obj = 0.0
            dU = np.zeros_like(U_f)
            dV = np.zeros_like(V_f)
            dW = np.zeros_like(W_f)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U_f[i*m+j, :] * V_f[j*p+k, :] * W_f[k*n+i, :]
                        val = np.sum(prod)
                        err = 2.0 * (val - 1.0)
                        obj += err * (val - 1.0)
                        dU[i*m+j, :] += err * V_f[j*p+k, :] * W_f[k*n+i, :]
                        dV[j*p+k, :] += err * U_f[i*m+j, :] * W_f[k*n+i, :]
                        dW[k*n+i, :] += err * U_f[i*m+j, :] * V_f[j*p+k, :]
            
            return obj, np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(lambda x: objective_and_grad(x)[0], x0,
                             jac=lambda x: objective_and_grad(x)[1],
                             method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < 1e-12:
                U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                V_opt = result.x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W_opt = result.x[2*n*m*rank:].reshape(p*n, rank)
                
                if verify(U_opt, V_opt, W_opt, tol=1e-8):
                    U_list = [[float(U_opt[i][j]) for j in range(rank)] for i in range(n*m)]
                    V_list = [[float(V_opt[i][j]) for j in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_opt[i][j]) for j in range(rank)] for i in range(p*n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
        
        return None

    result = try_rank31()
    if result is not None:
        return result

    # Return rank-32
    U_list = [[float(U32[i][j]) for j in range(32)] for i in range(n*m)]
    V_list = [[float(V32[i][j]) for j in range(32)] for i in range(m*p)]
    W_list = [[float(W32[i][j]) for j in range(32)] for i in range(p*n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
