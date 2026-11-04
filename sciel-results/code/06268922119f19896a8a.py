import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Pre-compute indices
    # Equation l: U[u_idx, l] * V[v_idx, l] * W[w_idx, l] = 1
    # This is not a standard linear system for U if V, W vary with l.
    # Wait, for a fixed l, the term is U[u, l] * (V[v, l] * W[w, l]).
    # The sum is over l.
    # So for a fixed equation (i,j,k), the LHS is sum_l U[u, l] * A_l where A_l = V[v, l] * W[w, l].
    # This is linear in U.
    
    # We can construct a matrix M_U of size (n*m*p, R)
    # M_U[eq_idx, l] = V[v_idx, l] * W[w_idx, l]
    # Then we want M_U @ U_flat = 1 (vector of ones)
    # U_flat is (R * dim_U, 1)? No.
    # U is (dim_U, R).
    # The equation is sum_l U[u, l] * M_U[eq, l] = 1.
    # This is not a standard matrix multiplication U @ M_U^T.
    # It is a weighted sum.
    
    # We can reshape this.
    # Let's just use iterative updates.
    
    def als_decompose(rank, max_iters=100, tol=1e-6):
        np.random.seed(0)
        U = np.random.randn(dim_U, rank)
        V = np.random.randn(dim_V, rank)
        W = np.random.randn(dim_W, rank)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for _ in range(max_iters):
            # Update U
            # For each u_idx, we want to find U[u_idx, :]
            # The equations involving u_idx are those where i*m+j == u_idx.
            # Let S_u be the set of equations (v_idx, w_idx) for this u_idx.
            # We want sum_l U[u, l] * V[v, l] * W[w, l] = 1 for all (v, w) in S_u.
            # This is a system of linear equations for U[u, :].
            # Let B_u be a matrix of size (|S_u|, rank) where B_u[row, l] = V[v, l] * W[w, l].
            # We want B_u @ U[u, :] = 1.
            
            U_new = np.zeros_like(U)
            for u_idx in range(dim_U):
                # Find all v_idx, w_idx such that i*m+j == u_idx
                i = u_idx // m
                j = u_idx % m
                # v_idx = j*p+k, w_idx = k*n+i
                # k ranges from 0 to p-1
                rows = []
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    row = V[v_idx] * W[w_idx]
                    rows.append(row)
                B_u = np.array(rows)
                # Solve B_u @ x = 1
                # Least squares
                try:
                    x, residuals, rank_u, s = np.linalg.lstsq(B_u, np.ones(len(rows)), rcond=None)
                    U_new[u_idx] = x
                except:
                    U_new[u_idx] = U[u_idx]
            
            # Update V
            V_new = np.zeros_like(V)
            for v_idx in range(dim_V):
                j = v_idx // p
                k = v_idx % p
                # u_idx = i*m+j, w_idx = k*n+i
                # i ranges from 0 to n-1
                rows = []
                for i in range(n):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    row = U_new[u_idx] * W[w_idx]
                    rows.append(row)
                B_v = np.array(rows)
                try:
                    x, residuals, rank_v, s = np.linalg.lstsq(B_v, np.ones(len(rows)), rcond=None)
                    V_new[v_idx] = x
                except:
                    V_new[v_idx] = V[v_idx]
            
            # Update W
            W_new = np.zeros_like(W)
            for w_idx in range(dim_W):
                k = w_idx // n
                i = w_idx % n
                # u_idx = i*m+j, v_idx = j*p+k
                # j ranges from 0 to m-1
                rows = []
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    row = U_new[u_idx] * V_new[v_idx]
                    rows.append(row)
                B_w = np.array(rows)
                try:
                    x, residuals, rank_w, s = np.linalg.lstsq(B_w, np.ones(len(rows)), rcond=None)
                    W_new[w_idx] = x
                except:
                    W_new[w_idx] = W[w_idx]
            
            U, V, W = U_new, V_new, W_new
            
        # Check error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    error += (val - 1.0)**2
        return U, V, W, error

    # Try ranks
    for rank in [15, 20, 25, 30, 45]:
        U, V, W, error = als_decompose(rank, max_iters=50)
        if error < 1e-4:
            # Try to rationalize
            # This is a heuristic
            pass
        if rank == 45:
            break
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
