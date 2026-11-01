import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 28 decomposition (Fallback)
    def build_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            # M1
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M2
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = -1
            col += 1
            # M3
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M4
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = -1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M5
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            col += 1
            # M6
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (0 + j_off)][col] = -1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M7
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = -1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            col += 1

        # Block 1: j=0
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
        # Block 2: j=1,2
        add_strassen(0, 1, 0)
        
        for i in range(n):
            for j in [1, 2]:
                U[i * m + j][col] = 1
                V[j * p + 2][col] = 1
                W[2 * n + i][col] = 1
                col += 1
                
        add_strassen(0, 1, 3)
        
        assert col == rank, f"Column count mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    # Try to find Rank 27
    def try_find_rank_27():
        rank = 27
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # ALS Optimization
        # Target: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # For fixed V, W, solve for U?
        # U[i,j,r] V[j,k,r] W[k,i,r] is linear in U if V,W fixed.
        # Sum over r: sum_r U[i,j,r] (V[j,k,r] W[k,i,r]) = 1.
        # This is a linear system for U[i,j,:].
        # Let A[i,j] be the vector of length R.
        # sum_r A[i,j,r] B[i,j,k,r] = 1, where B[i,j,k,r] = V[j,k,r] W[k,i,r].
        # This is a system of m*p equations for each (i,j)?
        # No, for each (i,j), we have p equations (one for each k).
        # Unknowns: R. Equations: p.
        # If p < R, underdetermined. We can use least squares.
        
        # Similarly for V and W.
        
        max_iters = 50
        start_time = time.time()
        
        for it in range(max_iters):
            if time.time() - start_time > 10:
                break
            
            # Update U
            # For each i, j:
            # sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1 for all k
            # Let M[i,j] be the matrix of size p x R where M[i,j,k,r] = V[j,k,r] * W[k,i,r]
            # We want M[i,j] @ U[i,j,:] = ones(p)
            # Solve least squares
            for i in range(n):
                for j in range(m):
                    M_ij = np.zeros((p, rank))
                    for k in range(p):
                        M_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve M_ij @ u = ones
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)[0]
                    except:
                        pass

            # Update V
            # For each j, k:
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i
            # Let M[j,k] be matrix of size n x R where M[j,k,i,r] = U[i,j,r] * W[k,i,r]
            # We want M[j,k] @ V[j,k,:] = ones(n)
            for j in range(m):
                for k in range(p):
                    M_jk = np.zeros((n, rank))
                    for i in range(n):
                        M_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M_jk, np.ones(n), rcond=None)[0]
                    except:
                        pass

            # Update W
            # For each k, i:
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all j
            # Let M[k,i] be matrix of size m x R where M[k,i,j,r] = U[i,j,r] * V[j,k,r]
            # We want M[k,i] @ W[k,i,:] = ones(m)
            for k in range(p):
                for i in range(n):
                    M_ki = np.zeros((m, rank))
                    for j in range(m):
                        M_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M_ki, np.ones(m), rcond=None)[0]
                    except:
                        pass

            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            
            if err < 1e-8:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped solution
                err_snap = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_snap[i * m + j, :], V_snap[j * p + k, :] * W_snap[k * n + i, :])
                            err_snap += (val - 1.0) ** 2
                
                if err_snap < 1e-6:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
        
        return None

    res_27 = try_find_rank_27()
    if res_27:
        return res_27
    else:
        return build_rank_28()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
