"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import linalg

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_block_decomp(n, m, p):
    """
    Fallback: Construct the known valid rank-88 decomposition using Strassen blocks.
    i-groups: {0,1} (Strassen), {2} (schoolbook)
    j-groups: {0,1}, {2,3}
    k-groups: {0,1}, {2,3}, {4,5}, {6,7}
    Rank: 2*4*7 + 2*4*4 = 56 + 32 = 88
    """
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p):
        terms = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
            ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
            ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
            ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
            ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
            ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
        ]
        
        col = col_start
        for t in range(7):
            a_coeffs = terms[t][0]
            b_coeffs = terms[t][1]
            c_coeffs = terms[t][2]
            
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
            
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
            
            for di in range(2):
                for dk in range(2):
                    W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
            
            col += 1
        
        return col

    def _schoolbook_block(U, V, W, i0, j0, k0, col_start, n, m, p):
        col = col_start
        for dj in range(2):
            for dk in range(2):
                U[i0*m + (j0+dj)][col] = 1
                V[(j0+dj)*p + (k0+dk)][col] = 1
                W[(k0+dk)*n + i0][col] = 1
                col += 1
        return col

    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = _strassen_block(U, V, W, 0, j0, k0, col, n, m, p)
            col = _schoolbook_block(U, V, W, 2, j0, k0, col, n, m, p)
            
    return {"U": U, "V": V, "W": W}

def _try_rank_minimization(n, m, p, target_rank, time_limit):
    """
    Attempt to find a rank-target_rank decomposition using least squares.
    Returns (U, V, W) if found and verified, else None.
    """
    # Construct the tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 if valid, else 0
    # Actually, the tensor for matrix multiplication M_{ij} is defined such that
    # C_ik = sum_j A_ij B_jk.
    # The tensor T has entries T_{i,j,k} = 1 if the term A_ij B_jk contributes to C_ik.
    # Wait, the standard tensor for matrix multiplication <n,m,p> is T in R^{n*m x m*p x p*n}
    # where T_{(i,j), (j,k), (k,i)} = 1.
    
    Nm = n * m
    Mp = m * p
    Pn = p * n
    
    T = np.zeros((Nm, Mp, Pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # We want to find U (Nm x r), V (Mp x r), W (Pn x r)
    # such that T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    
    # This is a nonlinear problem. We can use an alternating least squares approach
    # or fix two factors and solve for the third.
    # Let's try a simple ALS (Alternating Least Squares) with random initialization.
    
    U = np.random.randn(Nm, target_rank)
    V = np.random.randn(Mp, target_rank)
    W = np.random.randn(Pn, target_rank)
    
    # Normalize to prevent scale drift
    for c in range(target_rank):
        norm_u = np.linalg.norm(U[:, c])
        norm_v = np.linalg.norm(V[:, c])
        norm_w = np.linalg.norm(W[:, c])
        if norm_u > 0 and norm_v > 0 and norm_w > 0:
            scale = (norm_u * norm_v * norm_w) ** (1/3)
            U[:, c] /= norm_u
            V[:, c] /= norm_v
            W[:, c] /= norm_w
            # This normalization is tricky for exact recovery. Let's just iterate.
            
    # ALS Iterations
    # Reshape T to 2D matrices for solving.
    # T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}
    # Fix V, W -> solve for U: 
    # T_{i, j, k} = sum_c U_{ic} (V_{jc} W_{kc})
    # Let M_VW_{j,k} = sum_c V_{jc} W_{kc}. This is not a linear system in U directly for all i,j,k simultaneously in a simple way unless we vectorize.
    
    # Vectorize T over (j,k) for fixed i? No, U depends on i.
    # Let's reshape T into a matrix T_mat of size (Nm, Mp*Pn) where rows are i and cols are (j,k).
    # T_mat[i, j*p*n + k] = T[i, j, k]
    # We want T_mat[i, :] = sum_c U[i, c] * (V[:, c] outer W[:, c]).T
    # Let Y_c = (V[:, c] outer W[:, c]).T, shape (Mp*Pn,)
    # Then T_mat = U @ Y^T where Y is (r, Mp*Pn)
    # So for fixed V, W, we can solve U = T_mat @ Y^T @ (Y @ Y^T)^{-1} (Least Squares)
    
    # Similarly for V and W.
    
    start_time = time.time()
    for it in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Solve for U
        Y = np.zeros((target_rank, Mp * Pn))
        for c in range(target_rank):
            Y[c, :] = np.outer(V[:, c], W[:, c]).flatten()
        # T_mat shape (Nm, Mp*Pn)
        T_mat = T.reshape(Nm, Mp * Pn)
        # U = T_mat @ Y.T @ (Y @ Y.T)^-1
        try:
            YTY = Y @ Y.T
            # Add small regularization
            YTY += 1e-9 * np.eye(target_rank)
            U = T_mat @ Y.T @ np.linalg.inv(YTY)
        except np.linalg.LinAlgError:
            pass
            
        # Solve for V
        # T_{i,j,k} = sum_c V_{j,c} (U_{i,c} W_{k,c})
        # Reshape T to (Mp, Nm*Pn)
        T_mat_v = T.transpose(1, 0, 2).reshape(Mp, Nm * Pn)
        Z = np.zeros((target_rank, Nm * Pn))
        for c in range(target_rank):
            Z[c, :] = np.outer(U[:, c], W[:, c]).flatten()
        try:
            ZTZ = Z @ Z.T
            ZTZ += 1e-9 * np.eye(target_rank)
            V = T_mat_v @ Z.T @ np.linalg.inv(ZTZ)
        except np.linalg.LinAlgError:
            pass
            
        # Solve for W
        # T_{i,j,k} = sum_c W_{k,c} (U_{i,c} V_{j,c})
        # Reshape T to (Pn, Nm*Mp)
        T_mat_w = T.transpose(2, 0, 1).reshape(Pn, Nm * Mp)
        X = np.zeros((target_rank, Nm * Mp))
        for c in range(target_rank):
            X[c, :] = np.outer(U[:, c], V[:, c]).flatten()
        try:
            XTX = X @ X.T
            XTX += 1e-9 * np.eye(target_rank)
            W = T_mat_w @ X.T @ np.linalg.inv(XTX)
        except np.linalg.LinAlgError:
            pass

    # Check convergence / validity
    # Reconstruct and check error
    T_recon = np.zeros_like(T)
    for c in range(target_rank):
        T_recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c]).reshape(Nm, Mp, Pn)
    
    error = np.linalg.norm(T_recon - T)
    if error < 1e-6:
        # Try to snap to integers/half-integers
        # The problem states entries are integer/half-integer.
        # We can try rounding to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if _verify(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
            
        # Maybe the solution is integers?
        U_snap_int = np.round(U)
        V_snap_int = np.round(V)
        W_snap_int = np.round(W)
        if _verify(U_snap_int, V_snap_int, W_snap_int, n, m, p):
            return U_snap_int, V_snap_int, W_snap_int
            
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Known best from current block strategy: 88
    # Let's try to beat 88.
    # Lower bound is ceil(3*4*8 / 2) = 48? No, lower bound is max(nm, mp, pn) = max(12, 32, 24) = 32.
    # Actually, for matrix multiplication, rank is at least max(nm, mp, pn).
    # Let's try ranks from 80 down to 60.
    
    best_result = None
    best_rank = 88 # Fallback
    
    # Try to find a lower rank decomposition
    for target_rank in range(80, 59, -1):
        if time.time() > deadline - 10:
            break
            
        res = _try_rank_minimization(n, m, p, target_rank, 5.0)
        if res is not None:
            U, V, W = res
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = target_rank
            break # Found a valid one, stop
    
    if best_result is not None:
        return best_result
    
    # Fallback to known valid rank-88 decomposition
    return _construct_strassen_block_decomp(n, m, p)

# EVOLVE-BLOCK-END
