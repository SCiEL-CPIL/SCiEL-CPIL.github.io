"""Matrix multiplication 3x3x7 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    # Compute tensor: T_recon[i,j,k] = sum_t U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U3[i, j, t] * V3[j, k, t] * W3[k, i, t]
                T_recon[i, j, k] = s
    return np.allclose(T_recon, T, atol=tol, rtol=0)

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _als_step(U, V, W, n, m, p, rank, max_iter=100, tol=1e-12):
    """
    Alternating least squares to refine U,V,W given fixed rank.
    We want: for all i,j,k: sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1
    This is a bilinear system in each factor given the others fixed.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for it in range(max_iter):
        # Fix V, W; solve for U
        # For each (i,j): sum_t U[i,j,t] * (sum_k V[j,k,t]*W[k,i,t]) = 1
        # This is: U[i,j,:] . g_ij = 1 where g_ij[t] = sum_k V[j,k,t]*W[k,i,t]
        # But we also want minimal norm or some constraint. Let's use least squares.
        # Actually, the constraint is just that the contraction equals 1.
        # We can set U[i,j,:] to minimize ||U[i,j,:]|^2 subject to U[i,j,:].g_ij = 1
        # Solution: U[i,j,:] = g_ij / ||g_ij||^2 (if g_ij != 0)
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                g = np.zeros(rank)
                for k in range(p):
                    g += V[j, k * 1 + k, :]  # This is wrong, let me fix indexing
                # V has shape (m*p, rank), so V[j*p + k, t]
                # W has shape (p*n, rank), so W[k*n + i, t]
                g = np.zeros(rank)
                for k in range(p):
                    g += V[j * p + k, :] * W[k * n + i, :]
                if np.linalg.norm(g) > 1e-15:
                    U_new[i * m + j, :] = g / (g @ g)
                else:
                    U_new[i * m + j, :] = 0
        
        # Fix U, W; solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
            # h[t] = sum_i U[i,j,t] * W[k,i,t]
            # V[j,k,:] . h = 1
                h = np.zeros(rank)
                for i in range(n):
                    h += U[i * m + j, :] * W[k * n + i, :]
                if np.linalg.norm(h) > 1e-15:
                    V_new[j * p + k, :] = h / (h @ h)
                else:
                    V_new[j * p + k, :] = 0
        
        # Fix U, V; solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # q[t] = sum_j U[i,j,t] * V[j,k,t]
                # W[k,i,:] . q = 1
                q = np.zeros(rank)
                for j in range(m):
                    q += U[i * m + j, :] * V[j * p + k, :]
                if np.linalg.norm(q) > 1e-15:
                    W_new[k * n + i, :] = q / (q @ q)
                else:
                    W_new[k * n + i, :] = 0
        
        # Check convergence
        diff = np.max(np.abs(U_new - U) + np.abs(V_new - V) + np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        if diff < tol:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = 115  # safety margin before 120s
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 63
    
    # Try to reduce rank by searching for valid lower-rank decompositions
    # Strategy: try ranks from low to high, use random initialization + ALS
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But for matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # Actually, rank >= max(n*m, m*p, p*n) is a lower bound.
    # For 3x3x7: max(9, 21, 21) = 21
    
    # Let's try various ranks
    for rank in range(21, min(63, 50)):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 25:
                break
            
            # Random initialization
            np.random.seed(rank * 1000 + trial)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Run ALS
            U, V, W = _als_step(U, V, W, n, m, p, rank, max_iter=200)
            
            # Check if valid
            if _check_decomposition(U, V, W, n, m, p, tol=1e-8):
                # Round to nearest half-integer if possible
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break
        
        if best_rank == rank:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
