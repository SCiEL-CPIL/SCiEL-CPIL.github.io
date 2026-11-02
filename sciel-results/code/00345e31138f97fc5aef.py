"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _construct_rank15_233():
    """
    Attempt to find a rank-15 exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Returns (U, V, W) as numpy arrays or None if not found.
    """
    n, m, p = 2, 3, 3
    # Tensor T is (n*m) x (m*p) x (p*n) = 6 x 9 x 6
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    target_rank = 15
    start_time = time.time()
    deadline = start_time + 30  # 30 seconds for sub-problem
    
    # Try to find decomposition
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U) + 1e-10
        V /= np.linalg.norm(V) + 1e-10
        W /= np.linalg.norm(W) + 1e-10
        
        # Alternating Least Squares
        converged = False
        for iteration in range(500):
            if time.time() > deadline:
                break
                
            # Solve for U
            # T_vec = sum_r U[:,r] (V[:,r] (kron) W[:,r])
            # (V_kr_W)^T (V_kr_W) U = (V_kr_W)^T T_vec
            VkrW = np.zeros((m*p * p*n, target_rank))
            for r in range(target_rank):
                VkrW[:, r] = np.kron(V[:, r], W[:, r])
            
            try:
                U_new, _, _, _ = np.linalg.lstsq(VkrW, T.reshape(-1), rcond=None)
            except:
                break
            U_new = U_new.reshape(n*m, target_rank)
            
            # Solve for V
            WkrU = np.zeros((p*n * n*m, target_rank))
            for r in range(target_rank):
                WkrU[:, r] = np.kron(W[:, r], U[:, r])
            try:
                V_new, _, _, _ = np.linalg.lstsq(WkrU, T.reshape(-1), rcond=None)
            except:
                break
            V_new = V_new.reshape(m*p, target_rank)
            
            # Solve for W
            UkrV = np.zeros((n*m * m*p, target_rank))
            for r in range(target_rank):
                UkrV[:, r] = np.kron(U[:, r], V[:, r])
            try:
                W_new, _, _, _ = np.linalg.lstsq(UkrV, T.reshape(-1), rcond=None)
            except:
                break
            W_new = W_new.reshape(p*n, target_rank)
            
            U, V, W = U_new, V_new, W_new
            
            # Check error
            recon = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            error = np.max(np.abs(recon - T))
            
            if error < 1e-8:
                converged = True
                break
        
        if converged:
            # Round to integers/half-integers if possible
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            recon_r = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                recon_r += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
            
            if np.max(np.abs(recon_r - T)) < 1e-8:
                return U_r, V_r, W_r
            else:
                return U, V, W
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Strategy: Decompose into two 2x3*3x3 multiplications.
    If we find a rank-15 decomposition for 2x3*3x3, we can construct a rank-30 decomposition for 2x3*3x6.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try to construct rank-30 decomposition via rank-15 2x3x3
    # 2x3x6 is equivalent to two independent 2x3x3 multiplications
    # C[0:3] comes from B[0:3], C[3:6] comes from B[3:6]
    
    res_233 = _construct_rank15_233()
    
    if res_233 is not None and time.time() < deadline:
        U_233, V_233, W_233 = res_233
        rank_233 = U_233.shape[1]
        
        # Construct 2x3x6 decomposition
        # U is (6, rank_233*2)
        # V is (18, rank_233*2)
        # W is (12, rank_233*2)
        
        # The first rank_233 columns correspond to the first 3 outputs (k=0,1,2)
        # The second rank_233 columns correspond to the last 3 outputs (k=3,4,5)
        
        U_full = np.zeros((n*m, rank_233 * 2))
        V_full = np.zeros((m*p, rank_233 * 2))
        W_full = np.zeros((p*n, rank_233 * 2))
        
        # First half (k=0,1,2)
        # U is same for both halves?
        # In 2x3x3, U[i*m+j, r] depends on i,j.
        # In 2x3x6, U[i*m+j, r] depends on i,j and which block of k.
        # Since the A matrix is the same, the U factors for the two blocks are identical in structure,
        # but they are distinct columns in the full decomposition.
        
        # V factors:
        # For the first block (k=0,1,2), V[j*p+k, r] = V_233[j*3+k, r]
        # For the second block (k=3,4,5), V[j*p+k, r] = V_233[j*3+(k-3), r + rank_233]
        
        # W factors:
        # For the first block, W[k*n+i, r] = W_233[k*2+i, r]
        # For the second block, W[k*n+i, r] = W_233[(k-3)*2+i, r + rank_233]
        
        # Fill U
        U_full[:, :rank_233] = U_233
        U_full[:, rank_233:] = U_233
        
        # Fill V
        for j in range(m):
            for k in range(3):
                V_full[j*p + k, :rank_233] = V_233[j*3 + k, :rank_233]
                V_full[j*p + 3 + k, rank_233:] = V_233[j*3 + k, :rank_233]
        
        # Fill W
        for k in range(3):
            for i in range(n):
                W_full[k*n + i, :rank_233] = W_233[k*2 + i, :rank_233]
                W_full[(3+k)*n + i, rank_233:] = W_233[k*2 + i, :rank_233]
        
        # Verify the full decomposition
        T_full = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_full[a, b, c] = 1.0
        
        recon_full = np.zeros((n*m, m*p, p*n))
        for r in range(rank_233 * 2):
            recon_full += np.einsum('a,b,c->abc', U_full[:, r], V_full[:, r], W_full[:, r])
        
        error = np.max(np.abs(recon_full - T_full))
        
        if error < 1e-8:
            # Return the rank-30 decomposition
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
        else:
            # Fallback to schoolbook if verification fails
            return {
                "U": U_school.tolist(),
                "V": V_school.tolist(),
                "W": W_school.tolist()
            }
    else:
        # Fallback to schoolbook
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
# EVOLVE-BLOCK-END
