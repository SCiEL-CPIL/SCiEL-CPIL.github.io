"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _als_decompose(T, rank, max_iter=100, tol=1e-8, deadline=None):
    """
    Alternating Least Squares for tensor decomposition.
    T is (n1, n2, n3).
    Returns U (n1, rank), V (n2, rank), W (n3, rank) or None.
    """
    n1, n2, n3 = T.shape
    vecT = T.reshape(-1)
    
    # Initialize
    np.random.seed(42) # For reproducibility in testing, but maybe random is better for search
    U = np.random.randn(n1, rank)
    V = np.random.randn(n2, rank)
    W = np.random.randn(n3, rank)
    
    # Normalize
    U /= np.linalg.norm(U) + 1e-10
    V /= np.linalg.norm(V) + 1e-10
    W /= np.linalg.norm(W) + 1e-10
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            return None
        
        # Solve for U
        # V_kr_W: (n2*n3, rank)
        VkrW = np.zeros((n2 * n3, rank))
        for r in range(rank):
            VkrW[:, r] = np.kron(V[:, r], W[:, r])
        
        try:
            U_new, _, _, _ = np.linalg.lstsq(VkrW, vecT, rcond=None)
        except:
            return None
        U_new = U_new.reshape(n1, rank)
        
        # Solve for V
        WkrU = np.zeros((n3 * n1, rank))
        for r in range(rank):
            WkrU[:, r] = np.kron(W[:, r], U[:, r])
        try:
            V_new, _, _, _ = np.linalg.lstsq(WkrU, vecT, rcond=None)
        except:
            return None
        V_new = V_new.reshape(n2, rank)
        
        # Solve for W
        UkrV = np.zeros((n1 * n2, rank))
        for r in range(rank):
            UkrV[:, r] = np.kron(U[:, r], V[:, r])
        try:
            W_new, _, _, _ = np.linalg.lstsq(UkrV, vecT, rcond=None)
        except:
            return None
        W_new = W_new.reshape(n3, rank)
        
        U, V, W = U_new, V_new, W_new
        
        # Check error
        recon = np.zeros((n1, n2, n3))
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        error = np.max(np.abs(recon - T))
        if error < tol:
            return U, V, W
            
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = time.time() + 140  # 10s safety margin
    
    # We exploit the block structure: 2x3 * 3x6 = (2x3 * 3x3) x 2
    # Rank(2x3x6) <= 2 * Rank(2x3x3)
    
    # Step 1: Find rank of 2x3x3 tensor
    n_sub, m_sub, p_sub = 2, 3, 3
    T_sub = _build_tensor(n_sub, m_sub, p_sub)
    
    best_rank_sub = None
    best_U_sub = None
    best_V_sub = None
    best_W_sub = None
    
    # Try ranks from 12 to 18
    for r in range(12, 19):
        if time.time() > deadline - 20:
            break
            
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(T_sub, r, max_iter=50, deadline=deadline - 10)
            if result is not None:
                U_sub, V_sub, W_sub = result
                best_rank_sub = r
                best_U_sub = U_sub
                best_V_sub = V_sub
                best_W_sub = W_sub
                break
        
        if best_rank_sub is not None:
            break
    
    # Step 2: Construct 2x3x6 decomposition from 2x3x3 decomposition
    if best_rank_sub is not None:
        r_sub = best_rank_sub
        r_total = 2 * r_sub
        
        # U is (2*3, r_total) = (6, 2*r_sub)
        U = np.zeros((n * m, r_total))
        # U is the same for both blocks
        U[:, 0:r_sub] = best_U_sub
        U[:, r_sub:2*r_sub] = best_U_sub
        
        # V is (3*6, r_total) = (18, 2*r_sub)
        V = np.zeros((m * p, r_total))
        # V rows: j * 6 + k
        # Block 0 (k=0,1,2): rows 0,1,2, 3,4,5, 6,7,8 correspond to V_sub rows 0,1,2, 3,4,5, 6,7,8
        # V_sub is (3*3, r_sub) = (9, r_sub)
        # V_sub row j*3+k
        for j in range(m_sub):
            for k in range(p_sub):
                v_row_sub = j * p_sub + k
                # Block 0
                v_row_full_0 = j * p + k
                V[v_row_full_0, 0:r_sub] = best_V_sub[v_row_sub, :]
                # Block 1
                v_row_full_1 = j * p + (k + p_sub)
                V[v_row_full_1, r_sub:2*r_sub] = best_V_sub[v_row_sub, :]
        
        # W is (6*2, r_total) = (12, 2*r_sub)
        W = np.zeros((p * n, r_total))
        # W rows: k * 2 + i
        # W_sub is (3*2, r_sub) = (6, r_sub)
        # W_sub row k*2+i
        for k in range(p_sub):
            for i in range(n_sub):
                w_row_sub = k * n_sub + i
                # Block 0
                w_row_full_0 = k * n + i
                W[w_row_full_0, 0:r_sub] = best_W_sub[w_row_sub, :]
                # Block 1
                w_row_full_1 = (k + p_sub) * n + i
                W[w_row_full_1, r_sub:2*r_sub] = best_W_sub[w_row_sub, :]
        
        # Verify
        T_full = _build_tensor(n, m, p)
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(r_total):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        if np.max(np.abs(recon - T_full)) < 1e-8:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook rank
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
