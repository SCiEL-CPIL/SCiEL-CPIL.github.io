import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _make_block(n, m, p, offset_i=0, offset_j=0, offset_k=0, rank_override=None):
    """
    Creates a decomposition for an n x m x p block.
    If rank_override is provided, it uses a specific strategy.
    Otherwise, it uses schoolbook.
    Returns U, V, W matrices for this block, sized appropriately for the full tensor.
    """
    if rank_override == "rank1_m":
        # n x 1 x p, Rank 1
        r = 1
        U = np.zeros((n * 1, r))
        V = np.zeros((1 * p, r))
        W = np.zeros((p * n, r))
        
        # U[i, 0] = 1
        for i in range(n):
            U[i, 0] = 1.0
        # V[0, 0] = 1 (for all k, V[k*1+0, 0] = 1? No, V index is j*p+k. j=0. So V[k, 0] = 1)
        for k in range(p):
            V[k, 0] = 1.0
        # W[k*n+i, 0] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i, 0] = 1.0
        
        return U, V, W, r

    elif rank_override == "rank_nm_p2":
        # n x m x 2, Rank n*m
        r = n * m
        U = np.zeros((n * m, r))
        V = np.zeros((m * 2, r))
        W = np.zeros((2 * n, r))
        
        for i in range(n):
            for j in range(m):
                c = i * m + j
                U[i * m + j, c] = 1.0
                # V[j*2 + k, c] = 1 for k=0,1
                V[j * 2 + 0, c] = 1.0
                V[j * 2 + 1, c] = 1.0
                # W[k*n + i, c] = 1 for k=0,1
                W[0 * n + i, c] = 1.0
                W[1 * n + i, c] = 1.0
                
        return U, V, W, r

    else:
        # Schoolbook n x m x p, Rank n*m*p
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                    
        return U, V, W, r

def _embed_block(U_block, V_block, W_block, r_block, n_full, m_full, p_full, 
                 i_start, j_start, k_start, i_end, j_end, k_end):
    """
    Embeds a block decomposition into the full size matrices.
    """
    n_b = i_end - i_start
    m_b = j_end - j_start
    p_b = k_end - k_start
    
    U_full = np.zeros((n_full * m_full, r_block))
    V_full = np.zeros((m_full * p_full, r_block))
    W_full = np.zeros((p_full * n_full, r_block))
    
    # U: indices i*m_full + j
    for i_b in range(n_b):
        i_full = i_start + i_b
        for j_b in range(m_b):
            j_full = j_start + j_b
            U_full[i_full * m_full + j_full, :] = U_block[i_b * m_b + j_b, :]
            
    # V: indices j*p_full + k
    for j_b in range(m_b):
        j_full = j_start + j_b
        for k_b in range(p_b):
            k_full = k_start + k_b
            V_full[j_full * p_full + k_full, :] = V_block[j_b * p_b + k_b, :]
            
    # W: indices k*n_full + i
    for k_b in range(p_b):
        k_full = k_start + k_b
        for i_b in range(n_b):
            i_full = i_start + i_b
            W_full[k_full * n_full + i_full, :] = W_block[k_b * n_b + i_b, :]
            
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Full matrices
    total_rank = 0
    U_total = None
    V_total = None
    W_total = None
    
    def add_block(n_b, m_b, p_b, i_start, j_start, k_start, strategy=None):
        nonlocal U_total, V_total, W_total, total_rank
        U_b, V_b, W_b, r_b = _make_block(n_b, m_b, p_b, rank_override=strategy)
        U_e, V_e, W_e = _embed_block(U_b, V_b, W_b, r_b, n, m, p, 
                                     i_start, j_start, k_start, 
                                     i_start + n_b, j_start + m_b, k_start + p_b)
        
        if U_total is None:
            U_total = U_e
            V_total = V_e
            W_total = W_e
        else:
            U_total = np.hstack([U_total, U_e])
            V_total = np.hstack([V_total, V_e])
            W_total = np.hstack([W_total, W_e])
        total_rank += r_b

    # Split K into 0-4 (size 5) and 5-6 (size 2)
    # Split J into 0-3 (size 4) and 4 (size 1)
    
    # Block 1: J=0-3, K=0-3 (3x4x4) -> Schoolbook
    add_block(3, 4, 4, 0, 0, 0)
    
    # Block 2: J=0-3, K=4 (3x4x1) -> Rank 1
    add_block(3, 4, 1, 0, 0, 4, strategy="rank1_m") # Note: rank1_m assumes m=1 in U/V dims? 
    # Wait, _make_block "rank1_m" logic:
    # U shape (n*m, 1). If m=4, U is 12x1.
    # V shape (m*p, 1). If p=1, V is 4x1.
    # W shape (p*n, 1). If p=1, W is 3x1.
    # Logic in _make_block for rank1_m:
    # U[i, 0] = 1 for i in range(n). But U has n*m rows.
    # This logic is flawed for m != 1.
    
    # I need a generic rank-1 block for n x m x p?
    # No, rank 1 is only possible if the tensor is separable.
    # T_{ijk} = 1 is NOT separable for m>1 or p>1.
    # T_{ijk} = 1 means C_{ik} = sum_j A_{ij} B_{jk}.
    # If m=1, C_{ik} = A_{i} B_{k}. This is rank 1.
    # If p=1, C_{i} = sum_j A_{ij} B_{j}. This is rank 1?
    # T_{ij1} = 1.
    # Sum_c U_{ij}^{(c)} V_{j1}^{(c)} W_{1i}^{(c)} = 1.
    # If we use rank 1: U_{ij} = 1, V_{j1} = 1, W_{1i} = 1.
    # Sum = 1.
    # So Rank(n, m, 1) is 1.
    # Rank(n, 1, p) is 1.
    # Rank(1, m, p) is 1.
    
    # So I need to handle these cases correctly.
    
    # Let's restart the block addition with correct strategies.
    
    U_total = None
    V_total = None
    W_total = None
    total_rank = 0
    
    def add_block_safe(n_b, m_b, p_b, i_start, j_start, k_start, strategy):
        nonlocal U_total, V_total, W_total, total_rank
        
        if strategy == "schoolbook":
            U_b, V_b, W_b, r_b = _make_block(n_b, m_b, p_b)
        elif strategy == "rank1":
            # Only valid if min(m_b, p_b) == 1 or n_b == 1?
            # Actually, Rank(n, m, 1) = 1. Rank(n, 1, p) = 1.
            # Rank(1, m, p) = 1?
            # T_{1jk} = 1. C_{1k} = sum_j A_{1j} B_{jk}.
            # If n=1, C is a row vector.
            # T_{1jk} = 1.
            # Sum_c U_{1j}^{(c)} V_{jk}^{(c)} W_{k1}^{(c)} = 1.
            # U_{1j}=1, V_{jk}=1, W_{k1}=1. Sum=1.
            # So Rank(1, m, p) = 1.
            
            if m_b == 1 or p_b == 1 or n_b == 1:
                U_b = np.ones((n_b * m_b, 1))
                V_b = np.ones((m_b * p_b, 1))
                W_b = np.ones((p_b * n_b, 1))
                r_b = 1
            else:
                # Fallback to schoolbook
                U_b, V_b, W_b, r_b = _make_block(n_b, m_b, p_b)
        elif strategy == "rank_nm_p2":
            if p_b == 2:
                U_b, V_b, W_b, r_b = _make_block(n_b, m_b, p_b, rank_override="rank_nm_p2")
            else:
                U_b, V_b, W_b, r_b = _make_block(n_b, m_b, p_b)
        else:
            U_b, V_b, W_b, r_b = _make_block(n_b, m_b, p_b)
            
        U_e, V_e, W_e = _embed_block(U_b, V_b, W_b, r_b, n, m, p, 
                                     i_start, j_start, k_start, 
                                     i_start + n_b, j_start + m_b, k_start + p_b)
        
        if U_total is None:
            U_total = U_e
            V_total = V_e
            W_total = W_e
        else:
            U_total = np.hstack([U_total, U_e])
            V_total = np.hstack([V_total, V_e])
            W_total = np.hstack([W_total, W_e])
        total_rank += r_b

    # Block 1: J=0-3, K=0-3 (3x4x4)
    add_block_safe(3, 4, 4, 0, 0, 0, "schoolbook")
    
    # Block 2: J=0-3, K=4 (3x4x1) -> Rank 1
    add_block_safe(3, 4, 1, 0, 0, 4, "rank1")
    
    # Block 3: J=0-3, K=5-6 (3x4x2) -> Rank 12
    add_block_safe(3, 4, 2, 0, 0, 5, "rank_nm_p2")
    
    # Block 4: J=4, K=0-4 (3x1x5) -> Rank 1
    add_block_safe(3, 1, 5, 0, 4, 0, "rank1")
    
    # Block 5: J=4, K=5-6 (3x1x2) -> Rank 1
    add_block_safe(3, 1, 2, 0, 4, 5, "rank1")
    
    # Convert to lists
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
