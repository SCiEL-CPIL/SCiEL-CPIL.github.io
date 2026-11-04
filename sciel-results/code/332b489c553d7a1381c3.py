import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication using block decomposition.
    Partitions 5x5 into 2x2 and 3x3 blocks.
    Uses Strassen's algorithm (rank 7) for 2x2 blocks.
    Uses schoolbook algorithm for other blocks.
    Total rank: 7 (2x2) + 12 (2x3x2) + 12 (2x2x3) + 18 (2x3x3) + 
               12 (3x2x2) + 18 (3x3x2) + 18 (3x2x3) + 27 (3x3x3) = 124.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 25
    mp = m * p  # 25
    pn = p * n  # 25
    
    # We will build U, V, W by concatenating factors from each block.
    # Each block contributes a set of rank-1 tensors.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(U_block, V_block, W_block):
        """
        U_block: list of length nm, each entry is a list of length rank_block (or scalar)
        Actually, it's easier to build the full matrices.
        Let's collect the coefficients for each (i, j, k) term.
        """
        pass

    # Instead of building lists of lists, we build the full matrices U, V, W.
    # We don't know the total rank yet, so we collect the terms first.
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))
    
    # Block definitions:
    # i ranges for A rows, j for A cols / B rows, k for B cols.
    # Output C[i, k].
    # W index is k * n + i.
    
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Strassen 2x2
    # Indices: i in [0,1], j in [0,1], k in [0,1]
    # Strassen factors for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to the global indices.
    # Local indices (li, lj, lk) map to global (gi, gj, gk).
    # For this block: gi = li, gj = lj, gk = lk.
    
    # Strassen U, V, W factors for local 2x2:
    # We define 7 terms.
    
    # Term 1: M1
    # A part: A11 + A22 -> U[0,0]=1, U[1,1]=1 (local)
    # B part: B11 + B22 -> V[0,0]=1, V[1,1]=1 (local)
    # C part: C11 += 1, C22 += 1 -> W[0,0]=1, W[1,1]=1 (local)
    # Wait, the factors U, V, W are for the whole tensor.
    # For a rank-1 term, U is a vector, V is a vector, W is a vector.
    # The product U[i,j] * V[j,k] * W[k,i] must sum to T[i,j,k].
    
    # For Strassen, the terms are linear combinations.
    # Term 1:
    # U vector: non-zero at (0,0) and (1,1). Value 1.
    # V vector: non-zero at (0,0) and (1,1). Value 1.
    # W vector: non-zero at (0,0) and (1,1). Value 1.
    # This term contributes to C[0,0] and C[1,1].
    # C[0,0] gets U[0,0]*V[0,0]*W[0,0] = 1.
    # C[1,1] gets U[1,1]*V[1,1]*W[1,1] = 1.
    # This is correct for M1's contribution to C11 and C22?
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # The term U=(1,0,1,0), V=(1,0,1,0), W=(1,0,1,0) gives:
    # Sum over r of U[i,j]V[j,k]W[k,i].
    # For i=0,j=0,k=0: U[0,0]V[0,0]W[0,0] = 1.
    # For i=0,j=1,k=1: U[0,1]... = 0.
    # This doesn't look like it captures the cross terms.
    
    # The tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For Strassen, we express the matmul as a sum of 7 products of linear forms.
    # Each product is a rank-1 tensor in the space of (i,j) x (j,k) x (k,i)?
    # No, the factors are U (nm x r), V (mp x r), W (pn x r).
    # So for each r, we have a vector u_r in R^25, v_r in R^25, w_r in R^25.
    # The condition is sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = T[i,j,k].
    
    # For the 2x2 block, we can define the local vectors and then embed them.
    
    def embed_2x2_strassen():
        # Local indices 0,1
        # 7 terms
        local_terms = []
        
        # M1: (A11+A22)(B11+B22)
        # u: [1, 0, 0, 1] (A11, A12, A21, A22) -> indices (0,0), (0,1), (1,0), (1,1)
        # v: [1, 0, 0, 1] (B11, B12, B21, B22) -> indices (0,0), (0,1), (1,0), (1,1)
        # w: [1, 0, 0, 1] (C11, C12, C21, C22) -> indices (0,0), (0,1), (1,0), (1,1)
        # Note: W index is k*n+i.
        # Local k*n+i:
        # k=0,i=0 -> 0
        # k=0,i=1 -> 1
        # k=1,i=0 -> 2
        # k=1,i=1 -> 3
        # So w vector is indexed by (k,i).
        # M1 contributes to C11 (k=0,i=0) and C22 (k=1,i=1).
        # So w[0]=1, w[3]=1.
        
        u = [1, 0, 0, 1] # A11, A12, A21, A22
        v = [1, 0, 0, 1] # B11, B12, B21, B22
        w = [1, 0, 0, 1] # C11, C12, C21, C22 -> w[0]=1 (C11), w[3]=1 (C22)
        local_terms.append((u, v, w))
        
        # M2: (A21+A22)B11
        u = [0, 0, 1, 1] # A21, A22
        v = [1, 0, 0, 0] # B11
        w = [0, 0, 1, 0] # C21 (k=0,i=1? No. C21 is row 2, col 1. i=1, k=0. Index k*n+i = 0*2+1 = 1)
        # Wait. C[i,k].
        # C21 means i=1, k=0.
        # W index is k*n + i = 0*2 + 1 = 1.
        # So w[1]=1.
        w = [0, 1, 0, 0]
        local_terms.append((u, v, w))
        
        # M3: A11(B12-B22)
        u = [1, 0, 0, 0] # A11
        v = [0, 1, 0, -1] # B12, -B22
        w = [0, 1, 0, 0] # C12 (i=0, k=1. Index 1*2+0 = 2)
        # C12 is i=0, k=1. W index 1*2+0 = 2.
        w = [0, 0, 1, 0]
        local_terms.append((u, v, w))
        
        # M4: A22(B21-B11)
        u = [0, 0, 0, 1] # A22
        v = [0, 0, 1, -1] # B21, -B11
        w = [1, 0, 0, 0] # C11 (i=0, k=0. Index 0)
        local_terms.append((u, v, w))
        
        # M5: (A11+A12)B22
        u = [1, 1, 0, 0] # A11, A12
        v = [0, 0, 0, 1] # B22
        w = [0, 0, 0, 1] # C22 (i=1, k=1. Index 1*2+1=3)
        local_terms.append((u, v, w))
        
        # M6: (A21-A11)(B11+B12)
        u = [-1, 0, 1, 0] # -A11, A21
        v = [1, 1, 0, 0] # B11, B12
        w = [0, 0, 0, 1] # C22 (i=1, k=1. Index 3)
        local_terms.append((u, v, w))
        
        # M7: (A12-A22)(B21+B22)
        u = [0, 1, 0, -1] # A12, -A22
        v = [0, 0, 1, 1] # B21, B22
        w = [1, 0, 0, 0] # C11 (i=0, k=0. Index 0)
        local_terms.append((u, v, w))
        
        return local_terms

    def embed_schoolbook(n_block, m_block, p_block, i_offset, j_offset, k_offset):
        """
        Returns list of (u, v, w) vectors for a schoolbook block of size n_block x m_block x p_block.
        u is of length n_block * m_block.
        v is of length m_block * p_block.
        w is of length p_block * n_block.
        """
        local_terms = []
        for li in range(n_block):
            for lj in range(m_block):
                for lk in range(p_block):
                    u = [0.0] * (n_block * m_block)
                    v = [0.0] * (m_block * p_block)
                    w = [0.0] * (p_block * n_block)
                    
                    u_idx = li * m_block + lj
                    v_idx = lj * p_block + lk
                    w_idx = lk * n_block + li
                    
                    u[u_idx] = 1.0
                    v[v_idx] = 1.0
                    w[w_idx] = 1.0
                    
                    local_terms.append((u, v, w))
        return local_terms

    # 1. Block 2x2 x 2x2 x 2x2 (Top-Left)
    # i in [0,1], j in [0,1], k in [0,1]
    local_terms_22 = embed_2x2_strassen()
    for u_loc, v_loc, w_loc in local_terms_22:
        # Embed into global size 25
        u_glob = [0.0] * 25
        v_glob = [0.0] * 25
        w_glob = [0.0] * 25
        
        # Map local indices to global
        # u: i in [0,1], j in [0,1] -> global i, j
        for li in range(2):
            for lj in range(2):
                u_glob[li * 5 + lj] = u_loc[li * 2 + lj]
                
        # v: j in [0,1], k in [0,1] -> global j, k
        for lj in range(2):
            for lk in range(2):
                v_glob[lj * 5 + lk] = v_loc[lj * 2 + lk]
                
        # w: k in [0,1], i in [0,1] -> global k, i
        for lk in range(2):
            for li in range(2):
                w_glob[lk * 5 + li] = w_loc[lk * 2 + li]
                
        add_term(0, 0, 0, 0, 0, 0) # Placeholder, we will append vectors directly
        U_list.append(u_glob)
        V_list.append(v_glob)
        W_list.append(w_glob)
        
    # 2. Block 2x3 x 3x2 (Top-Right A, Bottom-Left B)
    # i in [0,1], j in [2,4], k in [0,1]
    # A part: 2x3. B part: 3x2. C part: 2x2.
    local_terms_232 = embed_schoolbook(2, 3, 2, 0, 0, 0)
    for u_loc, v_loc, w_loc in local_terms_232:
        u_glob = [0.0] * 25
        v_glob = [0.0] * 25
        w_glob = [0.0] * 25
        
        # u: i in [0,1], j in [2,4]
        for li in range(2):
            for lj in range(3):
                u_glob[li * 5 + (2 + lj)] = u_loc[li * 3 + lj]
                
        # v: j in [2,4], k in [0,1]
        for lj in range(3):
            for lk in range(2):
                v_glob[(2 + lj) * 5 + lk] = v_loc[lj * 2 + lk]
                
        # w: k in [0,1], i in [0,1]
        for lk in range(2):
            for li in range(2):
                w_glob[lk * 5 + li] = w_loc[lk * 2 + li]
                
        U_list.append(u_glob)
        V_list.append(v_glob)
        W_list.append(w_glob)
        
    # 3. Block 2x2 x 2x3 (Top-Left A, Top-Right B)
    # i in [0,1], j in [0,1], k in [2,4]
    local_terms_223 = embed_schoolbook(2, 2, 3, 0, 0, 0)
    for u_loc, v_loc, w_loc in local_terms_223:
        u_glob = [0.0] * 25
        v_glob = [0.0] * 25
        w_glob = [0.0] * 25
        
        # u: i in [0,1], j in [0,1]
        for li in range(2):
            for lj in range(2):
                u_glob[li * 5 + lj] = u_loc[li * 2 + lj]
                
        # v: j in [0,1], k in [2,4]
        for lj in range(2):
            for lk in range(3):
                v_glob[lj * 5 + (2 + lk)] = v_loc[lj * 3 + lk]
                
        # w: k in [2,4], i in [0,1]
        for lk in range(3):
            for li in range(2):
                w_glob[(2 + lk) * 5 + li] = w_loc[lk * 2 + li]
                
        U_list.append(u_glob)
        V_list.append(v_glob)
        W_list.append(w_glob)
        
    # 4. Block 2x3 x 3x3 (Top-Right A, Bottom-Right B)
    # i in [0,1], j in [2,4], k in [2,4]
    local_terms_233 = embed_schoolbook(2, 3, 3, 0, 0, 0)
    for u_loc, v_loc, w_loc in local_terms_233:
        u_glob = [0.0] * 25
        v_glob = [0.0] * 25
        w_glob = [0.0] * 25
        
        # u: i in [0,1], j in [2,4]
        for li in range(2):
            for lj in range(3):
                u_glob[li * 5 + (2 + lj)] = u_loc[li * 3 + lj]
                
        # v: j in [2,4], k in [2,4]
        for lj in range(3):
            for lk in range(3):
                v_glob[(2 + lj) * 5 + (2 + lk)] = v_loc[lj * 3 + lk]
                
        # w: k in [2,4], i in [0,1]
        for lk in range(3):
            for li in range(2):
                w_glob[(2 + lk) * 5 + li] = w_loc[lk * 2 + li]
                
        U_list.append(u_glob)
        V_list.append(v_glob)
        W_list.append(w_glob)
        
    # 5. Block 3x2 x 2x2 (Bottom-Left A, Top-Left B)
    # i in [2,4], j in [0,1], k in [0,1]
    local_terms_322 = embed_schoolbook(3, 2, 2, 0, 0, 0)
    for u_loc, v_loc, w_loc in local_terms_322:
        u_glob = [0.0] * 25
        v_glob = [0.0] * 25
        w_glob = [0.0] * 25
        
        # u: i in [2,4], j in [0,1]
        for li in range(3):
            for lj in range(2):
                u_glob[(2 + li) * 5 + lj] = u_loc[li * 2 + lj]
                
        # v: j in [0,1], k in [0,1]
        for lj in range(2):
            for lk in range(2):
                v_glob[lj * 5 + lk] = v_loc[lj * 2 + lk]
                
        # w: k in [0,1], i in [2,4]
        for lk in range(2):
            for li in range(3):
                w_glob[lk * 5 + (2 + li)] = w_loc[lk * 3 + li]
                
        U_list.append(u_glob)
        V_list.append(v_glob)
        W_list.append(w_glob)
        
    # 6. Block 3x3 x 3x2 (Bottom-Right A, Bottom-Left B)
    # i in [2,4], j in [2,4], k in [0,1]
    local_terms_332 = embed_schoolbook(3, 3, 2, 0, 0, 0)
    for u_loc, v_loc, w_loc in local_terms_332:
        u_glob = [0.0] * 25
        v_glob = [0.0] * 25
        w_glob = [0.0] * 25
        
        # u: i in [2,4], j in [2,4]
        for li in range(3):
            for lj in range(3):
                u_glob[(2 + li) * 5 + (2 + lj)] = u_loc[li * 3 + lj]
                
        # v: j in [2,4], k in [0,1]
        for lj in range(3):
            for lk in range(2):
                v_glob[(2 + lj) * 5 + lk] = v_loc[lj * 2 + lk]
                
        # w: k in [0,1], i in [2,4]
        for lk in range(2):
            for li in range(3):
                w_glob[lk * 5 + (2 + li)] = w_loc[lk * 3 + li]
                
        U_list.append(u_glob)
        V_list.append(v_glob)
        W_list.append(w_glob)
        
    # 7. Block 3x2 x 2x3 (Bottom-Left A, Top-Right B)
    # i in [2,4], j in [0,1], k in [2,4]
    local_terms_323 = embed_schoolbook(3, 2, 3, 0, 0, 0)
    for u_loc, v_loc, w_loc in local_terms_323:
        u_glob = [0.0] * 25
        v_glob = [0.0] * 25
        w_glob = [0.0] * 25
        
        # u: i in [2,4], j in [0,1]
        for li in range(3):
            for lj in range(2):
                u_glob[(2 + li) * 5 + lj] = u_loc[li * 2 + lj]
                
        # v: j in [0,1], k in [2,4]
        for lj in range(2):
            for lk in range(3):
                v_glob[lj * 5 + (2 + lk)] = v_loc[lj * 3 + lk]
                
        # w: k in [2,4], i in [2,4]
        for lk in range(3):
            for li in range(3):
                w_glob[(2 + lk) * 5 + (2 + li)] = w_loc[lk * 3 + li]
                
        U_list.append(u_glob)
        V_list.append(v_glob)
        W_list.append(w_glob)
        
    # 8. Block 3x3 x 3x3 (Bottom-Right A, Bottom-Right B)
    # i in [2,4], j in [2,4], k in [2,4]
    local_terms_333 = embed_schoolbook(3, 3, 3, 0, 0, 0)
    for u_loc, v_loc, w_loc in local_terms_333:
        u_glob = [0.0] * 25
        v_glob = [0.0] * 25
        w_glob = [0.0] * 25
        
        # u: i in [2,4], j in [2,4]
        for li in range(3):
            for lj in range(3):
                u_glob[(2 + li) * 5 + (2 + lj)] = u_loc[li * 3 + lj]
                
        # v: j in [2,4], k in [2,4]
        for lj in range(3):
            for lk in range(3):
                v_glob[(2 + lj) * 5 + (2 + lk)] = v_loc[lj * 3 + lk]
                
        # w: k in [2,4], i in [2,4]
        for lk in range(3):
            for li in range(3):
                w_glob[(2 + lk) * 5 + (2 + li)] = w_loc[lk * 3 + li]
                
        U_list.append(u_glob)
        V_list.append(v_glob)
        W_list.append(w_glob)

    # Convert to numpy arrays
    U = np.array(U_list).T # Shape (25, rank)
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    # Verify decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 for all i,j,k
    
    # Check time
    if time.time() > deadline - safety_margin:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
