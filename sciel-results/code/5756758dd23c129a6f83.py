"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin within 150s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_true = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1
    
    # Strategy: Use block partitioning with Strassen for 2x2x2 blocks
    # and schoolbook for the rest.
    # We partition the indices into blocks and apply appropriate algorithms.
    
    # Partition of n=3 rows of A (index i): [2, 1]
    # Partition of m=5 cols of A (index j): [2, 2, 1]
    # Partition of p=5 cols of B (index k): [2, 2, 1]
    #
    # This gives us 2*2*2 = 8 blocks of size (2,2,2) for Strassen,
    # plus smaller blocks for the remainder.
    # Strassen blocks: 8 * 7 = 56 rank terms
    # Schoolbook blocks: the rest
    #
    # Total schoolbook terms:
    # (2,2,1): 2*2*1 = 4
    # (2,1,2): 2*1*2 = 4
    # (1,2,2): 1*2*2 = 4
    # (2,2,1): 4
    # (2,1,2): 4
    # (1,2,2): 4
    # (2,2,1): 4
    # (2,1,2): 4
    # (1,2,2): 4
    # (1,1,1): 1
    # (1,1,1): 1
    # (1,1,1): 1
    # (1,1,1): 1
    # (1,1,1): 1
    # (1,1,1): 1
    # (1,1,1): 1
    # (1,1,1): 1
    # (1,1,1): 1
    #
    # Actually let me just compute: total = 3*5*5 = 75
    # Strassen saves 1 per 2x2x2 block. With partition [2,1] x [2,2,1] x [2,2,1]:
    # 2x2x2 blocks: 2*2*2 = 8 blocks, saving 8 ranks
    # Remaining: 75 - 8*8 = 75 - 64 = 11 schoolbook terms
    # Total rank: 8*7 + 11 = 56 + 11 = 67
    #
    # Alternative: [2,1] x [2,1,2] x [2,1,2]:
    # 2x2x2 blocks: 2*2*2 = 8 blocks, saving 8 ranks
    # Same count.
    #
    # Let's use [2,1] x [2,2,1] x [2,2,1]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    i_sizes = [2, 1]
    j_sizes = [2, 2, 1]
    k_sizes = [2, 2, 1]
    
    i_offsets = [0, 2]
    j_offsets = [0, 2, 4]
    k_offsets = [0, 2, 4]
    
    for I, (i_sz, i_off) in enumerate(zip(i_sizes, i_offsets)):
        for J, (j_sz, j_off) in enumerate(zip(j_sizes, j_offsets)):
            for K, (k_sz, k_off) in enumerate(zip(k_sizes, k_offsets)):
                # Determine if we can use Strassen for this block
                # Strassen applies when i_sz == j_sz == k_sz == 2
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    # Use Strassen decomposition for 2x2x2 block
                    # Strassen's 7 products for C = A * B where A is 2x2, B is 2x2
                    # M1 = (A00 + A11)(B00 + B11)
                    # M2 = (A10 + A11)B00
                    # M3 = A00(B10 - B11)
                    # M4 = A11(B01 - B00)
                    # M5 = (A00 + A01)B11
                    # M6 = (A10 - A00)(B00 + B10)
                    # M7 = (A01 - A11)(B01 + B11)
                    
                    # For each of the 7 terms, create a column
                    # Term 1: M1 = (A00 + A11)(B00 + B11)
                    # U indices: i_off+0, j_off+0 and i_off+1, j_off+1
                    # V indices: j_off+0, k_off+0 and j_off+1, k_off+1
                    # W indices: k_off+0, i_off+0 and k_off+1, i_off+1
                    
                    # U: [ (i_off, j_off), (i_off+1, j_off+1) ] with values [1, 1]
                    # V: [ (j_off, k_off), (j_off+1, k_off+1) ] with values [1, 1]
                    # W: [ (k_off, i_off), (k_off+1, i_off+1) ] with values [1, 1]
                    # But Strassen reconstructs C via linear combinations.
                    # Actually, each "product" M_t contributes to specific C entries.
                    # The tensor decomposition approach: we need to express
                    # T_block as sum of 7 rank-1 tensors.
                    
                    # For Strassen 2x2x2, the 7 rank-1 decompositions are:
                    # T1: U=(1,0,1,0), V=(1,0,1,0), W=(1,0,1,0) for the "diagonal sum" pattern
                    # But it's more complex. Let me use the standard Strassen formula.
                    
                    # C00 = M1 + M4 - M5 - M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # Each Mt is a product of two linear forms.
                    # We decompose each Mt into rank-1 tensors.
                    
                    # M1 = (A00 + A11)(B00 + B11)
                    # U1: indices (i_off, j_off)=1, (i_off+1, j_off+1)=1
                    # V1: indices (j_off, k_off)=1, (j_off+1, k_off+1)=1
                    # W1: contributes to C00(+1), C11(+1)
                    # So W1: (k_off, i_off)=1, (k_off+1, i_off+1)=1
                    
                    U1 = np.zeros(nm)
                    U1[i_off*m + j_off] = 1
                    U1[(i_off+1)*m + (j_off+1)] = 1
                    V1 = np.zeros(mp)
                    V1[j_off*p + k_off] = 1
                    V1[(j_off+1)*p + (k_off+1)] = 1
                    W1 = np.zeros(pn)
                    W1[k_off*n + i_off] = 1
                    W1[(k_off+1)*n + (i_off+1)] = 1
                    U_cols.append(U1)
                    V_cols.append(V1)
                    W_cols.append(W1)
                    
                    # M2 = (A10 + A11)B00
                    # U2: (i_off+1, j_off)=1, (i_off+1, j_off+1)=1
                    # V2: (j_off, k_off)=1
                    # W2: contributes to C10(+1), C11(-1)
                    # W2: (k_off, i_off+1)=1, (k_off+1, i_off+1)=-1
                    
                    U2 = np.zeros(nm)
                    U2[(i_off+1)*m + j_off] = 1
                    U2[(i_off+1)*m + (j_off+1)] = 1
                    V2 = np.zeros(mp)
                    V2[j_off*p + k_off] = 1
                    W2 = np.zeros(pn)
                    W2[k_off*n + (i_off+1)] = 1
                    W2[(k_off+1)*n + (i_off+1)] = -1
                    U_cols.append(U2)
                    V_cols.append(V2)
                    W_cols.append(W2)
                    
                    # M3 = A00(B10 - B11)
                    # U3: (i_off, j_off)=1
                    # V3: (j_off+1, k_off)=1, (j_off+1, k_off+1)=-1
                    # W3: contributes to C01(+1), C11(+1)
                    # W3: (k_off+1, i_off)=1, (k_off+1, i_off+1)=1
                    
                    U3 = np.zeros(nm)
                    U3[i_off*m + j_off] = 1
                    V3 = np.zeros(mp)
                    V3[(j_off+1)*p + k_off] = 1
                    V3[(j_off+1)*p + (k_off+1)] = -1
                    W3 = np.zeros(pn)
                    W3[(k_off+1)*n + i_off] = 1
                    W3[(k_off+1)*n + (i_off+1)] = 1
                    U_cols.append(U3)
                    V_cols.append(V3)
                    W_cols.append(W3)
                    
                    # M4 = A11(B01 - B00)
                    # U4: (i_off+1, j_off+1)=1
                    # V4: (j_off, k_off+1)=1, (j_off, k_off)=-1
                    # W4: contributes to C00(+1), C10(+1)
                    # W4: (k_off, i_off)=1, (k_off, i_off+1)=1
                    
                    U4 = np.zeros(nm)
                    U4[(i_off+1)*m + (j_off+1)] = 1
                    V4 = np.zeros(mp)
                    V4[j_off*p + (k_off+1)] = 1
                    V4[j_off*p + k_off] = -1
                    W4 = np.zeros(pn)
                    W4[k_off*n + i_off] = 1
                    W4[k_off*n + (i_off+1)] = 1
                    U_cols.append(U4)
                    V_cols.append(V4)
                    W_cols.append(W4)
                    
                    # M5 = (A00 + A01)B11
                    # U5: (i_off, j_off)=1, (i_off, j_off+1)=1
                    # V5: (j_off+1, k_off+1)=1
                    # W5: contributes to C00(-1), C01(+1)
                    # W5: (k_off, i_off)=-1, (k_off+1, i_off)=1
                    
                    U5 = np.zeros(nm)
                    U5[i_off*m + j_off] = 1
                    U5[i_off*m + (j_off+1)] = 1
                    V5 = np.zeros(mp)
                    V5[(j_off+1)*p + (k_off+1)] = 1
                    W5 = np.zeros(pn)
                    W5[k_off*n + i_off] = -1
                    W5[(k_off+1)*n + i_off] = 1
                    U_cols.append(U5)
                    V_cols.append(V5)
                    W_cols.append(W5)
                    
                    # M6 = (A10 - A00)(B00 + B10)
                    # U6: (i_off+1, j_off)=1, (i_off, j_off)=-1
                    # V6: (j_off, k_off)=1, (j_off+1, k_off)=1
                    # W6: contributes to C11(+1)
                    # W6: (k_off+1, i_off+1)=1
                    
                    U6 = np.zeros(nm)
                    U6[(i_off+1)*m + j_off] = 1
                    U6[i_off*m + j_off] = -1
                    V6 = np.zeros(mp)
                    V6[j_off*p + k_off] = 1
                    V6[(j_off+1)*p + k_off] = 1
                    W6 = np.zeros(pn)
                    W6[(k_off+1)*n + (i_off+1)] = 1
                    U_cols.append(U6)
                    V_cols.append(V6)
                    W_cols.append(W6)
                    
                    # M7 = (A01 - A11)(B01 + B11)
                    # U7: (i_off, j_off+1)=1, (i_off+1, j_off+1)=-1
                    # V7: (j_off, k_off+1)=1, (j_off+1, k_off+1)=1
                    # W7: contributes to C00(-1)
                    # W7: (k_off, i_off)=-1
                    
                    U7 = np.zeros(nm)
                    U7[i_off*m + (j_off+1)] = 1
                    U7[(i_off+1)*m + (j_off+1)] = -1
                    V7 = np.zeros(mp)
                    V7[j_off*p + (k_off+1)] = 1
                    V7[(j_off+1)*p + (k_off+1)] = 1
                    W7 = np.zeros(pn)
                    W7[k_off*n + i_off] = -1
                    U_cols.append(U7)
                    V_cols.append(V7)
                    W_cols.append(W7)
                    
                else:
                    # Schoolbook for this block: rank = i_sz * j_sz * k_sz
                    for i_loc in range(i_sz):
                        for j_loc in range(j_sz):
                            for k_loc in range(k_sz):
                                U_col = np.zeros(nm)
                                V_col = np.zeros(mp)
                                W_col = np.zeros(pn)
                                
                                i_glob = i_off + i_loc
                                j_glob = j_off + j_loc
                                k_glob = k_off + k_loc
                                
                                U_col[i_glob * m + j_glob] = 1
                                V_col[j_glob * p + k_glob] = 1
                                W_col[k_glob * n + i_glob] = 1
                                
                                U_cols.append(U_col)
                                V_cols.append(V_col)
                                W_cols.append(W_col)
    
    # Convert to matrices
    U = np.column_stack(U_cols) if U_cols else np.zeros((nm, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((mp, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((pn, 0))
    
    current_rank = U.shape[1]
    
    # Verify the decomposition
    def verify_decomp(U, V, W, n, m, p, T_true):
        """Verify that the decomposition exactly reconstructs the matrix mult tensor."""
        rank = U.shape[1]
        if rank == 0:
            return False
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_recon, T_true, atol=1e-10)
    
    if not verify_decomp(U, V, W, n, m, p, T_true):
        # Fallback to schoolbook if Strassen decomposition is incorrect
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        current_rank = rank
    
    # Try to reduce rank further by removing columns and checking if residual is rank-1
    def check_rank_one_residual(T_res):
        """Check if T_res is a rank-1 tensor. Returns (u, v, w) if yes, else None."""
        nz = np.nonzero(T_res)
        if len(nz[0]) == 0:
            return None
        
        a0, b0, c0 = nz[0][0], nz[1][0], nz[2][0]
        val = T_res[a0, b0, c0]
        if val == 0:
            return None
            
        u = T_res[:, b0, c0] / val
        v = T_res[a0, :, c0] / val
        w = T_res[a0, b0, :] / val
        
        recon = np.einsum('a,b,c->abc', u, v, w)
        if np.allclose(recon, T_res, atol=1e-10):
            return u, v, w
        return None
    
    # Greedy column removal - try multiple random orderings
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        reduced = False
        cols = list(range(current_rank))
        np.random.shuffle(cols)
        
        for c in cols:
            if time.time() > deadline - 5:
                break
            
            mask = [i for i in range(current_rank) if i != c]
            
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            
            T_sub = np.einsum('at,bt,ct->abc', U_sub, V_sub, W_sub)
            T_res = T_true - T_sub
            
            res_vecs = check_rank_one_residual(T_res)
            if res_vecs is not None:
                u_new, v_new, w_new = res_vecs
                U = np.column_stack([U_sub, u_new])
                V = np.column_stack([V_sub, v_new])
                W = np.column_stack([W_sub, w_new])
                current_rank -= 1
                reduced = True
                break
        
        if not reduced:
            break
    
    # Second pass: try removing pairs of columns and replacing with 1 rank-1 tensor
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        reduced = False
        cols = list(range(current_rank))
        np.random.shuffle(cols)
        
        for c1 in cols:
            if time.time() > deadline - 5:
                break
            for c2 in cols:
                if c2 == c1 or time.time() > deadline - 5:
                    continue
                
                mask = [i for i in range(current_rank) if i != c1 and i != c2]
                
                U_sub = U[:, mask]
                V_sub = V[:, mask]
                W_sub = W[:, mask]
                
                T_sub = np.einsum('at,bt,ct->abc', U_sub, V_sub, W_sub)
                T_res = T_true - T_sub
                
                # Check if T_res is rank-1
                res_vecs = check_rank_one_residual(T_res)
                if res_vecs is not None:
                    u_new, v_new, w_new = res_vecs
                    U = np.column_stack([U_sub, u_new])
                    V = np.column_stack([V_sub, v_new])
                    W = np.column_stack([W_sub, w_new])
                    current_rank -= 1
                    reduced = True
                    break
            if reduced:
                break
        
        if not reduced:
            break
    
    # Final verification
    if verify_decomp(U, V, W, n, m, p, T_true):
        # Ensure entries are clean integers or half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Re-verify after rounding
        if verify_decomp(U, V, W, n, m, p, T_true):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
