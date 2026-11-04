import numpy as np
import time

def construct():
    """
    Strategy: Start with a known valid rank-44 decomposition (Strassen-based) for 3x4x4.
    Then, attempt to reduce the rank by identifying and removing redundant rank-1 terms.
    
    The tensor T_{i,j,k} = delta_{i,k} delta_{j,k} (using trace convention W[k*n+i,r] implies C[i,k] = sum_j A[i,j]B[j,k]).
    Actually, the standard matrix multiplication tensor M_{n,m,p} has entries M_{i,j,k} = delta_{i,k} delta_{j,j}? 
    Let's stick to the verification logic in the seed:
    C[i,k] = sum_j A[i,j] B[j,k].
    Tensor component T_{i,j,k} = 1 if i==l and j==k? No.
    The seed verification checks:
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+l, r] == 1 if (i==l and j==k) else 0.
    Wait, the seed says: `val = 1.0 if (i==l and j==k) else 0.0`.
    Here `l` is the row index of C (0..n-1), `k` is the col index of C (0..p-1).
    `i` is row of A, `j` is col of A / row of B, `k` is col of B / col of C.
    So the target tensor is T_{i,j,k,l} = delta_{i,l} delta_{j,k}.
    This is the standard matrix multiplication tensor.
    
    We construct the rank-44 decomposition first.
    Then we try to reduce rank.
    """
    t_start = time.time()
    DEADLINE = 130.0  # Safety margin

    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    # 1. Build the Rank-44 Decomposition
    # This is the same as the seed program, but cleaned up.
    
    # Strassen 2x2*2x2 = 2x2, rank 7
    sU = np.zeros((4, 7), dtype=float)
    sV = np.zeros((4, 7), dtype=float)
    sW = np.zeros((4, 7), dtype=float)
    
    # P1 = (A00+A11)(B00+B11)
    sU[0,0]=1; sU[3,0]=1; sV[0,0]=1; sV[3,0]=1; sW[0,0]=1; sW[3,0]=1
    # P2 = (A10+A11)B00
    sU[2,1]=1; sU[3,1]=1; sV[0,1]=1; sW[2,1]=1; sW[3,1]=-1
    # P3 = A00(B10-B11)
    sU[0,2]=1; sV[2,2]=1; sV[3,2]=-1; sW[1,2]=1; sW[3,2]=1
    # P4 = A11(B00-B01)
    sU[3,3]=1; sV[0,3]=1; sV[1,3]=-1; sW[0,3]=1; sW[2,3]=1
    # P5 = (A00+A01)B11
    sU[0,4]=1; sU[1,4]=1; sV[3,4]=1; sW[0,4]=-1; sW[1,4]=1
    # P6 = (A10-A00)(B00+B01)
    sU[2,5]=1; sU[0,5]=-1; sV[0,5]=1; sV[1,5]=1; sW[3,5]=1
    # P7 = (A01-A11)(B10+B11)
    sU[1,6]=1; sU[3,6]=-1; sV[2,6]=1; sV[3,6]=1; sW[0,6]=1
    
    # Schoolbook 1x2 * 2x2 = 1x2, rank 4
    bU = np.zeros((2, 4), dtype=float)
    bV = np.zeros((4, 4), dtype=float)
    bW = np.zeros((2, 4), dtype=float)
    
    # C0 = A0*B00 + A1*B10
    bU[0,0]=1; bV[0,0]=1; bW[0,0]=1
    bU[1,1]=1; bV[2,1]=1; bW[0,1]=1
    # C1 = A0*B01 + A1*B11
    bU[0,2]=1; bV[1,2]=1; bW[1,2]=1
    bU[1,3]=1; bV[3,3]=1; bW[1,3]=1
    
    U_full = np.zeros((dim_u, 44), dtype=float)
    V_full = np.zeros((dim_v, 44), dtype=float)
    W_full = np.zeros((dim_w, 44), dtype=float)
    
    col_offset = 0
    
    def add_block(block_U, block_V, block_W, u_indices, v_indices, w_indices):
        nonlocal col_offset
        rank = block_U.shape[1]
        for r in range(rank):
            for u_idx, b_u_idx in zip(u_indices, range(block_U.shape[0])):
                U_full[u_idx, col_offset + r] = block_U[b_u_idx, r]
            for v_idx, b_v_idx in zip(v_indices, range(block_V.shape[0])):
                V_full[v_idx, col_offset + r] = block_V[b_v_idx, r]
            for w_idx, b_w_idx in zip(w_indices, range(block_W.shape[0])):
                W_full[w_idx, col_offset + r] = block_W[b_w_idx, r]
        col_offset += rank

    # Index mappings
    # A (3x4): U indices = i*4+j
    # A1a (2x2): rows 0,1 cols 0,1 -> [0,1,4,5]
    # A1b (2x2): rows 0,1 cols 2,3 -> [2,3,6,7]
    # A2a (1x2): row 2 cols 0,1 -> [8,9]
    # A2b (1x2): row 2 cols 2,3 -> [10,11]
    
    # B (4x4): V indices = j*4+k
    # B1a (2x2): rows 0,1 cols 0,1 -> [0,1,4,5]
    # B1b (2x2): rows 2,3 cols 0,1 -> [8,9,12,13]
    # B2a (2x2): rows 0,1 cols 2,3 -> [2,3,6,7]
    # B2b (2x2): rows 2,3 cols 2,3 -> [10,11,14,15]
    
    # C (3x4): W indices = k*3+l
    # C1 (2x2): rows 0,1 cols 0,1 -> [0,1,3,4]
    # C2 (2x2): rows 0,1 cols 2,3 -> [6,7,9,10]
    # C3 (1x2): row 2 cols 0,1 -> [2,5]
    # C4 (1x2): row 2 cols 2,3 -> [8,11]
    
    # C1 = A1a*B1a + A1b*B1b
    add_block(sU, sV, sW, [0,1,4,5], [0,1,4,5], [0,1,3,4])
    add_block(sU, sV, sW, [2,3,6,7], [8,9,12,13], [0,1,3,4])
    
    # C2 = A1a*B2a + A1b*B2b
    add_block(sU, sV, sW, [0,1,4,5], [2,3,6,7], [6,7,9,10])
    add_block(sU, sV, sW, [2,3,6,7], [10,11,14,15], [6,7,9,10])
    
    # C3 = A2a*B1a + A2b*B1b
    add_block(bU, bV, bW, [8,9], [0,1,4,5], [2,5])
    add_block(bU, bV, bW, [10,11], [8,9,12,13], [2,5])
    
    # C4 = A2a*B2a + A2b*B2b
    add_block(bU, bV, bW, [8,9], [2,3,6,7], [8,11])
    add_block(bU, bV, bW, [10,11], [10,11,14,15], [8,11])
    
    # 2. Verification Function
    def _verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(n):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        val = 1.0 if (i==l and j==k) else 0.0
                        err = abs(s - val)
                        if err > max_err:
                            max_err = err
        return max_err < tol, max_err

    # Initial check
    valid, err = _verify(U_full, V_full, W_full)
    if not valid:
        # Fallback to schoolbook if construction failed (should not happen)
        rank = n*m*p
        U = np.zeros((dim_u, rank), dtype=float)
        V = np.zeros((dim_v, rank), dtype=float)
        W = np.zeros((dim_w, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U_full, V_full, W_full
    best_rank = 44
    
    # 3. Rank Reduction Search
    # We try to remove rank terms one by one.
    # If removing a term breaks the decomposition, we check if we can adjust other terms to compensate.
    # This is complex. A simpler heuristic: 
    # Check for linear dependencies in the set of rank-1 tensors.
    # If one rank-1 tensor is a linear combination of others, we can remove it.
    
    # Represent the decomposition as a list of tuples (u_vec, v_vec, w_vec)
    # We can flatten U, V, W columns.
    
    def try_reduce_rank(U, V, W, max_iter=100):
        rank = U.shape[1]
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        current_rank = rank
        
        # We will try to remove columns one by one.
        # To check if a column is redundant, we can try to solve for it in terms of others.
        # However, tensor rank reduction is NP-hard in general.
        # For this small size, we can try a greedy approach:
        # For each column r, set it to 0 and see if we can find small adjustments to other columns to restore the tensor.
        
        # A more robust approach for small ranks:
        # Construct the "tensor space" basis.
        # The tensor T is fixed.
        # We have R rank-1 terms.
        # If we remove term r, we need to see if T - sum_{r!=r} t_r can be represented with rank < R-1? No, we want to keep rank R-1.
        # So we check if T - sum_{r!=r} t_r is a rank-1 tensor? No, it must be representable by the remaining R-1 terms with modified coefficients/vectors.
        
        # Actually, a simpler check: 
        # If the vectors (u_r, v_r, w_r) are linearly dependent in the "tensor product space" in a specific way, we can merge.
        
        # Let's try a simpler heuristic: Randomized search for lower rank.
        # But we have a deadline.
        
        # Alternative: Use the fact that many entries are 0, 1, -1.
        # Check for duplicate or negated rank-1 terms.
        
        # Let's just try to remove each column and verify. If it fails, keep it.
        # This only works if the decomposition is over-complete.
        
        for r in range(current_rank):
            if time.time() > DEADLINE:
                break
            # Try removing column r
            test_U = np.delete(current_U, r, axis=1)
            test_V = np.delete(current_V, r, axis=1)
            test_W = np.delete(current_W, r, axis=1)
            
            # Verify
            valid, err = _verify(test_U, test_V, test_W)
            if valid:
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                # Restart the loop as indices changed
                r = 0
        
        return current_U, current_V, current_W, current_rank

    # Run the reduction
    U_red, V_red, W_red, rank_red = try_reduce_rank(U_full, V_full, W_full)
    
    if rank_red < best_rank:
        best_U, best_V, best_W = U_red, V_red, W_red
        best_rank = rank_red

    # 4. Try a different structural decomposition: 
    # Can we do better than 44?
    # Lower bound is 16.
    # Known bounds for 3x4x4:
    # R(3x4x4) <= R(3x3x3) + R(3x4x1) + R(1x4x4)? No.
    # R(3x4x4) <= R(2x4x4) + R(1x4x4).
    # R(2x4x4) <= R(2x2x4) + R(2x2x4) = 2 * R(2x2x4).
    # R(2x2x4) <= R(2x2x2) + R(2x2x2) = 14? No, R(2x2x4) is rank of 2x2*4x4? No, 2x2 * 2x4 = 2x4?
    # Let's stick to the current best. The reduction search might find a few redundant terms if the Strassen construction has overlaps.
    
    # If no improvement, return the best found.
    
    # Final Verification
    valid, err = _verify(best_U, best_V, best_W)
    if not valid:
        # Fallback to original 44
        best_U, best_V, best_W = U_full, V_full, W_full
        best_rank = 44

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
