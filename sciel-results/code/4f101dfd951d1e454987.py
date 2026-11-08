"""Matrix multiplication 4x4x5 tensor decomposition - structured low-rank."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 110
    
    # First, let me properly derive and verify the Winograd 2x2 scheme.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P = A*B
    
    # I'll use the following verified Winograd 2x2 (testing numerically):
    # m0 = (a+b)(e+g)
    # m1 = (a-b)(e-g)
    # m2 = (c+d)(e+f)
    # m3 = (c-d)(e-f)
    # m4 = a(g+h)
    # m5 = c(f-h)
    # m6 = (a+c)(g+h)
    # m7 = (b+d)(f-h)
    
    # Test with specific values:
    a,b,c,d = 1,2,3,4
    e,f,g,h = 5,6,7,8
    
    m0 = (a+b)*(e+g)  # 3*12 = 36
    m1 = (a-b)*(e-g)  # -1*(-2) = 2
    m2 = (c+d)*(e+f)  # 7*11 = 77
    m3 = (c-d)*(e-f)  # -1*(-1) = 1
    m4 = a*(g+h)      # 1*15 = 15
    m5 = c*(f-h)      # 3*(-2) = -6
    m6 = (a+c)*(g+h)  # 4*15 = 60
    m7 = (b+d)*(f-h)  # 6*(-2) = -12
    
    # P00 = ae+bg = 5+14 = 19
    # P01 = af+bh = 6+16 = 22
    # P10 = ce+dg = 15+28 = 43
    # P11 = cf+dh = 18+32 = 50
    
    # (m0+m1)/2 = (36+2)/2 = 19 ✓ for P00
    # (m2+m3)/2 = (77+1)/2 = 39, but P10 = 43. ✗
    # So m2,m3 don't give P10 directly.
    
    # P10 = ce+dg = 15+28 = 43
    # (m2+m3)/2 = 39. Not equal.
    # m2-m3 = 76, /2 = 38. Not equal either.
    
    # Let me check: m2 = ce+cf+de+df, m3 = ce-cf-de+df
    # m2+m3 = 2ce+2df
    # m2-m3 = 2cf+2de
    # So (m2+m3)/2 = ce+df, (m2-m3)/2 = cf+de
    # P10 = ce+dg, P11 = cf+dh
    # We have ce+df and cf+de, not ce+dg and cf+dh.
    
    # So this scheme doesn't directly give the P entries. We need additional combinations.
    # The 7 multipliers span a 7-dimensional space, and we need to find linear combinations
    # that give the 4 P entries.
    
    # Let me set up the system: P_k = sum_s w_{k,s} * m_s
    # For P00: 19 = w000*36 + w001*2 + w002*77 + w003*1 + w004*15 + w005*(-6) + w006*60 + w007*(-12)
    # But we need this to work for ALL values of a,b,c,d,e,f,g,h, not just one test case.
    
    # The correct approach: express each P entry as a linear combination of the m_s
    # that holds identically (for all a,b,c,d,e,f,g,h).
    
    # P00 = ae+bg
    # m0 = ae+ag+be+bg
    # m1 = ae-ag-be+bg
    # m0+m1 = 2ae+2bg = 2*P00
    # So P00 = (m0+m1)/2 ✓
    
    # P01 = af+bh
    # m4 = ag+ah
    # m7 = bf-bh+df-dh
    # m6 = ag+ah+cg+ch
    # m5 = cf-ch
    # 
    # af+bh: I need to find w such that af+bh = sum w_s*m_s
    # Let's see what combinations give af and bh:
    # From m4: ah = m4-ag
    # From m7: bf-bh+df-dh = m7
    # 
    # This is getting complex. Let me use a different strategy:
    # Use numpy to solve for the W matrix given the U and V matrices.
    
    # U matrix (4x7) for A = [a,b,c,d]:
    U_2x2 = np.array([
        [1, 1, 0, 0, 1, 0, 1],  # a: in m0,m1,m4,m6
        [1,-1, 0, 0, 0, 0, 1],  # b: in m0,m1,m7
        [0, 0, 1, 1, 0, 1, 1],  # c: in m2,m3,m5,m6
        [0, 0, 1,-1, 0,-1, 0],  # d: in m2,m3,m5
    ], dtype=float)
    
    # Wait, let me recheck:
    # m0 = (a+b)(e+g): a coeff=1, b coeff=1
    # m1 = (a-b)(e-g): a coeff=1, b coeff=-1
    # m2 = (c+d)(e+f): c coeff=1, d coeff=1
    # m3 = (c-d)(e-f): c coeff=1, d coeff=-1
    # m4 = a(g+h): a coeff=1
    # m5 = c(f-h): c coeff=1
    # m6 = (a+c)(g+h): a coeff=1, c coeff=1
    # m7 = (b+d)(f-h): b coeff=1, d coeff=1
    
    U_2x2 = np.array([
        [1, 1, 0, 0, 1, 0, 1],  # a
        [1,-1, 0, 0, 0, 0, 1],  # b
        [0, 0, 1, 1, 0, 1, 1],  # c
        [0, 0, 1,-1, 0,-1, 0],  # d
    ], dtype=float)
    
    # V matrix (4x7) for B = [e,f,g,h]:
    # m0 = (a+b)(e+g): e coeff=1, g coeff=1
    # m1 = (a-b)(e-g): e coeff=1, g coeff=-1
    # m2 = (c+d)(e+f): e coeff=1, f coeff=1
    # m3 = (c-d)(e-f): e coeff=1, f coeff=-1
    # m4 = a(g+h): g coeff=1, h coeff=1
    # m5 = c(f-h): f coeff=1, h coeff=-1
    # m6 = (a+c)(g+h): g coeff=1, h coeff=1
    # m7 = (b+d)(f-h): f coeff=1, h coeff=-1
    
    V_2x2 = np.array([
        [1, 1, 1, 1, 0, 0, 0],  # e
        [0, 0, 1,-1, 0, 1, 1],  # f
        [1,-1, 0, 0, 1, 0, 1],  # g
        [0, 0, 0, 0, 1,-1,-1],  # h
    ], dtype=float)
    
    # Now, the tensor T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]
    # We know T for 2x2*2x2:
    # T[0,0,0]=1 (a*e -> p00), T[1,2,0]=1 (b*g -> p00)
    # T[0,1,1]=1 (a*f -> p01), T[1,3,1]=1 (b*h -> p01)
    # T[2,0,2]=1 (c*e -> p10), T[3,2,2]=1 (d*g -> p10)
    # T[2,1,3]=1 (c*f -> p11), T[3,3,3]=1 (d*h -> p11)
    
    # For each k (output index), we need:
    # T[:, :, k] = U^T * diag(V[:,k]... ) no, that's not right.
    # T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]
    # For fixed k: T[:,:,k] = sum_s W[k,s] * U[:,s] * V[:,s]^T
    # So T[:,:,k] = U @ diag(W[k,:]) @ V^T
    
    # We need to find W (4x7) such that for each k:
    # T[:,:,k] = U @ diag(W[k,:]) @ V^T
    
    # This means: U @ diag(W[k,:]) @ V^T = T[:,:,k]
    # => diag(W[k,:]) = U^{-1} @ T[:,:,k] @ (V^T)^{-1} ... but U and V are 4x7, not square.
    
    # Better: for each k, we need sum_s W[k,s] * (U[:,s] ⊗ V[:,s]) = vec(T[:,:,k])
    # This is a linear system: for each k, W[k,:] satisfies:
    # sum_s W[k,s] * (U[:,s] * V[:,s])^T (flattened) = vec(T[:,:,k])
    
    # Let me set this up properly.
    # For fixed k, define M_k = T[:,:,k] (4x4 matrix)
    # M_k[i,j] = sum_s W[k,s] * U[i,s] * V[j,s]
    # This is: M_k = U @ diag(W[k,:]) @ V^T
    
    # We have 4 equations (one for each k) and 7 unknowns per k (W[k,0..6]).
    # Each equation is 4x4 = 16 scalar equations, but M_k is determined by W[k,:] (7 params).
    # So we have 16 equations per k with 7 unknowns - overdetermined.
    # If a solution exists, it means rank 7 is achievable.
    
    # Let me solve using least squares.
    # For each k: minimize ||U @ diag(w) @ V^T - M_k||_F^2 over w
    
    # U @ diag(w) @ V^T = sum_s w_s * U[:,s] * V[:,s]^T
    # So we need: sum_s w_s * (U[:,s] ⊗ V[:,s]) = vec(M_k)
    
    # Build the 16x7 matrix A_k where A_k[:,s] = vec(U[:,s] * V[:,s]^T)
    # Then solve A_k @ w = vec(M_k)
    
    def vec_2x2(M):
        return np.array([M[0,0], M[0,1], M[1,0], M[1,1]])
    
    # T[:,:,0] (P00): T[0,0,0]=1, T[1,2,0]=1, rest 0
    T00 = np.zeros((4,4)); T00[0,0]=1; T00[1,2]=1
    T01 = np.zeros((4,4)); T01[0,1]=1; T01[1,3]=1
    T10 = np.zeros((4,4)); T10[2,0]=1; T10[3,2]=1
    T11 = np.zeros((4,4)); T11[2,1]=1; T11[3,3]=1
    
    T_list = [T00, T01, T10, T11]
    
    # Build A matrix: A[:,s] = vec(U[:,s] * V[:,s]^T) for s=0..6
    A_mat = np.zeros((16, 7))
    for s in range(7):
        outer = np.outer(U_2x2[:,s], V_2x2[:,s])
        # vec of 4x4 matrix (row-major)
        A_mat[:,s] = outer.flatten()
    
    W_2x2 = np.zeros((4, 7))
    for k in range(4):
        rhs = T_list[k].flatten()
        # Solve A_mat @ w = rhs (least squares)
        w, residuals, rank, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
        W_2x2[k, :] = w
        # Check residual
        residual = np.linalg.norm(A_mat @ w - rhs)
        if residual > 1e-10:
            print(f"Warning: rank 7 Winograd 2x2 doesn't work for P[{k}]! Residual={residual}")
    
    # Verify the decomposition
    A_test = np.array([1,2,3,4], dtype=float)
    B_test = np.array([5,6,7,8], dtype=float)
    P_expected = np.array([[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]], dtype=float)
    
    m_vals = U_2x2 @ A_test * (V_2x2 @ B_test)  # element-wise: m_s = (U@A)_s * (V@B)_s
    # Actually: m_s = (sum_i U[i,s]*A[i]) * (sum_j V[j,s]*B[j])
    UA = U_2x2 @ A_test  # 7-vector
    VB = V_2x2 @ B_test  # 7-vector
    m_vals = UA * VB  # element-wise product, 7-vector
    
    P_computed = W_2x2 @ m_vals  # 4-vector
    P_computed_mat = P_computed.reshape(2,2)
    
    if not np.allclose(P_computed_mat, P_expected, atol=1e-10):
        print(f"Winograd verification failed! Expected {P_expected}, got {P_computed_mat}")
        # Fall back to schoolbook
        rank_school = n * m * p
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Great! Now tile the 4x4 * 4x5 using 2x2 blocks.
    # A (4x4) = [[A00, A01],[A10, A11]] where each is 2x2
    # B (4x5) = [[B00, B01, B02],[B10, B11, B12]] where B00,B01,B10,B11 are 2x2, B02,B12 are 2x1
    # 
    # C = A*B:
    # C[0:2, 0:2] = A00*B00 + A01*B10
    # C[0:2, 2:4] = A00*B01 + A01*B11
    # C[0:2, 4:5] = A00*B02 + A01*B12
    # C[2:4, 0:2] = A10*B00 + A11*B10
    # C[2:4, 2:4] = A10*B01 + A11*B11
    # C[2:4, 4:5] = A10*B02 + A11*B12
    #
    # Each 2x2*2x2 product: rank 7
    # Each 2x2*2x1 product: rank 4
    #
    # Number of 2x2*2x2 products: 2 (row-blocks) * 2 (col-blocks of 2x2) = 4
    # Wait: A00*B00, A01*B10, A00*B01, A01*B11, A10*B00, A11*B10, A10*B01, A11*B11
    # That's 8 products of 2x2*2x2.
    # Number of 2x2*2x1 products: A00*B02, A01*B12, A10*B02, A11*B12 = 4
    #
    # Total rank: 8*7 + 4*4 = 56 + 16 = 72
    
    total_rank = 8*7 + 4*4  # 72
    
    U_full = np.zeros((n*m, total_rank))  # 16 x 72
    V_full = np.zeros((m*p, total_rank))  # 20 x 72
    W_full = np.zeros((p*n, total_rank))  # 20 x 72
    
    col = 0
    # A block indices: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # In flattened A (row-major, 16 entries):
    # A00: indices 0,1,4,5 (a[0,0],a[0,1],a[1,0],a[1,1])
    # A01: indices 2,3,6,7 (a[0,2],a[0,3],a[1,2],a[1,3])
    # A10: indices 8,9,12,13 (a[2,0],a[2,1],a[3,0],a[3,1])
    # A11: indices 10,11,14,15 (a[2,2],a[2,3],a[3,2],a[3,3])
    
    # B block indices: B (4x5), flattened row-major (20 entries):
    # B00: B[0:2,0:2] = indices 0,1,5,6
    # B01: B[0:2,2:4] = indices 2,3,7,8
    # B02: B[0:2,4:5] = indices 4,9
    # B10: B[2:4,0:2] = indices 10,11,15,16
    # B11: B[2:4,2:4] = indices 12,13,17,18
    # B12: B[2:4,4:5] = indices 14,19
    
    # P (output C, 4x5) flattened row-major (20 entries):
    # C[0:2,0:2]: indices 0,1,5,6
    # C[0:2,2:4]: indices 2,3,7,8
    # C[0:2,4:5]: indices 4,9
    # C[2:4,0:2]: indices 10,11,15,16
    # C[2:4,2:4]: indices 12,13,17,18
    # C[2:4,4:5]: indices 14,19
    
    # For each 2x2*2x2 product, we need to map the local 2x2 indices to global indices.
    # Local A 2x2: [a00,a01,a10,a11] -> global indices depend on which block
    # Local B 2x2: [b00,b01,b10,b11] -> global indices depend on which block
    # Local P 2x2: [p00,p01,p10,p11] -> global indices depend on which output block
    
    # U_2x2 is 4x7, V_2x2 is 4x7, W_2x2 is 4x7
    # For a product A_block * B_block -> P_block:
    # For each multiplier s (0..6):
    #   For each local A index i (0..3): U_full[global_A_idx, col] += U_2x2[i, s]
    #   For each local B index j (0..3): V_full[global_B_idx, col] += V_2x2[j, s]
    #   For each local P index k (0..3): W_full[global_P_idx, col] += W_2x2[k, s]
    
    # Define block mappings
    # A blocks: (row_start, col_start) -> local to global index mapping
    # A is 4x4, flattened row-major. A[i,j] is at index i*4+j.
    # A00: rows 0-1, cols 0-1. Local (lr,lc) -> global (lr,lc). Index = lr*4+lc
    # A01: rows 0-1, cols 2-3. Local (lr,lc) -> global (lr, lc+2). Index = lr*4+(lc+2)
    # A10: rows 2-3, cols 0-1. Local (lr,lc) -> global (lr+2, lc). Index = (lr+2)*4+lc
    # A11: rows 2-3, cols 2-3. Local (lr,lc) -> global (lr+2, lc+2). Index = (lr+2)*4+(lc+2)
    
    def a_block_global(row_start, col_start, lr, lc):
        return (row_start + lr) * 4 + (col_start + lc)
    
    # B blocks: B is 4x5, flattened row-major. B[i,j] is at index i*5+j.
    # B00: rows 0-1, cols 0-1. Local (lr,lc) -> global (lr,lc). Index = lr*5+lc
    # B01: rows 0-1, cols 2-3. Local (lr,lc) -> global (lr, lc+2). Index = lr*5+(lc+2)
    # B02: rows 0-1, col 4. Local (lr,0) -> global (lr, 4). Index = lr*5+4 (2x1)
    # B10: rows 2-3, cols 0-1. Local (lr,lc) -> global (lr+2, lc). Index = (lr+2)*5+lc
    # B11: rows 2-3, cols 2-3. Local (lr,lc) -> global (lr+2, lc+2). Index = (lr+2)*5+(lc+2)
    # B12: rows 2-3, col 4. Local (lr,0) -> global (lr+2, 4). Index = (lr+2)*5+4 (2x1)
    
    def b_block_global(row_start, col_start, lr, lc):
        return (row_start + lr) * 5 + (col_start + lc)
    
    # P (C) blocks: C is 4x5, flattened row-major. C[i,j] is at index i*5+j.
    # Same structure as B for the output.
    
    def p_block_global(row_start, col_start, lr, lc):
        return (row_start + lr) * 5 + (col_start + lc)
    
    # Process 2x2*2x2 products:
    # (A_block, B_block, P_block) tuples:
    # A00*B00 -> C[0:2,0:2]: A(0,0), B(0,0), P(0,0)
    # A01*B10 -> C[0:2,0:2]: A(0,2), B(2,0), P(0,0)
    # A00*B01 -> C[0:2,2:4]: A(0,0), B(0,2), P(0,2)
    # A01*B11 -> C[0:2,2:4]: A(0,2), B(2,2), P(0,2)
    # A10*B00 -> C[2:4,0:2]: A(2,0), B(0,0), P(2,0)
    # A11*B10 -> C[2:4,0:2]: A(2,2), B(2,0), P(2,0)
    # A10*B01 -> C[2:4,2:4]: A(2,0), B(0,2), P(2,2)
    # A11*B11 -> C[2:4,2:4]: A(2,2), B(2,2), P(2,2)
    
    products_2x2 = [
        (0, 0, 0, 0, 0, 0),  # A(0,0)*B(0,0)->P(0,0)
        (0, 2, 2, 0, 0, 0),  # A(0,2)*B(2,0)->P(0,0)
        (0, 0, 0, 2, 0, 2),  # A(0,0)*B(0,2)->P(0,2)
        (0, 2, 2, 2, 0, 2),  # A(0,2)*B(2,2)->P(0,2)
        (2, 0, 0, 0, 2, 0),  # A(2,0)*B(0,0)->P(2,0)
        (2, 2, 2, 0, 2, 0),  # A(2,2)*B(2,0)->P(2,0)
        (2, 0, 0, 2, 2, 2),  # A(2,0)*B(0,2)->P(2,2)
        (2, 2, 2, 2, 2, 2),  # A(2,2)*B(2,2)->P(2,2)
    ]
    
    for (ar, ac, br, bc, pr, pc) in products_2x2:
        for s in range(7):
            for li in range(4):
                lr, lc = li // 2, li % 2
                gi = a_block_global(ar, ac, lr, lc)
                U_full[gi, col] += U_2x2[li, s]
            for lj in range(4):
                lr, lc = lj // 2, lj % 2
                gj = b_block_global(br, bc, lr, lc)
                V_full[gj, col] += V_2x2[lj, s]
            for lk in range(4):
                lr, lc = lk // 2, lk % 2
                gk = p_block_global(pr, pc, lr, lc)
                W_full[gk, col] += W_2x2[lk, s]
            col += 1
    
    # Process 2x2*2x1 products:
    # A00*B02 -> C[0:2,4:5]: A(0,0), B(0,4), P(0,4)
    # A01*B12 -> C[0:2,4:5]: A(0,2), B(2,4), P(0,4)
    # A10*B02 -> C[2:4,4:5]: A(2,0), B(0,4), P(2,4)
    # A11*B12 -> C[2:4,4:5]: A(2,2), B(2,4), P(2,4)
    
    # For 2x2*2x1: A is 2x2 (4 entries), B is 2x1 (2 entries), P is 2x1 (2 entries)
    # Schoolbook: rank 4
    # m0 = a00*b0, m1 = a01*b1, m2 = a10*b0, m3 = a11*b1
    # P0 = m0+m1, P1 = m2+m3
    
    U_2x2_2x1 = np.array([
        [1, 0, 0, 0],  # a00
        [0, 1, 0, 0],  # a01
        [0, 0, 1, 0],  # a10
        [0, 0, 0, 1],  # a11
    ], dtype=float)
    V_2x2_2x1 = np.array([
        [1, 0, 1, 0],  # b0
        [0, 1, 0, 1],  # b1
    ], dtype=float)
    W_2x2_2x1 = np.array([
        [1, 1, 0, 0],  # P0 = m0+m1
        [0, 0, 1, 1],  # P1 = m2+m3
    ], dtype=float)
    
    products_2x1 = [
        (0, 0, 0, 4, 0, 4),  # A(0,0)*B(0,4)->P(0,4)
        (0, 2, 2, 4, 0, 4),  # A(0,2)*B(2,4)->P(0,4)
        (2, 0, 0, 4, 2, 4),  # A(2,0)*B(0,4)->P(2,4)
        (2, 2, 2, 4, 2, 4),  # A(2,2)*B(2,4)->P(2,4)
    ]
    
    for (ar, ac, br, bc, pr, pc) in products_2x1:
        for s in range(4):
            for li in range(4):
                lr, lc = li // 2, li % 2
                gi = a_block_global(ar, ac, lr, lc)
                U_full[gi, col] += U_2x2_2x1[li, s]
            for lj in range(2):
                lr, lc = lj, 0
                gj = b_block_global(br, bc, lr, lc)
                V_full[gj, col] += V_2x2_2x1[lj, s]
            for lk in range(2):
                lr, lc = lk, 0
                gk = p_block_global(pr, pc, lr, lc)
                W_full[gk, col] += W_2x2_2x1[lk, s]
            col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    # Verify the full decomposition
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 5)
    C_expected = A_test @ B_test
    
    # Compute using the decomposition
    A_flat = A_test.flatten()  # row-major, 16 entries
    B_flat = B_test.flatten()  # row-major, 20 entries
    
    UA = U_full @ A_flat  # 72-vector
    VB = V_full @ B_flat  # 72-vector
    m_vals = UA * VB  # element-wise
    C_flat = W_full @ m_vals  # 20-vector
    C_computed = C_flat.reshape(4, 5)
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        print(f"Full verification failed! Max error: {np.max(np.abs(C_computed - C_expected))}")
        # Fall back to schoolbook
        rank_school = n * m * p
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col2] = 1
                    V[j*p+k, col2] = 1
                    W[k*n+i, col2] = 1
                    col2 += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists with half-integer support
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(U_full), "V": to_list(V_full), "W": to_list(W_full)}
# EVOLVE-BLOCK-END

