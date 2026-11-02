import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct by checking tensor identity."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    
    # Construct the decomposition tensor T_decomp[k*m+i, r*n+j, t*p+q] 
    # Actually, let's use the standard check:
    # For each term t, we have vectors u_t, v_t, w_t
    # U[i*m+k, t] = u_t(i,k)
    # V[k*p+j, t] = v_t(k,j)
    # W[j*n+i, t] = w_t(j,i)
    
    # The tensor is sum_t u_t \otimes v_t \otimes w_t
    # We need this to equal the multiplication tensor T[k*m+i, r*n+j, t*p+q]
    # T[k*m+i, r*n+j, t*p+q] = 1 if i==r and k==t and j==q else 0
    
    # Let's reshape U, V, W to 3D tensors
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # Contract over r
    # Result[i, k, j] = sum_r U3[i, k, r] * V3[k, j, r] * ... wait
    # The term is U[i,k,t] * V[k,j,t] * W[j,i,t]?
    # Let's check the indices.
    # U is (n*m, r). Index: i*m + k
    # V is (m*p, r). Index: k*p + j
    # W is (p*n, r). Index: j*n + i
    
    # The product for term t is U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
    # This product should be non-zero only if it contributes to C[i,j] from A[i,k]B[k,j]
    # Specifically, the tensor identity is:
    # sum_t U[i,m+k,t] V[k,p+j,t] W[j,n+i,t] = delta_{i,r} delta_{k,t} delta_{j,q}
    # Wait, the standard matrix multiplication tensor T is defined by:
    # T_{i,j,k} = delta_{i,k'} delta_{k,j'} ...
    # Let's just evaluate the sum for all i,j,k and see if it matches delta.
    
    # Actually, a faster check:
    # Generate random A (n x m) and B (m x p)
    # Compute C = A @ B
    # Compute C_decomp = sum_t (U_vec[t] @ A) * (B @ V_vec[t]) ... no.
    
    # The decomposition says:
    # C_{i,j} = sum_k A_{i,k} B_{k,j}
    # Also C_{i,j} = sum_t (sum_k A_{i,k} U_{i*m+k, t}) (sum_k B_{k,j} V_{k*p+j, t}) ? No.
    
    # The standard form is:
    # C_{i,j} = sum_t M_t A_t B_t
    # Where A_t is a linear combo of A, B_t is linear combo of B, M_t is scalar.
    # In the U,V,W formulation:
    # C_{i,j} = sum_t (sum_{k} U[i*m+k, t] A_{i,k}) * (sum_{l} V[l*p+j, t] B_{l,j}) * W[j*n+i, t] ?
    # No, W is usually just 1 for the specific i,j.
    
    # Let's check the trace convention given: W[k*n+i, r]
    # This implies the output index is mapped differently or W absorbs the output selection.
    
    # Let's use the tensor contraction check which is robust.
    # T_decomp[a, b, c] = sum_t U[a, t] V[b, t] W[c, t]
    # We want T_decomp == T_mm
    # T_mm[a, b, c] = 1 if a=i*m+k, b=k*p+j, c=j*n+i for some i,j,k else 0.
    # Actually, T_mm is the set of triples (i*m+k, k*p+j, j*n+i)
    
    # Let's create the target tensor (sparse)
    # n=3, m=5, p=6
    # Size: (15, 30, 18)
    
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i*m + k
                b = k*p + j
                c = j*n + i
                target[a, b, c] = 1.0
                
    # Compute decomp tensor
    # U: (15, r), V: (30, r), W: (18, r)
    # Tensor = np.einsum('it,jt,kt->ijk', U, V, W)
    decomp = np.einsum('it,jt,kt->ijk', U, V, W)
    
    diff = np.abs(decomp - target)
    return np.max(diff) < tol

def construct():
    """
    Returns {'U', 'V', 'W'} with integer entries that exactly decompose
    the 3x5 @ 5x6 matrix multiplication tensor.
    
    Strategy:
    Block decomposition:
    A (3x5) split into A1 (2x2), A2 (2x3), A3 (1x2), A4 (1x3)
    B (5x6) split into B1 (2x2), B2 (2x4), B3 (3x2), B4 (3x4)
    
    C = A1 B1 + A2 B3 + A3 B2 + A4 B4
    
    Use Strassen (Rank 7) for A1 B1 (2x2 * 2x2)
    Use Schoolbook (Rank 12) for A2 B3 (2x3 * 3x2)
    Use Schoolbook (Rank 8) for A3 B2 (1x2 * 2x4)
    Use Schoolbook (Rank 12) for A4 B4 (1x3 * 3x4)
    
    Total Rank = 7 + 12 + 8 + 12 = 39? Wait.
    A1(2x2) B1(2x2) -> 2x2. Rank 7.
    A2(2x3) B3(3x2) -> 2x2. Rank 2*3*2 = 12.
    A3(1x2) B2(2x4) -> 1x4. Rank 1*2*4 = 8.
    A4(1x3) B4(3x4) -> 1x4. Rank 1*3*4 = 12.
    Total = 7 + 12 + 8 + 12 = 39.
    
    Wait, is the split valid?
    A = [ A1 A2 ]  (2x2 + 2x3)
        [ A3 A4 ]  (1x2 + 1x3)
    B = [ B1 B2 ]  (2x2 + 2x4)
        [ B3 B4 ]  (3x2 + 3x4)
        
    C = A B
    C[0:2, 0:2] = A1 B1 + A2 B3
    C[0:2, 2:6] = A1 B2 + A2 B4
    C[2:3, 0:2] = A3 B1 + A4 B3
    C[2:3, 2:6] = A3 B2 + A4 B4
    
    My previous sum was C = A1 B1 + A2 B3 + A3 B2 + A4 B4.
    I missed A1 B2, A2 B4, A3 B1, A4 B3.
    
    Let's re-calculate ranks for all 8 blocks:
    1. A1 B1: 2x2 * 2x2. Rank 7 (Strassen).
    2. A2 B3: 2x3 * 3x2. Rank 12 (Schoolbook).
    3. A1 B2: 2x2 * 2x4. Rank 16 (Schoolbook).
    4. A2 B4: 2x3 * 3x4. Rank 24 (Schoolbook).
    5. A3 B1: 1x2 * 2x2. Rank 4 (Schoolbook).
    6. A4 B3: 1x3 * 3x2. Rank 6 (Schoolbook).
    7. A3 B2: 1x2 * 2x4. Rank 8 (Schoolbook).
    8. A4 B4: 1x3 * 3x4. Rank 12 (Schoolbook).
    
    Total Rank = 7 + 12 + 16 + 24 + 4 + 6 + 8 + 12 = 89.
    
    This is better than 90.
    """
    deadline = time.time() + 140.0
    n, m, p = 3, 5, 6
    
    rank = 89
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1, col=0):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
    
    col = 0
    
    # Helper to add schoolbook terms for block A_sub (rows x mid) * B_sub (mid x cols)
    # A_sub is located at rows [r_start:r_end], cols [c_start:c_end] of A
    # B_sub is located at rows [r_start:r_end] of B, cols [c_start:c_end] of B
    # Result is at rows [r_start:r_end] of C, cols [c_start:c_end] of C
    
    def add_schoolbook_block(A_r_start, A_r_end, A_c_start, A_c_end, 
                             B_r_start, B_r_end, B_c_start, B_c_end):
        nonlocal col
        # A_sub dimensions: (A_r_end - A_r_start) x (A_c_end - A_c_start)
        # B_sub dimensions: (B_r_end - B_r_start) x (B_c_end - B_c_start)
        # Check dims: A_c_end - A_c_start should equal B_r_end - B_r_start
        assert A_c_end - A_c_start == B_r_end - B_r_start
        
        for i in range(A_r_start, A_r_end):
            for k in range(A_c_start, A_c_end):
                for j in range(B_c_start, B_c_end):
                    # A index: i*m + k
                    # B index: k*p + j
                    # C index: j*n + i
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
                    if time.time() > deadline:
                        return False
        return True
    
    # Block 1: A1 (0:2, 0:2) * B1 (0:2, 0:2) -> C1 (0:2, 0:2)
    # Use Strassen
    # A1 = [[a11, a12], [a21, a22]]
    # B1 = [[b11, b12], [b21, b22]]
    # C1 = [[c11, c12], [c21, c22]]
    
    # Strassen Terms:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 - P4 + P6 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 + P2 + P5
    
    # We need to map these to U, V, W vectors.
    # Each Pk corresponds to a product of a linear combo of A1 entries and a linear combo of B1 entries.
    # Let u_k be the vector for A part, v_k for B part.
    # For term t, we add u_k * v_k * w to the decomposition.
    # W is determined by how Pk contributes to C11, C12, C21, C22.
    
    # Let's define the linear combos for A1 and B1.
    # A1 indices in global A:
    # a11: (0,0) -> idx 0
    # a12: (0,1) -> idx 1
    # a21: (1,0) -> idx 5 (1*5+0)
    # a22: (1,1) -> idx 6 (1*5+1)
    
    # B1 indices in global B:
    # b11: (0,0) -> idx 0
    # b12: (0,1) -> idx 1
    # b21: (1,0) -> idx 6 (1*6+0)
    # b22: (1,1) -> idx 7 (1*6+1)
    
    # C1 indices in global C:
    # c11: (0,0) -> idx 0 (0*3+0)
    # c12: (0,1) -> idx 1 (1*3+0) -- Wait, W index is j*n+i.
    # W index for C[i,j] is j*n + i.
    # c11 (i=0,j=0) -> 0*3+0 = 0
    # c12 (i=0,j=1) -> 1*3+0 = 3
    # c21 (i=1,j=0) -> 0*3+1 = 1
    # c22 (i=1,j=1) -> 1*3+1 = 4
    
    # Strassen Coefficients for Pk:
    # P1: A: (1,1,0,0) for (a11,a12,a21,a22)? No, (a11+a22). So (1,0,0,1).
    #      B: (b11+b22) -> (1,0,0,1) for (b11,b12,b21,b22).
    # P2: A: (a21+a22) -> (0,0,1,1).
    #      B: b11 -> (1,0,0,0).
    # P3: A: a11 -> (1,0,0,0).
    #      B: (b12-b22) -> (0,1,0,-1).
    # P4: A: a22 -> (0,0,0,1).
    #      B: (b21-b11) -> (-1,0,1,0).
    # P5: A: (a11+a12) -> (1,1,0,0).
    #      B: b22 -> (0,0,0,1).
    # P6: A: (a21-a11) -> (-1,0,1,0).
    #      B: (b11+b12) -> (1,1,0,0).
    # P7: A: (a12-a22) -> (0,1,0,-1).
    #      B: (b21+b22) -> (0,0,1,1).
    
    # Contributions to C:
    # C11 = P1 - P4 + P6 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P3 + P2 + P5
    
    # For each Pk, we have one column in U,V,W?
    # No, Pk is one scalar product.
    # The term is Pk * (contribution to C).
    # But C is a matrix.
    # So for each Pk, we add terms to each C element it affects.
    # Example: P1 affects C11 (+1) and C22 (+1).
    # So we need two terms?
    # No, the decomposition is a sum of rank-1 tensors.
    # Pk * delta_{C11} is one term.
    # Pk * delta_{C22} is another term.
    # This would increase the rank.
    
    # Wait, Strassen reduces the number of multiplications, but the rank of the tensor
    # is the number of terms in the sum.
    # In Strassen, we compute 7 products.
    # Then we combine them with additions.
    # The additions are "free" in the rank count?
    # No, the rank of the tensor is the minimum number of terms in the bilinear decomposition.
    # Strassen shows that the rank of 2x2 matrix multiplication is <= 7.
    # This means there exist 7 vectors u_t, v_t, w_t such that
    # T = sum_t u_t \otimes v_t \otimes w_t.
    # The w_t vectors are not just delta_{c11} etc. They are linear combinations that reconstruct the C entries.
    
    # Let's find the explicit rank-7 decomposition for 2x2.
    # Source: Strassen (1969)
    
    # The 7 terms are:
    # 1. u1 = a11+a22, v1 = b11+b22, w1 = c11 + c22 - c12 - c21 ? No.
    
    # Let's look up the explicit tensor decomposition for Strassen.
    # T_{i,j,k} = sum_{t=1}^7 u_{t,i,k} v_{t,k,j} w_{t,j,i}
    
    # Actually, it's easier to just use the known identity.
    # The rank-7 decomposition corresponds to the 7 products.
    # The w_t vectors are defined by the linear combinations used to recover C.
    
    # Let's define the 7 products M_t:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M5
    
    # In the tensor decomposition, each M_t corresponds to a term in the sum.
    # But M_t contributes to multiple C entries.
    # This means the w_t vector for term t has non-zero entries at those C positions.
    # Specifically:
    # t=1: w1 has 1 at C11, 1 at C22.
    # t=2: w2 has 1 at C21, 1 at C22.
    # t=3: w3 has 1 at C12, -1 at C22.
    # t=4: w4 has -1 at C11, 1 at C21.
    # t=5: w5 has 1 at C12, 1 at C22.
    # t=6: w6 has 1 at C11.
    # t=7: w7 has 1 at C11.
    
    # Let's verify:
    # C11 = 1(M1) + 0(M2) + 0(M3) - 1(M4) + 0(M5) + 1(M6) + 1(M7) = M1 - M4 + M6 + M7. Correct.
    # C12 = 0(M1) + 0(M2) + 1(M3) + 0(M4) + 1(M5) + 0(M6) + 0(M7) = M3 + M5. Correct.
    # C21 = 0(M1) + 1(M2) + 0(M3) + 1(M4) + 0(M5) + 0(M6) + 0(M7) = M2 + M4. Correct.
    # C22 = 1(M1) + 1(M2) - 1(M3) + 0(M4) + 1(M5) + 0(M6) + 0(M7) = M1 + M2 - M3 + M5. Correct.
    
    # So we can add these 7 terms.
    
    # Define indices for A1, B1, C1
    # A1: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # B1: (0,0)=0, (0,1)=1, (1,0)=6, (1,1)=7
    # C1: (0,0)=0, (0,1)=3, (1,0)=1, (1,1)=4
    
    A_idx_11, A_idx_12, A_idx_21, A_idx_22 = 0, 1, 5, 6
    B_idx_11, B_idx_12, B_idx_21, B_idx_22 = 0, 1, 6, 7
    C_idx_11, C_idx_12, C_idx_21, C_idx_22 = 0, 3, 1, 4
    
    # Term 1: M1
    # u: a11+a22 -> [1, 0, 0, 1]
    # v: b11+b22 -> [1, 0, 0, 1]
    # w: c11+c22 -> [1, 0, 1, 1] (at c11, c22)
    
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        # u_coeffs is list of (idx, coeff)
        # v_coeffs is list of (idx, coeff)
        # w_coeffs is list of (idx, coeff)
        
        # This is NOT a single rank-1 tensor if u and v are sums.
        # Wait. A rank-1 tensor is u \otimes v \otimes w where u,v,w are vectors.
        # If u = u1 + u2, then u \otimes v \otimes w = (u1+u2) \otimes v \otimes w = u1 \otimes v \otimes w + u2 \otimes v \otimes w.
        # This would be TWO terms.
        # So Strassen's "7 multiplications" does NOT directly translate to rank 7 tensor decomposition
        # if the linear combinations are not rank-1 themselves?
        # No, u is a vector in the space of A entries. It IS a single vector.
        # The tensor is sum_t u_t \otimes v_t \otimes w_t.
        # u_t is a vector of length 4 (for 2x2).
        # So yes, it is a single term in the sum.
        
        for idx, coeff in u_coeffs:
            U[idx][col] = coeff
        for idx, coeff in v_coeffs:
            V[idx][col] = coeff
        for idx, coeff in w_coeffs:
            W[idx][col] = coeff
        col += 1
        if time.time() > deadline:
            return False
        return True
    
    # T1
    if not add_strassen_term([(A_idx_11,1), (A_idx_22,1)], [(B_idx_11,1), (B_idx_22,1)], [(C_idx_11,1), (C_idx_22,1)]): return {"U": U, "V": V, "W": W}
    # T2
    if not add_strassen_term([(A_idx_21,1), (A_idx_22,1)], [(B_idx_11,1)], [(C_idx_21,1), (C_idx_22,1)]): return {"U": U, "V": V, "W": W}
    # T3
    if not add_strassen_term([(A_idx_11,1)], [(B_idx_12,1), (B_idx_22,-1)], [(C_idx_12,1), (C_idx_22,-1)]): return {"U": U, "V": V, "W": W}
    # T4
    if not add_strassen_term([(A_idx_22,1)], [(B_idx_21,1), (B_idx_11,-1)], [(C_idx_11,-1), (C_idx_21,1)]): return {"U": U, "V": V, "W": W}
    # T5
    if not add_strassen_term([(A_idx_11,1), (A_idx_12,1)], [(B_idx_22,1)], [(C_idx_12,1), (C_idx_22,1)]): return {"U": U, "V": V, "W": W}
    # T6
    if not add_strassen_term([(A_idx_21,1), (A_idx_11,-1)], [(B_idx_11,1), (B_idx_12,1)], [(C_idx_11,1)]): return {"U": U, "V": V, "W": W}
    # T7
    if not add_strassen_term([(A_idx_12,1), (A_idx_22,-1)], [(B_idx_21,1), (B_idx_22,1)], [(C_idx_11,1)]): return {"U": U, "V": V, "W": W}
    
    # Block 2: A2 (0:2, 2:5) * B3 (2:5, 0:2) -> C21 (0:2, 0:2)
    # A2 is 2x3. B3 is 3x2.
    # Schoolbook.
    if not add_schoolbook_block(0, 2, 2, 5, 2, 5, 0, 2): return {"U": U, "V": V, "W": W}
    
    # Block 3: A1 (0:2, 0:2) * B2 (0:2, 2:6) -> C12 (0:2, 2:6)
    # A1 is 2x2. B2 is 2x4.
    # Schoolbook.
    if not add_schoolbook_block(0, 2, 0, 2, 0, 2, 2, 6): return {"U": U, "V": V, "W": W}
    
    # Block 4: A2 (0:2, 2:5) * B4 (2:5, 2:6) -> C22 (0:2, 2:6)
    # A2 is 2x3. B4 is 3x4.
    # Schoolbook.
    if not add_schoolbook_block(0, 2, 2, 5, 2, 5, 2, 6): return {"U": U, "V": V, "W": W}
    
    # Block 5: A3 (2:3, 0:2) * B1 (0:2, 0:2) -> C31 (2:3, 0:2)
    # A3 is 1x2. B1 is 2x2.
    # Schoolbook.
    if not add_schoolbook_block(2, 3, 0, 2, 0, 2, 0, 2): return {"U": U, "V": V, "W": W}
    
    # Block 6: A4 (2:3, 2:5) * B3 (2:5, 0:2) -> C32 (2:3, 0:2)
    # A4 is 1x3. B3 is 3x2.
    # Schoolbook.
    if not add_schoolbook_block(2, 3, 2, 5, 2, 5, 0, 2): return {"U": U, "V": V, "W": W}
    
    # Block 7: A3 (2:3, 0:2) * B2 (0:2, 2:6) -> C33 (2:3, 2:6)
    # A3 is 1x2. B2 is 2x4.
    # Schoolbook.
    if not add_schoolbook_block(2, 3, 0, 2, 0, 2, 2, 6): return {"U": U, "V": V, "W": W}
    
    # Block 8: A4 (2:3, 2:5) * B4 (2:5, 2:6) -> C34 (2:3, 2:6)
    # A4 is 1x3. B4 is 3x4.
    # Schoolbook.
    if not add_schoolbook_block(2, 3, 2, 5, 2, 5, 2, 6): return {"U": U, "V": V, "W": W}
    
    # Check col
    if col != rank:
        # Fallback or error
        pass
    
    # Verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 90
        U = [[0] * (n*m*p) for _ in range(n * m)]
        V = [[0] * (n*m*p) for _ in range(m * p)]
        W = [[0] * (n*m*p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        
    return {"U": U, "V": V, "W": W}
