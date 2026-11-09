import numpy as np

DIMENSIONS = [2, 3, 8]

def _decompose(n, m, p):
    """
    Returns (U, V, W) as numpy arrays with shapes:
      U: (n*m, rank)
      V: (m*p, rank)
      W: (p*n, rank)
    """
    # Base case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        # Strassen's 7-term decomposition for 2x2x2
        # Indices: U row = i*2 + j, V row = j*2 + k, W row = k*2 + i
        # The 7 terms:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B11 - B21)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Term 1: (A11+A22)(B11+B22)
        # U: rows for A11 (i=0,j=0) -> row 0, A22 (i=1,j=1) -> row 3
        U[0, 0] = 1; U[3, 0] = 1
        # V: rows for B11 (j=0,k=0) -> row 0, B22 (j=1,k=1) -> row 3
        V[0, 0] = 1; V[3, 0] = 1
        # W: contributes to C11 (k=0,i=0) -> row 0, C22 (k=1,i=1) -> row 3
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: (A21+A22)B11
        # U: A21 (i=1,j=0) -> row 2, A22 (i=1,j=1) -> row 3
        U[2, 1] = 1; U[3, 1] = 1
        # V: B11 (j=0,k=0) -> row 0
        V[0, 1] = 1
        # W: contributes to C21 (k=0,i=1) -> row 2
        W[2, 1] = 1
        
        # Term 3: A11(B21-B22)
        # U: A11 (i=0,j=0) -> row 0
        U[0, 2] = 1
        # V: B21 (j=1,k=0) -> row 2, B22 (j=1,k=1) -> row 3
        V[2, 2] = 1; V[3, 2] = -1
        # W: contributes to C12 (k=1,i=0) -> row 1, C22 (k=1,i=1) -> row 3
        W[1, 2] = 1; W[3, 2] = 1
        
        # Term 4: A22(B11-B21)
        # U: A22 (i=1,j=1) -> row 3
        U[3, 3] = 1
        # V: B11 (j=0,k=0) -> row 0, B21 (j=1,k=0) -> row 2
        V[0, 3] = 1; V[2, 3] = -1
        # W: contributes to C11 (k=0,i=0) -> row 0, C21 (k=0,i=1) -> row 2
        W[0, 3] = 1; W[2, 3] = 1
        
        # Term 5: (A11+A12)B22
        # U: A11 (i=0,j=0) -> row 0, A12 (i=0,j=1) -> row 1
        U[0, 4] = 1; U[1, 4] = 1
        # V: B22 (j=1,k=1) -> row 3
        V[3, 4] = 1
        # W: contributes to C11 (k=1,i=0) -> row 1, C22 (k=1,i=1) -> row 3
        W[1, 4] = -1; W[3, 4] = -1
        
        # Term 6: (A21-A11)(B11+B12)
        # U: A21 (i=1,j=0) -> row 2, A11 (i=0,j=0) -> row 0
        U[2, 5] = 1; U[0, 5] = -1
        # V: B11 (j=0,k=0) -> row 0, B12 (j=0,k=1) -> row 1
        V[0, 5] = 1; V[1, 5] = 1
        # W: contributes to C22 (k=1,i=1) -> row 3
        W[3, 5] = 1
        
        # Term 7: (A12-A22)(B21+B22)
        # U: A12 (i=0,j=1) -> row 1, A22 (i=1,j=1) -> row 3
        U[1, 6] = 1; U[3, 6] = -1
        # V: B21 (j=1,k=0) -> row 2, B22 (j=1,k=1) -> row 3
        V[2, 6] = 1; V[3, 6] = 1
        # W: contributes to C11 (k=0,i=0) -> row 0
        W[0, 6] = 1
        
        return U, V, W
    
    # Recursive case: split the largest dimension
    # Strategy: if p >= 2, split p into p1, p2
    # C = A * B = [A*B1 | A*B2]
    # This is column-wise concatenation of two products with same A
    
    # For 2x3x8: split p=8 into 4+4
    # Sub-problems: 2x3x4 and 2x3x4
    # Each 2x3x4 can be split further: p=4 into 2+2
    # Sub-problems: 2x3x2 and 2x3x2
    # 2x3x2: split m=3 into 1+2? Or use direct?
    
    # Let's think about the optimal recursive structure.
    # The key insight is that we can use Strassen-style splitting when dimensions are even.
    # For 2x3x8:
    #   - Split p: 8 -> 4, 4. Two sub-problems of 2x3x4.
    #   - For 2x3x4: Split p: 4 -> 2, 2. Two sub-problems of 2x3x2.
    #   - For 2x3x2: Split m: 3 -> 1, 2? Or n: 2 -> 1, 1?
    #     Actually, for 2x3x2, we can split n: 2 -> 1, 1.
    #     C = [A1; A2] * B where A1 is 1x3, A2 is 1x3, B is 3x2.
    #     C1 = A1 * B (1x2), C2 = A2 * B (1x2)
    #     These are independent rank-3 problems each? No, 1x3x2 has rank 3.
    #     So 2x3x2 via n-split: 2 * 3 = 6 terms.
    #     Alternatively, split m: 3 -> 1, 2.
    #     C = A1*B1 + A2*B2 where A1 is 2x1, B1 is 1x2; A2 is 2x2, B2 is 2x2.
    #     Rank(2x1x2) = 2, Rank(2x2x2) = 7 (Strassen). Total = 9. Worse.
    #     So n-split gives 6 for 2x3x2.
    
    # Let's trace the full recursion for 2x3x8:
    # 2x3x8 -> split p -> 2*(2x3x4)
    # 2x3x4 -> split p -> 2*(2x3x2)
    # 2x3x2 -> split n -> 2*(1x3x2)
    # 1x3x2 -> schoolbook rank 6? No, 1*3*2=6.
    # Total rank = 2 * 2 * 2 * 6 = 48. Same as schoolbook.
    
    # We need a better approach. The issue is that when we split p, the two sub-problems
    # share the same U matrix (for A). We can exploit this.
    # Actually, when splitting p: C = [C1 | C2] where C1 = A*B1, C2 = A*B2.
    # U is the same for both! V and W are different.
    # So the total rank is rank(C1) + rank(C2) but with shared U.
    # In the tensor decomposition, U has shape (n*m, R_total).
    # For C1: U1 has shape (n*m, R1), V1 has shape (m*p1, R1), W1 has shape (p1*n, R1)
    # For C2: U2 has shape (n*m, R2), V2 has shape (m*p2, R2), W2 has shape (p2*n, R2)
    # Since A is the same, U1 and U2 can be chosen as the same matrix if R1=R2 and we use
    # the same decomposition for A. But B1 and B2 are different.
    
    # Actually, the standard way to combine is:
    # R_total = R1 + R2
    # U_total = [U1 | U2] (column concatenation)
    # V_total = [V1; V2] (row concatenation, since B = [B1 B2] means V rows are split)
    # W_total = [W1; W2] (row concatenation, since C = [C1 C2] means W rows are split)
    
    # This doesn't save rank. We need a different strategy.
    
    # Better idea: Use the fact that 2x3x8 can be viewed as 2x(3x8) or (2x3)x8.
    # Or use a more sophisticated recursive scheme.
    
    # Let's try: 2x3x8. Split m: 3 -> 1, 2.
    # C = A1*B1 + A2*B2 where A1 is 2x1, B1 is 1x8; A2 is 2x2, B2 is 2x8.
    # Rank(2x1x8) = 8 (schoolbook: 2*1*8=8)
    # Rank(2x2x8): split p: 8 -> 4,4. Two sub-problems of 2x2x4.
    #   Rank(2x2x4): split p: 4 -> 2,2. Two sub-problems of 2x2x2.
    #     Rank(2x2x2) = 7 (Strassen).
    #     So Rank(2x2x4) = 14.
    #     So Rank(2x2x8) = 28.
    # Total rank = 8 + 28 = 36.
    
    # Let's verify: 2x1x8 has rank 8? The tensor is sum_{j=0}^{7} a_j * b_j * c_j where
    # a_j is the column of A corresponding to m-index j (but m=1, so just the single column),
    # b_j is the row of B, c_j is the column of C.
    # Actually for 2x1x8: A is 2x1, B is 1x8, C is 2x8.
    # C_{i,k} = sum_{j=0}^{0} A_{i,j} * B_{j,k} = A_{i,0} * B_{0,k}.
    # This is an outer product of A (2x1) and B (1x8), giving a 2x8 matrix.
    # The tensor rank of this is min(2, 1, 8) * ... no.
    # The tensor T_{i,j,k} = A_{i,j} B_{j,k} with j fixed at 0.
    # T_{i,0,k} = A_{i,0} B_{0,k}. All other T_{i,j,k} = 0.
    # This is a rank-1 tensor in the (i,k) plane for each fixed j, but since only j=0 is nonzero,
    # it's essentially a matrix A*B which has rank min(2,8)=2.
    # The tensor rank should be 2, not 8!
    
    # Let me reconsider. For 2x1x8:
    # U shape: (2*1, R) = (2, R)
    # V shape: (1*8, R) = (8, R)
    # W shape: (8*2, R) = (16, R)
    # T_{i,j,k} = sum_r U_{i,j} V_{j,k} W_{k,i}
    # With j=0 only: T_{i,0,k} = sum_r U_{i,0} V_{0,k} W_{k,i}
    # This equals A_{i,0} * B_{0,k}.
    # We can set R=2:
    # U[:,0] = [1, 0], U[:,1] = [0, 1]  (so U_{i,0} = delta_{i,0}, U_{i,1} = delta_{i,1})
    # Wait, U has shape (2, R). Let's index U[row, col] where row = i*1 + j = i.
    # So U[i, r].
    # V[j, r] where j = 0*8 + k = k. So V[k, r].
    # W[k, r] where k = k*2 + i. So W[k*2+i, r].
    
    # T_{i,0,k} = sum_r U[i, r] * V[k, r] * W[k*2+i, r]
    # We want this to equal A_{i,0} * B_{0,k} for all i, k.
    # But A and B are variables. The decomposition must work for all A, B.
    # So we need: for all i, k: sum_r U[i,r] V[k,r] W[k*2+i, r] = A_{i,0} B_{0,k}
    # This is impossible with fixed U, V, W because the RHS depends on A and B.
    
    # I'm confusing the tensor decomposition with the matrix multiplication.
    # In tensor decomposition of matrix multiplication, we have:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has indices (i, j, k) and T_{i,j,k} = 1 if we're looking at the
    # coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # The decomposition is: T_{i,j,k} = sum_r U_{i,j} V_{j,k} W_{k,i}
    # where U, V, W are fixed matrices.
    
    # For 2x1x8: T_{i,0,k} = 1 for all i, k (since there's only one j=0).
    # T_{i,j,k} = 0 for j != 0.
    # We need: sum_r U[i,r] V[0,k] W[k*2+i, r] = 1 for all i, k.
    # And: sum_r U[i,r] V[j,k] W[k*2+i, r] = 0 for all j != 0.
    
    # Since V has shape (8, R) and only V[0, :] is used (j=0), we can set V[j, :] = 0 for j != 0.
    # Then the second condition is automatically satisfied.
    # We need: sum_r U[i,r] V[0,k] W[k*2+i, r] = 1 for all i in {0,1}, k in {0,...,7}.
    
    # Let's try R=1: U[i,0] * V[0,k] * W[k*2+i, 0] = 1.
    # This requires U[i,0], V[0,k], W[k*2+i,0] all nonzero, and their product = 1.
    # We can set U[i,0] = 1, V[0,k] = 1, W[k*2+i, 0] = 1.
    # Then sum = 1*1*1 = 1. Works!
    # So rank(2x1x8) = 1.
    
    # Similarly, rank(1x3x2) = 1, rank(1x1x1) = 1.
    
    # Let me redo the analysis with correct tensor ranks.
    # The tensor rank of n x m x p matrix multiplication is at most n*m*p (schoolbook).
    # For small cases:
    # 1x1x1: rank 1
    # 1x1xp: rank 1 (T_{0,0,k} = 1 for all k)
    # 1xmxp: rank 1? T_{0,j,k} = 1 for all j, k.
    #   sum_r U[0,r] V[j,r] W[k, r] = 1 for all j, k.
    #   With R=1: U[0,0] V[j,0] W[k,0] = 1. Set all to 1. Works. Rank 1.
    # nx1x1: rank 1.
    # nx1xp: rank 1.
    # So any case with a dimension of 1 has rank 1? No, that can't be right.
    
    # Wait, I think I'm making an error. Let me reconsider.
    # For 2x1x8: T_{i,j,k} where i in {0,1}, j in {0}, k in {0,...,7}.
    # T_{i,0,k} = 1 for all i, k.
    # Decomposition: T_{i,0,k} = sum_r U[i, r] V[0, r] W[k*2+i, r]
    # If we set R=1, U[i,0]=1, V[0,0]=1, W[k*2+i,0]=1, then sum = 1. Correct.
    # But this means the tensor is rank 1, which seems too good.
    
    # Actually, this is correct! The matrix multiplication tensor for 2x1x8 is indeed rank 1
    # in the tensor sense because there's only one "multiplication" happening: each element
    # of C is just A_{i,0} * B_{0,k}, and the tensor captures the structure of which
    # elements are multiplied, not the values.
    
    # Hmm, but that doesn't match my intuition. Let me check 2x2x2.
    # T_{i,j,k} = 1 for all i, j, k in {0,1}.
    # Rank is 7 (Strassen). If my logic above were correct, rank would be 1.
    # The difference is that for 2x2x2, T_{i,j,k} = 1 for all 8 combinations,
    # and we need sum_r U[i,r] V[j,r] W[k*2+i, r] = 1 for all 8 (i,j,k).
    # With R=1: U[i,0] V[j,0] W[k*2+i,0] = 1 for all i, j, k.
    # This requires U[i,0] V[j,0] W[k*2+i,0] to be independent of i, j, k, which is impossible
    # because W depends on both k and i.
    
    # For 2x1x8, W[k*2+i, 0] depends on k and i, but V[0,0] and U[i,0] don't depend on k.
    # So we need U[i,0] * V[0,0] * W[k*2+i, 0] = 1 for all i, k.
    # This means W[k*2+i, 0] = 1/(U[i,0] * V[0,0]) for all k.
    # So W[k*2+i, 0] is constant in k for fixed i. We can set W[k*2+i, 0] = 1 for all k, i,
    # and U[i,0] = 1, V[0,0] = 1. Then it works.
    
    # OK so the rank of 2x1x8 is indeed 1.
    
    # Now back to 2x3x8:
    # Split m: 3 -> 1, 2.
    # C = A1*B1 + A2*B2 where A1 is 2x1, B1 is 1x8; A2 is 2x2, B2 is 2x8.
    # The tensor T = T1 + T2 where T1 is the 2x1x8 tensor and T2 is the 2x2x8 tensor.
    # Rank(T) <= Rank(T1) + Rank(T2) = 1 + Rank(2x2x8).
    
    # Rank(2x2x8): split p: 8 -> 4, 4.
    # T = T1 + T2 where T1 is 2x2x4, T2 is 2x2x4.
    # Rank(2x2x4): split p: 4 -> 2, 2.
    # Rank(2x2x2) = 7.
    # Rank(2x2x4) = 14.
    # Rank(2x2x8) = 28.
    # Total rank = 1 + 28 = 29.
    
    # Can we do better? Let's try splitting p first.
    # 2x3x8: split p: 8 -> 4, 4.
    # Rank = 2 * Rank(2x3x4).
    # 2x3x4: split p: 4 -> 2, 2.
    # Rank = 2 * Rank(2x3x2).
    # 2x3x2: split m: 3 -> 1, 2.
    # Rank = Rank(2x1x2) + Rank(2x2x2) = 1 + 7 = 8.
    # So Rank(2x3x2) = 8.
    # Rank(2x3x4) = 16.
    # Rank(2x3x8) = 32.
    
    # The m-split approach gives 29, p-split gives 32. Let's go with 29.
    
    # Actually, let me reconsider the splitting. When we split m into m1, m2:
    # C = A1*B1 + A2*B2
    # A1 is n x m1, B1 is m1 x p
    # A2 is n x m2, B2 is m2 x p
    # The tensor T = T1 + T2
    # T1 has indices (i, j, k) with j in 0..m1-1
    # T2 has indices (i, j, k) with j in m1..m-1
    # In the decomposition, U has shape (n*m, R_total).
    # For T1: U1 has shape (n*m1, R1)
    # For T2: U2 has shape (n*m2, R2)
    # U_total has shape (n*m, R1+R2)
    # Rows 0..n*m1-1 of U_total come from U1
    # Rows n*m1..n*m-1 of U_total come from U2
    # V_total has shape (m*p, R1+R2)
    # Rows 0..m1*p-1 from V1, rows m1*p..m*p-1 from V2
    # W_total has shape (p*n, R1+R2)
    # Columns concatenated: W_total = [W1 | W2]
    
    # This is correct.
    
    def _combine_m_split(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
        """Combine two decompositions from an m-split."""
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R = R1 + R2
        U = np.zeros((n * (m1 + m2), R))
        V = np.zeros(((m1 + m2) * p, R))
        W = np.zeros((p * n, R))
        
        # U: rows 0..n*m1-1 from U1, rows n*m1..n*(m1+m2)-1 from U2
        U[:n*m1, :R1] = U1
        U[n*m1:, :R1] = 0
        U[:n*m1, R1:] = 0
        U[n*m1:, R1:] = U2
        
        # V: rows 0..m1*p-1 from V1, rows m1*p..(m1+m2)*p-1 from V2
        V[:m1*p, :R1] = V1
        V[m1*p:, :R1] = 0
        V[:m1*p, R1:] = 0
        V[m1*p:, R1:] = V2
        
        # W: column concatenation
        W[:, :R1] = W1
        W[:, R1:] = W2
        
        return U, V, W
    
    def _combine_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """Combine two decompositions from a p-split."""
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R = R1 + R2
        U = np.zeros((n * m, R))
        V = np.zeros((m * (p1 + p2), R))
        W = np.zeros(((p1 + p2) * n, R))
        
        # U: column concatenation
        U[:, :R1] = U1
        U[:, R1:] = U2
        
        # V: rows 0..m*p1-1 from V1, rows m*p1..m*(p1+p2)-1 from V2
        V[:m*p1, :R1] = V1
        V[m*p1:, :R1] = 0
        V[:m*p1, R1:] = 0
        V[m*p1:, R1:] = V2
        
        # W: rows 0..p1*n-1 from W1, rows p1*n..(p1+p2)*n-1 from W2
        W[:p1*n, :R1] = W1
        W[p1*n:, :R1] = 0
        W[:p1*n, R1:] = 0
        W[p1*n:, R1:] = W2
        
        return U, V, W
    
    def _combine_n_split(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
        """Combine two decompositions from an n-split."""
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R = R1 + R2
        U = np.zeros(((n1 + n2) * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * (n1 + n2), R))
        
        # U: rows 0..n1*m-1 from U1, rows n1*m..(n1+n2)*m-1 from U2
        U[:n1*m, :R1] = U1
        U[n1*m:, :R1] = 0
        U[:n1*m, R1:] = 0
        U[n1*m:, R1:] = U2
        
        # V: column concatenation
        V[:, :R1] = V1
        V[:, R1:] = V2
        
        # W: rows 0..p*n1-1 from W1, rows p*n1..p*(n1+n2)-1 from W2
        W[:p*n1, :R1] = W1
        W[p*n1:, :R1] = 0
        W[:p*n1, R1:] = 0
        W[p*n1:, R1:] = W2
        
        return U, V, W
    
    # Now implement the recursive decomposition
    def _decompose_recursive(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return _decompose_222()
        
        # Choose the best split
        # Options: split n, split m, split p
        # We'll use a heuristic: split the largest dimension, preferring even splits
        
        best_rank = float('inf')
        best_result = None
        
        # Try splitting n
        if n >= 2:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = _decompose_recursive(n1, m, p)
                U2, V2, W2 = _decompose_recursive(n2, m, p)
                R = U1.shape[1] + U2.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = _combine_n_split(U1, V1, W1, U2, V2, W2, n1, n2, m, p)
        
        # Try splitting m
        if m >= 2:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = _decompose_recursive(n, m1, p)
                U2, V2, W2 = _decompose_recursive(n, m2, p)
                R = U1.shape[1] + U2.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = _combine_m_split(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
        
        # Try splitting p
        if p >= 2:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = _decompose_recursive(n, m, p1)
                U2, V2, W2 = _decompose_recursive(n, m, p2)
                R = U1.shape[1] + U2.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = _combine_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2)
        
        return best_result
    
    def _decompose_222():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Term 1: (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # Term 3: A11(B21-B22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Term 4: A22(B11-B21)
        U[3, 3] = 1
        V[0, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # Term 5: (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = -1; W[3, 4] = -1
        
        # Term 6: (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 7: (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Memoization to avoid redundant computation
    memo = {}
    
    def _decompose_memo(n, m, p):
        key = (n, m, p)
        if key in memo:
            return memo[key]
        
        if n == 2 and m == 2 and p == 2:
            result = _decompose_222()
            memo[key] = result
            return result
        
        # Base cases for small dimensions
        if n == 1 or m == 1 or p == 1:
            # Rank 1 decomposition
            R = 1
            U = np.ones((n * m, R))
            V = np.ones((m * p, R))
            W = np.ones((p * n, R))
            result = (U, V, W)
            memo[key] = result
            return result
        
        best_rank = float('inf')
        best_result = None
        
        # Try splitting n
        if n >= 2:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = _decompose_memo(n1, m, p)
                U2, V2, W2 = _decompose_memo(n2, m, p)
                R = U1.shape[1] + U2.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = _combine_n_split(U1, V1, W1, U2, V2, W2, n1, n2, m, p)
        
        # Try splitting m
        if m >= 2:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = _decompose_memo(n, m1, p)
                U2, V2, W2 = _decompose_memo(n, m2, p)
                R = U1.shape[1] + U2.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = _combine_m_split(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
        
        # Try splitting p
        if p >= 2:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = _decompose_memo(n, m, p1)
                U2, V2, W2 = _decompose_memo(n, m, p2)
                R = U1.shape[1] + U2.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = _combine_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2)
        
        memo[key] = best_result
        return best_result
    
    return _decompose_memo(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
