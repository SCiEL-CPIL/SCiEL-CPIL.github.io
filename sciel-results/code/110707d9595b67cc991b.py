import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    
    start_time = time.time()
    deadline = 140  # seconds, leave margin

    # Schoolbook decomposition as fallback (Rank 18)
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((nm, rank_schoolbook))
    V_schoolbook = np.zeros((mp, rank_schoolbook))
    W_schoolbook = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1

    def to_lists(mat):
        return mat.tolist()

    # Target tensor T
    # T is a 3D array of shape (nm, mp, pn)
    # Indices: a = i*m + j, b = j*p + k, c = k*n + i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def decompose_to_tensor(U, V, W):
        """
        U: (nm, r)
        V: (mp, r)
        W: (pn, r)
        Returns T: (nm, mp, pn)
        T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        """
        r = U.shape[1]
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return T

    def verify_decomposition(U, V, W):
        """
        Checks if the decomposition is exact and entries are integers or half-integers.
        """
        # Check integer/half-integer constraint
        # Multiply by 2, check if close to integer
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        if not np.allclose(U2, np.round(U2), atol=1e-9):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-9):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-9):
            return False
            
        # Check tensor equality
        T_approx = decompose_to_tensor(U, V, W)
        if np.allclose(T_approx, T_target, atol=1e-9):
            return True
        return False

    # Strategy: Construct a rank-17 decomposition using Strassen for the 2x2 part
    # The problem is (2x3) * (3x3) -> (2x3)
    # Let A be 2x3, B be 3x3.
    # We can view A as [A1 A2] where A1, A2 are 2x3? No, A is 2x3.
    # Let's split the columns of A and rows of B.
    # A = [A_col0, A_col1, A_col2] (each 2x1)
    # B = [B_row0; B_row1; B_row2] (each 1x3)
    # This is not a standard block multiplication that reduces rank easily without Strassen on sub-blocks.
    
    # Alternative View:
    # Multiply (2x3) by (3x3).
    # Split the 3x3 matrix B into a 2x2 block and a 1x1 block?
    # Let's try to decompose using the known rank 17 algorithm for 2x3x3.
    # One known construction uses Strassen for a 2x2 sub-problem.
    
    # Let's define the specific rank-17 decomposition found in literature for 2x3x3.
    # Reference: "The rank of the 2x3x3 matrix multiplication tensor is 17"
    # We can construct it by combining:
    # 1. A 2x2x2 Strassen decomposition (rank 7) for the first 2 columns of A and first 2 rows of B?
    #    If we take A[:, :2] (2x2) and B[:2, :] (2x3), the product is 2x3.
    #    This requires decomposing (2x2) * (2x3).
    #    Rank of (2x2) * (2x3) is 10? No, schoolbook is 8. Strassen doesn't directly apply to rectangular without padding.
    
    # Let's look at the "Winograd" or "Strassen-like" scheme for 2x3x3.
    # Actually, a simpler approach:
    # We know rank(2x2x2) = 7.
    # We know rank(2x1x3) = 3 (schoolbook, diagonal).
    # We know rank(2x3x1) = 3 (schoolbook, diagonal).
    
    # Let's try to build it by splitting the middle dimension m=3 into 2+1.
    # A = [A_2x2 | A_2x1]
    # B = [B_2x3; B_1x3]
    # Product C = A_2x2 * B_2x3 + A_2x1 * B_1x3
    # C is 2x3.
    
    # Term 1: A_2x2 * B_2x3.
    # This is (2x2) * (2x3).
    # We can decompose this using Strassen on the 2x2 part of B?
    # Let B_2x3 = [B_2x2 | B_2x1].
    # A_2x2 * B_2x2 (rank 7 via Strassen) + A_2x2 * B_2x1 (rank ?).
    # A_2x2 * B_2x1 is (2x2)*(2x1). This is equivalent to 2 independent (1x2)*(2x1)? No.
    # (2x2)*(2x1) is a 2x1 vector.
    # It can be done in rank 4? Schoolbook is 4. Can it be done in less?
    # Rank of (2x2)x(2x1) is 4.
    # So Term 1 rank = 7 + 4 = 11?
    
    # Term 2: A_2x1 * B_1x3.
    # This is (2x1) * (1x3) = 2x3.
    # This is an outer product of a 2x1 vector and a 1x3 vector.
    # Rank is 1? No, the tensor is T[i,k] = A[i] * B[k].
    # This is a rank 1 tensor in the output space?
    # Wait, the multiplication tensor T_{i,k,j} = A_{i,j} B_{j,k}.
    # For A_2x1 (indices i, j=0..1) and B_1x3 (indices j=0..1? No, j is the shared index).
    # A is 2x1, so j index is 0..0? No, A is 2x1, so columns=1. B is 1x3, so rows=1.
    # Shared dimension is 1.
    # So A_{i,0} * B_{0,k}.
    # This is a rank 1 term in the tensor decomposition?
    # U_{i,0, r} V_{0,k, r} W_{i,k, r}.
    # If we set r=1, U_{i,0,1}=A_{i,0}, V_{0,k,1}=B_{0,k}, W_{i,k,1}=1.
    # But U, V, W must be constant matrices (linear combinations of inputs).
    # In tensor decomposition, U, V, W are the coefficients.
    # For the term A_{i,0} * B_{0,k}, we need:
    # Sum_r U[i,0,r] V[0,k,r] W[i,k,r] = A[i,0] B[0,k]
    # This is satisfied by 1 term:
    # U[i,0,1] = 1 (if i is the row index of A? No, U is indexed by input index of A).
    # Let's map indices carefully.
    # A indices: (i, j) with i in 0..1, j in 0..0 (for the 2x1 part).
    # B indices: (j, k) with j in 0..0, k in 0..2.
    # C indices: (i, k).
    # T[i, j, k, l, m, n]... no, T[a, b, c] where a=(i,j), b=(j,k), c=(i,k).
    # For the 2x1 * 1x3 part:
    # a = (i, 0), b = (0, k), c = (i, k).
    # We need U[a, r] V[b, r] W[c, r] to sum to 1 for these specific indices.
    # Let r=1.
    # U[(i,0), 1] = 1 for all i? No, U is a fixed matrix.
    # The value of the term in the product is A[i,0] * B[0,k].
    # The tensor entry is 1.
    # So we need Sum_r U[i,0,r] V[0,k,r] W[i,k,r] = 1.
    # If we choose U[i,0,1]=1, V[0,k,1]=1, W[i,k,1]=1, then the sum is 1.
    # This works for all i, k.
    # So this part contributes rank 1.
    
    # Total rank so far: 11 (for 2x2*2x3) + 1 (for 2x1*1x3) = 12?
    # Wait, is rank(2x2*2x3) really 11?
    # Schoolbook for 2x2*2x3 is 2*2*3 = 12.
    # Can we do better?
    # Decompose 2x2*2x3 as (2x2)*(2x2) + (2x2)*(2x1).
    # Rank(2x2*2x2) = 7 (Strassen).
    # Rank(2x2*2x1) = 4 (Schoolbook).
    # Total = 11.
    # So total rank = 11 + 1 = 12?
    # This would be a massive improvement over 17!
    # Is this valid?
    # Let's check the dimensions.
    # A is 2x3. Split into A_L (2x2) and A_R (2x1).
    # B is 3x3. Split into B_T (2x3) and B_B (1x3).
    # C = A_L B_T + A_R B_B.
    # A_L B_T is (2x2)(2x3) -> 2x3.
    # A_R B_B is (2x1)(1x3) -> 2x3.
    # Yes, the decomposition is valid.
    # Why is the known rank 17?
    # Maybe Rank(2x2*2x1) is not 4?
    # (2x2) * (2x1).
    # Input A (2x2), B (2x1). Output C (2x1).
    # C[i] = sum_j A[i,j] B[j,0].
    # This is 4 multiplications.
    # Can we do it in 3?
    # Tensor T[i, j, k] where i is row of C (0..1), j is col of A (0..1) AND row of B (0..1), k is col of B (0..0).
    # Wait, B is 2x1, so k is fixed at 0.
    # T indices: a=(i,j), b=(j,k), c=(i,k).
    # k is always 0.
    # So we only care about T[i, j, 0, j, 0, i] = 1.
    # This is a tensor of shape (2*2, 2*1, 2*1) = (4, 2, 2).
    # Non-zero entries:
    # i=0, j=0: a=0, b=0, c=0.
    # i=0, j=1: a=1, b=2, c=0.
    # i=1, j=0: a=2, b=0, c=2.
    # i=1, j=1: a=3, b=2, c=2.
    # Let's check the rank of this tensor.
    # It looks like the identity matrix tensor?
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    # Here m=2, p=1, n=2.
    # a=i*2+j, b=j*1+0=j, c=0*2+i=i.
    # (0,0,0): i=0,j=0. a=0, b=0, c=0.
    # (1,2,0): i=0,j=1. a=1, b=1, c=0. Wait. b=j*p+k = 1*1+0=1.
    # My previous index mapping:
    # b index is j*p + k.
    # For B (2x1), p=1. k=0. So b=j.
    # So b=0 for j=0, b=1 for j=1.
    # Let's re-evaluate non-zeros for (2x2)*(2x1):
    # i=0, j=0: a=0, b=0, c=0.
    # i=0, j=1: a=1, b=1, c=0.
    # i=1, j=0: a=2, b=0, c=2.
    # i=1, j=1: a=3, b=1, c=2.
    # Tensor T[4,2,2].
    # T[0,0,0]=1
    # T[1,1,0]=1
    # T[2,0,2]=1
    # T[3,1,2]=1
    # This tensor is block diagonal?
    # Slice c=0:
    # [1, 0]
    # [0, 1]
    # [0, 0]
    # [0, 0]
    # Slice c=2:
    # [0, 0]
    # [0, 0]
    # [1, 0]
    # [0, 1]
    # This is the identity tensor on 4 dimensions?
    # Rank of identity tensor of size N is N.
    # So Rank(2x2*2x1) = 4.
    # So my calculation of 12 rank holds?
    # Why is the literature saying 17?
    # Perhaps Rank(2x2*2x2) is not 7 when embedded?
    # Or perhaps the splitting introduces cross terms that are not independent?
    # No, the sum of ranks is an upper bound.
    # If I can construct a rank 12 decomposition, I should return it.
    # Let's try to construct the rank 12 decomposition explicitly.
    
    # Components:
    # 1. Strassen for A_L (2x2) * B_TL (2x2). (Part of A_L * B_T)
    #    A_L = [[a00, a01], [a10, a11]]
    #    B_TL = [[b00, b01], [b10, b11]] (Top-Left 2x2 of B)
    #    Product P_TL = A_L * B_TL.
    #    Strassen uses 7 multiplications.
    #    U, V, W matrices for this part will be non-zero only for indices corresponding to A_L and B_TL.
    
    # 2. Schoolbook for A_L (2x2) * B_TR (2x1). (Part of A_L * B_T)
    #    B_TR = [[b02], [b12]] (Top-Right 2x1 of B)
    #    Product P_TR = A_L * B_TR. (2x1 vector)
    #    Rank 4.
    
    # 3. Schoolbook for A_R (2x1) * B_B (1x3).
    #    A_R = [[a02], [a12]]
    #    B_B = [[b20, b21, b22]]
    #    Product P_R = A_R * B_B. (2x3 matrix)
    #    Rank 1.
    
    # Total Rank = 7 + 4 + 1 = 12.
    
    # Let's verify if this is correct.
    # If rank 12 is possible, it would be a huge discovery (or I am missing a constraint).
    # Wait, the "Idea target" mentions rank 17.
    # Is it possible that Rank(2x2*2x3) is NOT 11?
    # Rank(2x2*2x3) >= Rank(2x2*2x2) = 7.
    # Rank(2x2*2x3) >= Rank(2x1*2x3) = 6?
    # Let's check Rank(2x2*2x1) again.
    # If Rank(2x2*2x1) = 4, then Rank(2x2*2x3) <= 7 + 4 = 11.
    # Is Rank(2x2*2x1) actually 4?
    # Yes, the tensor is the identity tensor of size 4, which has rank 4.
    # So Rank(2x2*2x3) <= 11.
    # Then Rank(2x3*3x3) <= 11 + 1 = 12.
    
    # Why do sources say 17?
    # Maybe I am confusing "rank" with "multiplications"?
    # In tensor decomposition, rank is the number of multiplications.
    # If I can do it in 12, I should do it.
    # Let's double check the index mapping for the "Rank 1" term.
    # A_R is 2x1. Indices (i, j=2).
    # B_B is 1x3. Indices (j=2, k).
    # C is 2x3. Indices (i, k).
    # Term: A[i,2] * B[2,k].
    # Tensor T[i*m+2, 2*p+k, k*n+i].
    # m=3, p=3, n=2.
    # a = i*3+2.
    # b = 2*3+k = 6+k.
    # c = k*2+i.
    # We need U[a,r] V[b,r] W[c,r] = 1.
    # If we set r=1 for this term:
    # U[i*3+2, 1] = 1 for i=0,1.
    # V[6+k, 1] = 1 for k=0,1,2.
    # W[k*2+i, 1] = 1 for all i,k.
    # This works.
    
    # So, I will implement the Rank 12 construction.
    # If it fails verification, I will fall back to Rank 17 (Strassen 2x2 + rest) or 18.
    
    # Construction of Rank 12:
    # U, V, W of shape (6, 12), (9, 12), (6, 12).
    
    # Part 1: Strassen 2x2*2x2 (Rank 7)
    # Indices in A: i in 0..1, j in 0..1. a = i*3+j.
    # Indices in B: j in 0..1, k in 0..1. b = j*3+k.
    # Indices in C: i in 0..1, k in 0..1. c = k*2+i.
    # Strassen multipliers:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # We need to map these to U, V, W columns.
    # Let's assign columns 0-6 for this part.
    
    U1 = np.zeros((6, 7))
    V1 = np.zeros((9, 7))
    W1 = np.zeros((6, 7))
    
    # m1: (a00+a11) * (b00+b11)
    # U: a00 (idx 0) = 1, a11 (idx 4) = 1.
    # V: b00 (idx 0) = 1, b11 (idx 4) = 1.
    # W: c00 (idx 0) = 1, c11 (idx 3) = 1.
    # Note: C00 = m1..., C11 = m1...
    # So W1[0, 0] = 1, W1[3, 0] = 1.
    
    U1[0, 0] = 1; U1[4, 0] = 1
    V1[0, 0] = 1; V1[4, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # m2: (a10+a11) * b00
    # U: a10 (idx 3) = 1, a11 (idx 4) = 1.
    # V: b00 (idx 0) = 1.
    # W: c10 (idx 1) = 1, c11 (idx 3) = -1.
    U1[3, 1] = 1; U1[4, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    
    # m3: a00 * (b01-b11)
    # U: a00 (idx 0) = 1.
    # V: b01 (idx 1) = 1, b11 (idx 4) = -1.
    # W: c00 (idx 0) = 1, c01 (idx 2) = 1.
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[4, 2] = -1
    W1[0, 2] = 1; W1[2, 2] = 1
    
    # m4: a11 * (b10-b00)
    # U: a11 (idx 4) = 1.
    # V: b10 (idx 3) = 1, b00 (idx 0) = -1.
    # W: c00 (idx 0) = 1, c10 (idx 1) = 1.
    U1[4, 3] = 1
    V1[3, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1
    
    # m5: (a00+a01) * b11
    # U: a00 (idx 0) = 1, a01 (idx 1) = 1.
    # V: b11 (idx 4) = 1.
    # W: c00 (idx 0) = -1, c01 (idx 2) = 1.
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[4, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # m6: (a10-a00) * (b00+b01)
    # U: a10 (idx 3) = 1, a00 (idx 0) = -1.
    # V: b00 (idx 0) = 1, b01 (idx 1) = 1.
    # W: c11 (idx 3) = 1.
    U1[3, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # m7: (a01-a11) * (b10+b11)
    # U: a01 (idx 1) = 1, a11 (idx 4) = -1.
    # V: b10 (idx 3) = 1, b11 (idx 4) = 1.
    # W: c00 (idx 0) = 1, c01 (idx 2) = 1.
    U1[1, 6] = 1; U1[4, 6] = -1
    V1[3, 6] = 1; V1[4, 6] = 1
    W1[0, 6] = 1; W1[2, 6] = 1
    
    # Part 2: Schoolbook 2x2*2x1 (Rank 4)
    # A_L (2x2) * B_TR (2x1).
    # B_TR indices: j in 0..1, k=2. b = j*3+2.
    # A_L indices: i in 0..1, j in 0..1. a = i*3+j.
    # C indices: i in 0..1, k=2. c = 2*2+i = 4+i.
    # Multiplications:
    # C02 = a00*b02 + a01*b12
    # C12 = a10*b02 + a11*b12
    
    U2 = np.zeros((6, 4))
    V2 = np.zeros((9, 4))
    W2 = np.zeros((6, 4))
    
    # m8: a00 * b02
    U2[0, 0] = 1
    V2[2, 0] = 1  # b index 0*3+2=2
    W2[4, 0] = 1  # c index 2*2+0=4
    
    # m9: a01 * b12
    U2[1, 1] = 1
    V2[5, 1] = 1  # b index 1*3+2=5
    W2[4, 1] = 1
    
    # m10: a10 * b02
    U2[3, 2] = 1
    V2[2, 2] = 1
    W2[5, 2] = 1  # c index 2*2+1=5
    
    # m11: a11 * b12
    U2[4, 3] = 1
    V2[5, 3] = 1
    W2[5, 3] = 1
    
    # Part 3: Schoolbook 2x1*1x3 (Rank 1)
    # A_R (2x1) * B_B (1x3).
    # A_R indices: i in 0..1, j=2. a = i*3+2.
    # B_B indices: j=2, k in 0..2. b = 2*3+k = 6+k.
    # C indices: i in 0..1, k in 0..2. c = k*2+i.
    
    U3 = np.zeros((6, 1))
    V3 = np.zeros((9, 1))
    W3 = np.zeros((6, 1))
    
    # m12: (a02+a12? No, separate terms)
    # Wait, the term is A[i,2] * B[2,k].
    # This is a single rank-1 term in the tensor?
    # T[i*3+2, 6+k, k*2+i] = 1.
    # We need U[a,r] V[b,r] W[c,r] = 1.
    # If we set U[i*3+2, 0] = 1, V[6+k, 0] = 1, W[k*2+i, 0] = 1.
    # This works for all i,k.
    U3[2, 0] = 1
    U3[5, 0] = 1
    V3[6, 0] = 1
    V3[7, 0] = 1
    V3[8, 0] = 1
    W3[0, 0] = 1
    W3[1, 0] = 1
    W3[2, 0] = 1
    W3[3, 0] = 1
    W3[4, 0] = 1
    W3[5, 0] = 1
    
    # Combine
    U_total = np.hstack([U1, U2, U3])
    V_total = np.hstack([V1, V2, V3])
    W_total = np.hstack([W1, W2, W3])
    
    # Verify
    if verify_decomposition(U_total, V_total, W_total):
        return {"U": to_lists(U_total), "V": to_lists(V_total), "W": to_lists(W_total)}
    else:
        # Fallback to Schoolbook
        return {"U": to_lists(U_schoolbook), "V": to_lists(V_schoolbook), "W": to_lists(W_schoolbook)}
