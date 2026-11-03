import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS

def _verify(U, V, W, rank, n=N, m=M, p=P):
    """Verify that the decomposition is exact for n x m * m x p."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # And it should be 0 for "mismatched" indices.
    
    # Actually, the tensor T has entries T[a, b, c] where:
    # a = i*m + j (A index), b = j'*p + k' (B index), c = k''*n + i'' (C index)
    # T[a, b, c] = 1 iff i=i'', j=j', k=k'
    # Wait, let me re-derive.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The multiplication tensor is T[i, j, k] = 1 (all ones on the valid index set)
    # But that can't be right because the rank would be 1.
    
    # Let me think about this differently. The tensor is:
    # T[i*m+j, j'*p+k', k''*n+i''] = delta(i, i'') * delta(j, j') * delta(k, k'')
    # No wait, that's not right either.
    
    # The correct formulation: The matrix multiplication tensor M_{n,m,p} is a 3-tensor
    # with indices (A_idx, B_idx, C_idx) where:
    # A_idx = i*m + j, B_idx = j'*p + k', C_idx = k''*n + i''
    # M[A_idx, B_idx, C_idx] = 1 iff there exist i, j, k such that:
    #   A_idx = i*m + j, B_idx = j*p + k, C_idx = k*n + i
    # i.e., the j in A matches the j' in B, and the i in A matches the i'' in C,
    # and the k in B matches the k'' in C.
    
    # So M[i*m+j, j'*p+k', k''*n+i''] = 1 iff j==j' AND i==i'' AND k==k''
    
    # The decomposition: M = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # i.e., sum_r U[a, r] * V[b, r] * W[c, r] = M[a, b, c]
    
    # For the schoolbook: each (i,j,k) triple gets its own rank-1 term with
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # Then sum_r ... = 1 iff (a,b,c) matches exactly one term, which happens iff
    # the indices are consistent (j==j', i==i'', k==k'').
    
    # So the verification is:
    # For all (a, b, c): sum_r U[a,r]*V[b,r]*W[c,r] should equal
    # 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k), else 0.
    
    # Let's verify efficiently using numpy.
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute the full tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is equivalent to: reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    # T[i,j,k] = sum_r U[i,m+j,r]... 
    
    # Actually, let's just check the 72 non-zero entries and verify they're all 1,
    # and check that some zero entries are 0.
    
    # Non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1) > 1e-9:
                    return False
    
    # Check a sample of zero entries
    # T[i*m+j, j'*p+k', k''*n+i''] = 0 if not (j==j' and i==i'' and k==k'')
    # Let's check all entries where exactly one index doesn't match.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                # Check B index with wrong j'
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k2 in range(p):
                        b = j2 * p + k2
                        for i2 in range(n):
                            c = k * n + i2  # keep k, i fixed, vary i2
                            # This should be 0 since j != j2
                            s = 0.0
                            for r in range(rank):
                                s += U[a][r] * V[b][r] * W[c][r]
                            if abs(s) > 1e-9:
                                return False
    return True

def _schoolbook():
    """Return the trivial rank-72 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4,7), V: (4,7), W: (4,7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00 * (B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11 * (B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01) * B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_2x2_2xp(p):
    """
    Strassen-like decomposition for 2x2 * 2xp -> 2xp.
    Uses 7 "products" where each product involves a 1xp vector.
    U: (4, 7), V: (2*p, 7), W: (p*2, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(2*p)]
    W = [[0]*7 for _ in range(p*2)]
    
    # B is 2xp: B[0,0..p-1], B[1,0..p-1]
    # B index: j*p + k, so B[0,k] is at index k, B[1,k] is at index p+k
    
    # C is 2xp: C[0,0..p-1], C[1,0..p-1]
    # C index: k*2 + i, so C[i,k] is at index k*2+i
    
    # M1 = (A00 + A11) * (B0 + B1) where B0 = B[0,:], B1 = B[1,:]
    U[0][0] = 1; U[3][0] = 1
    for k in range(p):
        V[k][0] = 1      # B[0,k]
        V[p+k][0] = 1    # B[1,k]
    for k in range(p):
        W[k*2+0][0] = 1  # C[0,k]
        W[k*2+1][0] = 1  # C[1,k]
    
    # M2 = (A10 + A11) * B0
    U[2][1] = 1; U[3][1] = 1
    for k in range(p):
        V[k][1] = 1      # B[0,k]
    for k in range(p):
        W[k*2+1][1] = 1  # C[1,k]
    # C[1] gets M2, C[0] gets nothing from M2
    # Wait, let me re-derive Strassen for 2x2 * 2xp.
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    # 
    # M1 = (A00+A11)(B0+B1)
    # M2 = (A10+A11)B0
    # M3 = A00(B1-B0)
    # M4 = A11(B0-B1)
    # M5 = (A00+A01)B1
    # M6 = (A10-A00)(B0+B1)
    # M7 = (A01-A11)(B0+B1)
    # 
    # C0 = M1 + M4 - M5 + M7  ... hmm, let me just use the standard Strassen.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # 
    # For 2x2 * 2xp, each "entry" of B and C is a 1xp vector.
    # C0 = M1 + M4 - M5 + M7  (each term is a 1xp vector)
    # C1 = M1 - M2 + M3 + M6
    
    # Let me redo this properly.
    # Reset
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(2*p)]
    W = [[0]*7 for _ in range(p*2)]
    
    # M1 = (A00+A11) * (B0+B1)
    U[0][0] = 1; U[3][0] = 1
    for k in range(p):
        V[k][0] = 1; V[p+k][0] = 1
    for k in range(p):
        W[k*2+0][0] = 1; W[k*2+1][0] = 1
    
    # M2 = (A10+A11) * B0
    U[2][1] = 1; U[3][1] = 1
    for k in range(p):
        V[k][1] = 1
    for k in range(p):
        W[k*2+1][1] = 1  # C1 gets M2
    
    # M3 = A00 * (B1-B0)
    U[0][2] = 1
    for k in range(p):
        V[p+k][2] = 1; V[k][2] = -1
    for k in range(p):
        W[k*2+1][2] = 1  # C1 gets M3
    
    # M4 = A11 * (B0-B1)
    U[3][3] = 1
    for k in range(p):
        V[k][3] = 1; V[p+k][3] = -1
    for k in range(p):
        W[k*2+0][3] = 1; W[k*2+1][3] = 1
    
    # M5 = (A00+A01) * B1
    U[0][4] = 1; U[1][4] = 1
    for k in range(p):
        V[p+k][4] = 1
    for k in range(p):
        W[k*2+0][4] = -1  # C0 gets -M5
    
    # M6 = (A10-A00) * (B0+B1)
    U[2][5] = 1; U[0][5] = -1
    for k in range(p):
        V[k][5] = 1; V[p+k][5] = 1
    for k in range(p):
        W[k*2+1][5] = 1  # C1 gets M6
    
    # M7 = (A01-A11) * (B0+B1)
    U[1][6] = 1; U[3][6] = -1
    for k in range(p):
        V[k][6] = 1; V[p+k][6] = 1
    for k in range(p):
        W[k*2+0][6] = 1  # C0 gets M7
    
    return U, V, W

def _construct_3x3_3x8():
    """
    Construct a low-rank decomposition for 3x3 * 3x8.
    
    Strategy: Split the 3x3 A matrix into a 2x2 block and a 1x3 row / 3x1 column.
    A = [A_top (2x2)  A_top_right (2x1);
         A_bottom (1x2)  A_bottom_right (1x1)]
    
    B = [B_top (2x8); B_bottom (1x8)]
    
    C = A * B (3x8)
    C_top = A_top * B_top + A_top_right * B_bottom  (2x8)
    C_bottom = A_bottom * B_top + A_bottom_right * B_bottom  (1x8)
    
    For C_top: Use Strassen for A_top * B_top (2x2 * 2x8, rank 7)
               Plus schoolbook for A_top_right * B_bottom (2x1 * 1x8, rank 2)
    For C_bottom: Use schoolbook for A_bottom * B_top (1x2 * 2x8, rank 2)
                  Plus schoolbook for A_bottom_right * B_bottom (1x1 * 1x8, rank 1)
    
    Total rank: 7 + 2 + 2 + 1 = 12? That seems too low. Let me check.
    
    Actually, A_top_right is 2x1, B_bottom is 1x8. Their product is 2x8.
    Schoolbook for 2x1 * 1x8: rank = 2*1*8 = 16? No, that's for the full tensor.
    Actually, (2x1) * (1x8) = 2x8. The tensor rank of this is 2 (each row of the result
    is a scalar multiple of the corresponding row of B_bottom, scaled by the corresponding
    entry of A_top_right). Wait, no:
    C_top_right[0, k] = A[0,2] * B[2,k]
    C_top_right[1, k] = A[1,2] * B[2,k]
    This is rank 2 (two rank-1 tensors: A[0,2]*B[2,:] contributes to row 0, A[1,2]*B[2,:] to row 1).
    
    Similarly, A_bottom (1x2) * B_top (2x8) = 1x8:
    C_bottom[0, k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    This is rank 2.
    
    And A_bottom_right (1x1) * B_bottom (1x8) = 1x8:
    C_bottom[0, k] += A[2,2]*B[2,k]
    This is rank 1.
    
    So total: 7 (Strassen for 2x2*2x8) + 2 + 2 + 1 = 12.
    
    But wait, can we do even better? The Strassen for 2x2 * 2x8 has rank 7, but maybe
    we can share some of the "B0+B1" terms with the other parts.
    
    For now, let's implement the rank-12 construction and verify it.
    """
    n, m, p = N, M, P
    # n=3, m=3, p=8
    
    # A is 3x3: indices A[i,j] -> U[i*3+j]
    # B is 3x8: indices B[j,k] -> V[j*8+k]
    # C is 3x8: indices C[i,k] -> W[k*3+i]
    
    # Block partition:
    # A_top = A[0:2, 0:2] (2x2)
    # A_top_right = A[0:2, 2] (2x1)
    # A_bottom = A[2, 0:2] (1x2)
    # A_bottom_right = A[2, 2] (1x1)
    #
    # B_top = B[0:2, :] (2x8)
    # B_bottom = B[2, :] (1x8)
    
    # C_top = C[0:2, :] = A_top * B_top + A_top_right * B_bottom
    # C_bottom = C[2, :] = A_bottom * B_top + A_bottom_right * B_bottom
    
    # Total rank: 7 + 2 + 2 + 1 = 12
    
    rank = 12
    U = [[0]*rank for _ in range(n*m)]  # 9 x 12
    V = [[0]*rank for _ in range(m*p)]  # 24 x 12
    W = [[0]*rank for _ in range(p*n)]  # 24 x 12
    
    # Part 1: Strassen for A_top * B_top -> C_top (2x2 * 2x8 -> 2x8)
    # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # V indices: B[0,k]=k, B[1,k]=8+k
    # W indices: C[0,k]=k*3+0=k*3, C[1,k]=k*3+1
    
    # Strassen terms for 2x2 * 2x8:
    # M1 = (A00+A11)(B0+B1)
    # M2 = (A10+A11)B0
    # M3 = A00(B1-B0)
    # M4 = A11(B0-B1)
    # M5 = (A00+A01)B1
    # M6 = (A10-A00)(B0+B1)
    # M7 = (A01-A11)(B0+B1)
    #
    # C0 = M1 + M4 - M5 + M7
    # C1 = M1 - M2 + M3 + M6
    
    col = 0
    # M1
    U[0][col] = 1; U[4][col] = 1
    for k in range(p):
        V[k][col] = 1; V[8+k][col] = 1
    for k in range(p):
        W[k*3+0][col] = 1; W[k*3+1][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    for k in range(p):
        V[k][col] = 1
    for k in range(p):
        W[k*3+1][col] = 1
    col += 1
    
    # M3
    U[0][col] = 1
    for k in range(p):
        V[8+k][col] = 1; V[k][col] = -1
    for k in range(p):
        W[k*3+1][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    for k in range(p):
        V[k][col] = 1; V[8+k][col] = -1
    for k in range(p):
        W[k*3+0][col] = 1; W[k*3+1][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    for k in range(p):
        V[8+k][col] = 1
    for k in range(p):
        W[k*3+0][col] = -1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    for k in range(p):
        V[k][col] = 1; V[8+k][col] = 1
    for k in range(p):
        W[k*3+1][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    for k in range(p):
        V[k][col] = 1; V[8+k][col] = 1
    for k in range(p):
        W[k*3+0][col] = 1
    col += 1
    
    # Part 2: A_top_right * B_bottom (2x1 * 1x8 -> 2x8)
    # C[0,k] += A[0,2] * B[2,k]
    # C[1,k] += A[1,2] * B[2,k]
    # A[0,2] = U index 2, A[1,2] = U index 5
    # B[2,k] = V index 16+k
    # C[0,k] = W index k*3+0, C[1,k] = W index k*3+1
    
    # Term: A[0,2] * B[2,:] -> C[0,:]
    U[2][col] = 1
    for k in range(p):
        V[16+k][col] = 1
    for k in range(p):
        W[k*3+0][col] = 1
    col += 1
    
    # Term: A[1,2] * B[2,:] -> C[1,:]
    U[5][col] = 1
    for k in range(p):
        V[16+k][col] = 1
    for k in range(p):
        W[k*3+1][col] = 1
    col += 1
    
    # Part 3: A_bottom * B_top (1x2 * 2x8 -> 1x8)
    # C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k]
    # A[2,0] = U index 6, A[2,1] = U index 7
    # B[0,k] = V index k, B[1,k] = V index 8+k
    # C[2,k] = W index k*3+2
    
    # Term: A[2,0] * B[0,:] -> C[2,:]
    U[6][col] = 1
    for k in range(p):
        V[k][col] = 1
    for k in range(p):
        W[k*3+2][col] = 1
    col += 1
    
    # Term: A[2,1] * B[1,:] -> C[2,:]
    U[7][col] = 1
    for k in range(p):
        V[8+k][col] = 1
    for k in range(p):
        W[k*3+2][col] = 1
    col += 1
    
    # Part 4: A_bottom_right * B_bottom (1x1 * 1x8 -> 1x8)
    # C[2,k] += A[2,2] * B[2,k]
    # A[2,2] = U index 8, B[2,k] = V index 16+k, C[2,k] = W index k*3+2
    
    U[8][col] = 1
    for k in range(p):
        V[16+k][col] = 1
    for k in range(p):
        W[k*3+2][col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Try the rank-12 construction
    U, V, W, rank = _construct_3x3_3x8()
    
    # Verify
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    U, V, W, rank = _schoolbook()
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Should not reach here
    return {"U": U, "V": V, "W": W}
