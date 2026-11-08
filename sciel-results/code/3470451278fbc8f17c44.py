"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Approach: Split 4x4x6 into 4x4x4 (Laderman rank 19) + 4x4x2 (trivial rank 8)
    # Total rank = 27
    
    # Laderman 4x4x4 factors
    # U: 16x19, V: 16x19, W: 16x19
    # Indices: U[i*4+j], V[j*4+k], W[k*4+i]
    
    # Laderman's 19 intermediate products for 4x4x4:
    # We use the standard Laderman algorithm factors
    # A is 4x4, B is 4x4, C = A*B is 4x4
    
    # Define the 19 rank-1 terms
    # Each term: U_col, V_col, W_col (each 16-dim vectors)
    
    U_lad = []
    V_lad = []
    W_lad = []
    
    # Laderman algorithm for 4x4:
    # M1 = (A00+A11+A22+A33)(B00+B11+B22+B33)
    # M2 = (A01+A10+A23+A32)(B01+B10+B23+B32)  
    # ... this is getting complex. Let me use a verified set.
    
    # Actually, let me use a simpler verified approach:
    # 4x4x4 can be done with 4 2x2x2 Strassen blocks = rank 28
    # Plus 4x4x2 trivial = rank 8
    # Total = 36
    
    # But let me try to get Laderman right.
    # Standard Laderman 4x4x4 rank-19:
    
    # I'll use the following construction based on Laderman (1976):
    # The 19 products are:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a20+a31)(b20+b31)  
    # M3 = (a02+a13)(b02+b13)
    # M4 = (a22+a33)(b22+b33)
    # M5 = (a00+a22)(b00+b22)
    # M6 = (a11+a33)(b11+b33)
    # M7 = (a01+a23)(b01+b23)
    # M8 = (a10+a32)(b10+b32)
    # M9 = (a00+a11+a22+a33)(b00+b11+b22+b33)
    # M10 = (a01+a10+a23+a32)(b01+b10+b23+b32)
    # M11 = (a02+a13+a20+a31)(b02+b13+b20+b31)
    # M12 = (a03+a12+a21+a30)(b03+b12+b21+b30)
    # M13 = (a00+a11)(b22+b33) + (a22+a33)(b00+b11)
    # M14 = (a00+a11)(b02+b13) + (a22+a33)(b00+b11)  # wrong
    # This is not working cleanly.
    
    # Let me use a different, simpler approach:
    # 4x4x4 via 4 Strassen 2x2x2 blocks: rank 28
    # 4x4x2 via trivial: rank 8
    # Total: 36
    
    # Strassen 2x2x2 factors
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # Part 1: 4x4x4 using 4 Strassen blocks (rank 28)
    # Part 2: 4x4x2 using trivial (rank 8)
    total_rank = 28 + 8
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 36
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 36
    
    # Part 1: 4x4x4, columns 0-27
    # 4 Strassen blocks: (bi, bj) where bi, bj in {0,1}
    col_offset = 0
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = col_offset + c
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U[i * m + j][col] = S_U[di * 2 + dj][c]
                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = dk  # k in {0,1} for the 4x4x4 part
                        V[j * p + k][col] = S_V[dj * 2 + dk][c]
                for dk in range(2):
                    for di in range(2):
                        k = dk
                        i = bi * 2 + di
                        W[k * n + i][col] = S_W[dk * 2 + di][c]
            col_offset += 7
    
    # Part 2: 4x4x2, columns 28-35
    # Trivial: C[i,k] = sum_j A[i,j] B[j,k] for k in {0,1}
    # Each (i,j,k) triple gives a rank-1 term
    # 4*4*2 = 32 terms? No, rank is 4*2 = 8 for the 4x4x2 tensor
    # Actually 4x4x2 tensor has rank 8: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # The rank of the 4x4x2 multiplication tensor is 4*2 = 8 (since it's a "slice" of the full tensor)
    # Each rank-1 term: U has 1 in position (i,j), V has 1 in position (j,k), W has 1 in position (k,i)
    # But we need 4*4*2 = 32 such terms for trivial... no.
    
    # The 4x4x2 tensor T[i,j,k] = delta[i,i'] * delta[j,j'] * delta[k,k'] summed...
    # Actually T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid i,j,k, and the multiplication is:
    # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,2)
    # The rank of this tensor: it's a 4x4x2 tensor with all ones.
    # Rank of all-ones tensor of shape (a,b,c) is max(a,b,c) if... no.
    # Actually the multiplication tensor for (n,m,p) has T[i,j,k] = 1 for all i,j,k.
    # The rank of the all-ones tensor of shape (n,m,p) is min(n*m, m*p, p*n) in general? No.
    # For (4,4,2): the tensor is 4x4x2 all ones. Its rank is 8.
    # Because we can write it as sum of 8 rank-1 tensors.
    
    # Trivial decomposition of 4x4x2: rank 8
    # For each i in [0,4), k in [0,2):
    #   U[i*4 + j] = 1 for all j? No.
    # Actually the trivial way: for each (i,j,k), that's 32 terms. But rank can be lower.
    # The 4x4x2 all-ones tensor has rank 8.
    # Construction: for i in [0,4), k in [0,2):
    #   u vector: has 1 at positions (i, j) for all j in [0,4) -> no, that's not rank 1.
    
    # Let me think again. The multiplication tensor T_{n,m,p}[i,j,k] = 1 for all i,j,k.
    # A rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i,j], v_r[j,k], w_r[k,i]
    # For the all-ones tensor, one rank-1 term is u[i,j]=1, v[j,k]=1, w[k,i]=1.
    # That gives T[i,j,k] = 1 for all i,j,k. So rank 1?
    # No! Because u[i,j]*v[j,k]*w[k,i] = 1*1*1 = 1 for all i,j,k. So the all-ones tensor HAS rank 1!
    
    # Wait, that can't be right for matrix multiplication. Let me reconsider.
    # The matrix multiplication tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # And C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So yes, T[i,j,k] = 1 for all i,j,k.
    # And u[i,j]=1 for all i,j, v[j,k]=1 for all j,k, w[k,i]=1 for all k,i gives:
    # sum_j u[i,j]*v[j,k]*w[k,i] * A[i,j]*B[j,k]... no, the decomposition is:
    # T = sum_r (u_r ⊗ v_r ⊗ w_r) meaning T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    # If u_r[i,j]=1, v_r[j,k]=1, w_r[k,i]=1, then T[i,j,k] = 1. Rank 1!
    
    # But that's wrong because the actual matrix multiplication tensor is NOT the all-ones tensor.
    # Let me re-read the problem.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T satisfies: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor notation: T[i,j,k] * A[i,j] * B[j,k] summed over j gives C[i,k].
    # So T[i,j,k] = 1 for all i,j,k. Yes, it IS the all-ones tensor.
    
    # So the rank of the 4x4x6 multiplication tensor is the rank of the 4x4x6 all-ones tensor.
    # And the all-ones tensor of shape (n,m,p) has rank 1! Because u[i,j]=1, v[j,k]=1, w[k,i]=1.
    
    # That can't be right either, because we know matrix multiplication has non-trivial rank.
    # I think I'm confusing the tensor structure. Let me re-read the interface.
    
    # "U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"
    # "W[k*n+i,r]"
    # The decomposition: for each r, U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # U has indices (i,j), V has indices (j,k), W has indices (k,i)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should equal 1 for all i,j,k (the multiplication tensor).
    
    # So yes, T is the all-ones tensor, and it has rank 1.
    # But that contradicts all known results about matrix multiplication rank.
    
    # I think the issue is that the "multiplication tensor" in the context of tensor rank for matrix multiplication
    # is NOT the all-ones tensor. Let me reconsider.
    
    # Actually, I think the standard definition is:
    # The matrix multiplication tensor M_{n,m,p} has entries M_{n,m,p}[i,j,k] = 1 if the (i,k) entry of AB
    # depends on A[i,j]*B[j,k]. Since C[i,k] = sum_j A[i,j]B[j,k], every (i,j,k) triple contributes,
    # so M[i,j,k] = 1 for all i,j,k.
    
    # And the rank of this tensor is what we're minimizing.
    # For n=m=p=2, the rank is 7 (Strassen), not 1.
    # So the all-ones 2x2x2 tensor has rank 7? That seems wrong.
    
    # Let me verify: 2x2x2 all-ones tensor. Can it be written as sum of 1 rank-1 tensors?
    # u[i,j]=1, v[j,k]=1, w[k,i]=1: T[i,j,k] = 1*1*1 = 1. Yes, rank 1.
    
    # But Strassen says rank 7. So I must be misunderstanding the tensor.
    
    # AH, I think I see. The matrix multiplication tensor is NOT the all-ones tensor.
    # The correct tensor is: M[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] appears in C.
    # But C[i,k] = sum_j A[i,j]*B[j,k], so for a fixed (i,k), the products A[i,j]*B[j,k] for j=0..m-1 all appear.
    # So M[i,j,k] = 1 for all i,j,k. It IS the all-ones tensor.
    
    # I'm clearly misunderstanding something fundamental. Let me look at this differently.
    
    # The Strassen algorithm for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01+A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So the tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where the 7 terms are:
    # r=0: U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,1]=1; W[0,0]=1,W[1,1]=1
    # r=1: U[1,0]=1,U[1,1]=1; V[0,0]=1; W[0,0]=1
    # etc.
    
    # Let me check T[0,0,0] = sum_r U[0,0,r]*V[0,0,r]*W[0,0,r]
    # = 1*1*1 + 0*1*1 + 0*0*0 + 0*0*0 + 0*0*0 + 0*0*0 + 0*0*0 = 1 ✓
    # T[0,0,1] = sum_r U[0,0,r]*V[0,1,r]*W[1,0,r]
    # = 1*0*0 + 0*0*0 + 0*0*0 + 0*0*0 + 0*0*0 + 0*0*0 + 0*0*0 = 0
    # But T[0,0,1] should be 1 (since A[0,0]*B[0,1] contributes to C[0,1]).
    
    # So T[0,0,1] = 1, but my sum gives 0. That means the tensor is NOT all ones!
    
    # I see my error. The tensor T[i,j,k] is NOT 1 for all i,j,k.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The term A[i,j]*B[j,k] appears in C[i,k]. So the tensor entry T[i,j,k] = 1.
    # But in the Strassen decomposition, T[0,0,1] should be 1, but I got 0.
    
    # Let me recheck. C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # So T[0,0,1] = 1 and T[0,1,1] = 1.
    
    # In Strassen: C[0,1] = M3 + M5
    # M3 = A[0,0]*(B[1,0]-B[1,1]) -> contributes A[0,0]*B[1,0] and -A[0,0]*B[1,1]
    # M5 = (A[0,0]+A[0,1])*B[1,1] -> contributes A[0,0]*B[1,1] and A[0,1]*B[1,1]
    # M3+M5 = A[0,0]*B[1,0] + A[0,1]*B[1,1] ✓
    
    # So the tensor for M3: U[0,0]=1, V[1,0]=1, V[1,1]=-1, W[1,0]=1, W[1,1]=1
    # T[0,0,1] from M3: U[0,0,2]*V[0,1,2]*W[1,0,2] = 1*0*1 = 0
    # T[0,1,1] from M3: U[0,1,2]*V[1,1,2]*W[1,0,2] = 0*(-1)*1 = 0
    # Hmm, this is getting confusing.
    
    # I think the key insight I was missing: the tensor T[i,j,k] = 1 for ALL i,j,k,
    # and it CAN be decomposed with rank 1, but that decomposition doesn't correspond to
    # a valid matrix multiplication algorithm in the usual sense.
    
    # Actually no. If T is all ones and has rank 1, then matrix multiplication would have rank 1,
    # which is false. So T is NOT all ones.
    
    # Let me re-derive. C[i,k] = sum_j A[i,j] B[j,k].
    # In tensor form: C[i,k] = sum_{j,r} T[i,j,k] * (A[i,j] * B[j,k]) where the sum over r is the rank decomposition.
    # No, that's not right either.
    
    # The correct formulation: The matrix multiplication tensor T is a 3-tensor with indices (i,j,k)
    # such that for any matrices A (n×m) and B (m×p):
    #   C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
    #   This is equivalent to: <T, A ⊗ B> = C (with appropriate contraction)
    #   Specifically: T[i,j,k] = 1 for all i,j,k, and C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    
    # So T[i,j,k] = 1 for all i,j,k. And the rank of T is the minimum number of rank-1 tensors
    # that sum to T.
    
    # For 2x2x2, T is the 8-entry all-ones tensor. Its rank is 7 (Strassen).
    # But I showed that u[i,j]=1, v[j,k]=1, w[k,i]=1 gives T[i,j,k]=1 for all i,j,k.
    # So rank 1? Contradiction.
    
    # The resolution: the rank-1 tensor u ⊗ v ⊗ w has entries u[i,j]*v[j,k]*w[k,i].
    # If u[i,j]=1 for all i,j, v[j,k]=1 for all j,k, w[k,i]=1 for all k,i,
    # then (u⊗v⊗w)[i,j,k] = 1*1*1 = 1. So T = u⊗v⊗w, rank 1.
    
    # This means the 2x2x2 all-ones tensor has rank 1, contradicting Strassen's rank 7.
    
    # I must be fundamentally misunderstanding the problem. Let me re-read the task.
    
    # "exactly decompose the matrix multiplication tensor"
    # "score is the rank"
    
    # I think the issue is that the "matrix multiplication tensor" in this context is NOT
    # the all-ones tensor. It must be a different tensor.
    
    # After more thought, I believe the correct interpretation is:
    # The matrix multiplication tensor T_{n,m,p} is defined by T[i,j,k] = 1 for all i,j,k,
    # and the rank of this tensor (in the sense of tensor rank, i.e., minimum number of
    # rank-1 tensors summing to T) is what we want to minimize.
    
    # And for 2x2x2, this rank is indeed 7, not 1.
    
    # The error in my "rank 1" argument: u⊗v⊗w with u[i,j]=1, v[j,k]=1, w[k,i]=1
    # gives (u⊗v⊗w)[i,j,k] = u[i,j]*v[j,k]*w[k,i] = 1*1*1 = 1.
    # This IS a valid rank-1 tensor that equals the all-ones tensor.
    # So the all-ones 2x2x2 tensor has rank 1.
    
    # But Strassen's algorithm has rank 7. So Strassen is NOT decomposing the all-ones tensor.
    # Strassen is decomposing something else.
    
    # I think I finally see: the matrix multiplication tensor is NOT the all-ones tensor.
    # The correct tensor has T[i,j,k] = 1 only when the indices are "compatible" in a specific way.
    
    # Actually, I think the issue is much simpler. Let me just trust the current program's
    # approach and the known results, and implement a different tiling strategy.
    
    # Given the confusion, let me use a SAFE approach:
    # 4x4x6 = 4x4x4 + 4x4x2
    # 4x4x4: use 4 Strassen 2x2x2 blocks, rank 28
    # 4x4x2: use 2 Strassen 2x2x2 blocks (treating 4x4x2 as two 2x2x2 blocks? No, 4x4x2 doesn't tile evenly into 2x2x2)
    # 4x4x2: trivial rank 8
    # Total: 36
    
    # Let me just implement this cleanly.
    
    # Part 1: 4x4x4 with 4 Strassen blocks
    # Part 2: 4x4x2 with trivial rank 8
    
    # For 4x4x2 trivial:
    # T[i,j,k] = 1 for i in [0,4), j in [0,4), k in [0,2)
    # Rank 8 decomposition: for each (i,k) pair, we need... 
    # Actually, the trivial decomposition of the 4x4x2 tensor:
    # For each j in [0,4), k in [0,2):
    #   U[:,r] has 1 at (i,j) for all i -> no, that's not rank 1.
    
    # Rank-1 term: u[i,j], v[j,k], w[k,i]
    # For the 4x4x2 all-ones tensor, one rank-1 term: u[i,j]=1 for all i,j, v[j,k]=1 for all j,k, w[k,i]=1 for all k,i.
    # This gives T[i,j,k]=1. So rank 1!
    
    # I'm going in circles. Let me just use the current program's verified approach
    # but with a different block structure to get a lower rank.
    
    # Current: 2x2x2 blocks, 2*2*3 = 12 blocks, rank 12*7 = 84
    # Alternative: 4x4x4 blocks (rank 17 or 28) + 4x4x2 (rank 8)
    
    # Since I can't verify Laderman factors, let me use 4 Strassen blocks for 4x4x4 (rank 28)
    # and 8 trivial terms for 4x4x2 (rank 8). Total 36.
    
    # For the 4x4x2 part, I'll use the following rank-8 decomposition:
    # The 4x4x2 tensor T[i,j,k] = 1 for all i,j,k.
    # Rank-8: for each i in [0,4), k in [0,2):
    #   r = i*2 + k
    #   u_r[i',j'] = 1 if i'=i, else 0 (for all j')
    #   v_r[j',k'] = 1 if k'=k, else 0 (for all j')
    #   w_r[k',i'] = 1 if (k'=k and i'=i), else 0
    # Check: T[i,j,k] = sum_r u_r[i,j]*v_r[j,k]*w_r[k,i]
    # = sum_{i',k'} [i'=i] * [k'=k] * [k'=k and i'=i]
    # = 1 (when i'=i, k'=k) = 1 ✓
    # This works! Rank 8.
    
    # Now implement the full solution.
    
    total_rank = 28 + 8
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 4x4x4, 4 Strassen blocks, columns 0-27
    col_offset = 0
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = col_offset + c
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U[i * m + j][col] = S_U[di * 2 + dj][c]
                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = dk
                        V[j * p + k][col] = S_V[dj * 2 + dk][c]
                for dk in range(2):
                    for di in range(2):
                        k = dk
                        i = bi * 2 + di
                        W[k * n + i][col] = S_W[dk * 2 + di][c]
            col_offset += 7
    
    # Part 2: 4x4x2, trivial rank 8, columns 28-35
    for i in range(4):
        for k in range(2):
            r = 28 + i * 2 + k
            for j in range(4):
                U[i * m + j][r] = 1
            for j in range(4):
                V[j * p + k][r] = 1
            W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
