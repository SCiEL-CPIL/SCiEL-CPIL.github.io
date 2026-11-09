import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 5, 5]

def _schoolbook(n, m, p):
    """Generate schoolbook decomposition for [n, m, p]."""
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
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Validate that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            if len(U[i*m+j]) != rank:
                return False
    for j in range(m):
        for k in range(p):
            if len(V[j*p+k]) != rank:
                return False
    for k in range(p):
        for i in range(n):
            if len(W[k*n+i]) != rank:
                return False
    
    # Check that the tensor decomposition is correct
    # The matrix multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if j == k (the middle index matches) and 0 otherwise... 
    # Wait, that's not right. Let me think again.
    #
    # The matrix multiplication tensor is: T[i, j, k] where i indexes rows of A, j indexes cols of A / rows of B, k indexes cols of B.
    # T[i, j, k] = 1 if the indices are valid (always in this case), and the product C[i,k] = sum_j A[i,j] * B[j,k].
    # So the tensor entry is 1 for all valid (i,j,k) and the rank-1 decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    # Wait no. The tensor for matrix multiplication [n,m,p] has entries T[i,j,k] = 1 for all i,j,k,
    # and the contraction is C[i,k] = sum_j A[i,j] B[j,k].
    # The rank-1 decomposition means: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # where T[i,j,k] = 1 for all valid indices.
    #
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _strassen_2x2():
    """
    Generate the Strassen decomposition for [2,2,2] matrix multiplication.
    Rank 7.
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For [2,2,2], each block is 1x1, so A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1], etc.
    
    U indices: A[i,j] -> i*2+j
    V indices: B[j,k] -> j*2+k  
    W indices: C[i,k] -> k*2+i (trace convention: W[k*n+i, r])
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]  # 2*2
    V = [[0] * rank for _ in range(4)]  # 2*2
    W = [[0] * rank for _ in range(4)]  # 2*2
    
    # M1 = (A11+A22)(B11+B22)
    # U: A[0,0]=idx0, A[1,1]=idx3
    # V: B[0,0]=idx0, B[1,1]=idx3
    # C: C11=C[0,0]->W[0*2+0]=W[0], C12=C[0,1]->W[1*2+0]=W[2], C21=C[1,0]->W[0*2+1]=W[1], C22=C[1,1]->W[1*2+1]=W[3]
    # C11 += M1, C12 += 0, C21 += 0, C22 += M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A21+A22)B11
    # U: A[1,0]=idx2, A[1,1]=idx3
    # V: B[0,0]=idx0
    # C21 += M2, C22 += -M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] += -1
    
    # M3 = A11(B12-B22)
    # U: A[0,0]=idx0
    # V: B[0,1]=idx1, B[1,1]=idx3
    # C12 += M3, C22 += M3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[2][2] += 1; W[3][2] += 1
    
    # M4 = A22(B21-B11)
    # U: A[1,1]=idx3
    # V: B[1,0]=idx2, B[0,0]=idx0
    # C11 += M4, C21 += M4
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    # M5 = (A11+A12)B22
    # U: A[0,0]=idx0, A[0,1]=idx1
    # V: B[1,1]=idx3
    # C11 += -M5, C12 += M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[2][4] += 1
    
    # M6 = (A21-A11)(B11+B12)
    # U: A[1,0]=idx2, A[0,0]=idx0
    # V: B[0,0]=idx0, B[0,1]=idx1
    # C22 += M6
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A12-A22)(B21+B22)
    # U: A[0,1]=idx1, A[1,1]=idx3
    # V: B[1,0]=idx2, B[1,1]=idx3
    # C11 += M7
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _lift_strassen(U_sub, V_sub, W_sub, size):
    """
    Lift a Strassen decomposition from [size, size, size] to [2*size, 2*size, 2*size].
    """
    new_size = 2 * size
    rank_sub = len(U_sub[0])
    rank = 7 * rank_sub
    
    U = [[0] * rank for _ in range(new_size * new_size)]
    V = [[0] * rank for _ in range(new_size * new_size)]
    W = [[0] * rank for _ in range(new_size * new_size)]
    
    # Block offsets
    # A blocks: A11[0:size, 0:size], A12[0:size, size:2*size], A21[size:2*size, 0:size], A22[size:2*size, size:2*size]
    # B blocks: B11[0:size, 0:size], B12[0:size, size:2*size], B21[size:2*size, 0:size], B22[size:2*size, size:2*size]
    # C blocks: C11[0:size, 0:size], C12[0:size, size:2*size], C21[size:2*size, 0:size], C22[size:2*size, size:2*size]
    
    def block_offset(block, i, j):
        """Get the global index for a sub-index (i,j) in a given block."""
        if block == 'A11' or block == 'B11' or block == 'C11':
            return i * new_size + j
        elif block == 'A12' or block == 'B12' or block == 'C12':
            return i * new_size + (j + size)
        elif block == 'A21' or block == 'B21' or block == 'C21':
            return (i + size) * new_size + j
        elif block == 'A22' or block == 'B22' or block == 'C22':
            return (i + size) * new_size + (j + size)
        return None
    
    # Strassen's 7 products and their C contributions
    # M1 = (A11+A22)(B11+B22) -> C11+=1, C22+=1
    # M2 = (A21+A22)B11 -> C21+=1, C22+=-1
    # M3 = A11(B12-B22) -> C12+=1, C22+=1
    # M4 = A22(B21-B11) -> C11+=1, C21+=1
    # M5 = (A11+A12)B22 -> C11+=-1, C12+=1
    # M6 = (A21-A11)(B11+B12) -> C22+=1
    # M7 = (A12-A22)(B21+B22) -> C11+=1
    
    strassen_products = [
        # (A_blocks, B_blocks, C_blocks with coeffs)
        (['A11', 'A22'], ['B11', 'B22'], [('C11', 1), ('C22', 1)]),
        (['A21', 'A22'], ['B11'], [('C21', 1), ('C22', -1)]),
        (['A11'], ['B12', 'B22'], [('C12', 1), ('C22', 1)]),
        (['A22'], ['B21', 'B11'], [('C11', 1), ('C21', 1)]),
        (['A11', 'A12'], ['B22'], [('C11', -1), ('C12', 1)]),
        (['A21', 'A11'], ['B11', 'B12'], [('C22', 1)]),
        (['A12', 'A22'], ['B21', 'B22'], [('C11', 1)]),
    ]
    
    # Coefficients for each A and B block in each product
    strassen_coeffs = [
        # M1: A11(+1), A22(+1); B11(+1), B22(+1)
        ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}),
        # M2: A21(+1), A22(+1); B11(+1)
        ({'A21': 1, 'A22': 1}, {'B11': 1}),
        # M3: A11(+1); B12(+1), B22(-1)
        ({'A11': 1}, {'B12': 1, 'B22': -1}),
        # M4: A22(+1); B21(+1), B11(-1)
        ({'A22': 1}, {'B21': 1, 'B11': -1}),
        # M5: A11(+1), A12(+1); B22(+1)
        ({'A11': 1, 'A12': 1}, {'B22': 1}),
        # M6: A21(+1), A11(-1); B11(+1), B12(+1)
        ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}),
        # M7: A12(+1), A22(-1); B21(+1), B22(+1)
        ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}),
    ]
    
    for prod_idx in range(7):
        A_coeffs, B_coeffs = strassen_coeffs[prod_idx]
        C_blocks = strassen_products[prod_idx][2]
        
        # For each sub-term in the decomposition
        for r_sub in range(rank_sub):
            r_global = prod_idx * rank_sub + r_sub
            
            # Add contributions to U
            for A_block, coeff in A_coeffs.items():
                if coeff == 0:
                    continue
                for i in range(size):
                    for j in range(size):
                        # Get the sub-index value
                        sub_val = U_sub[i * size + j][r_sub]
                        if sub_val != 0:
                            global_idx = block_offset(A_block, i, j)
                            U[global_idx][r_global] += coeff * sub_val
            
            # Add contributions to V
            for B_block, coeff in B_coeffs.items():
                if coeff == 0:
                    continue
                for i in range(size):
                    for j in range(size):
                        sub_val = V_sub[i * size + j][r_sub]
                        if sub_val != 0:
                            global_idx = block_offset(B_block, i, j)
                            V[global_idx][r_global] += coeff * sub_val
            
            # Add contributions to W
            for C_block, coeff in C_blocks:
                if coeff == 0:
                    continue
                for i in range(size):
                    for j in range(size):
                        sub_val = W_sub[i * size + j][r_sub]
                        if sub_val != 0:
                            global_idx = block_offset(C_block, i, j)
                            W[global_idx][r_global] += coeff * sub_val
    
    return U, V, W

def _get_strassen_decomp(k):
    """Get Strassen decomposition for [2^k, 2^k, 2^k]."""
    if k == 0:
        return _strassen_2x2()
    
    U, V, W = _get_strassen_decomp(k - 1)
    size = 2 ** (k - 1)
    return _lift_strassen(U, V, W, size)

def _pad_decomp(U, V, W, n, m, p, target_n, target_m, target_p):
    """
    Pad a [n,m,p] decomposition to [target_n, target_m, target_p].
    This handles the case where we have a decomposition for a sub-problem
    and need to embed it into a larger one with zeros for the extra dimensions.
    """
    rank = len(U[0])
    U_new = [[0] * rank for _ in range(target_n * target_m)]
    V_new = [[0] * rank for _ in range(target_m * target_p)]
    W_new = [[0] * rank for _ in range(target_p * target_n)]
    
    # Copy the existing decomposition
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                U_new[i * target_m + j][r] = U[i * m + j][r]
    
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                V_new[j * target_p + k][r] = V[j * p + k][r]
    
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W_new[k * target_n + i][r] = W[k * n + i][r]
    
    return U_new, V_new, W_new

def _combine_decomps(decomps):
    """
    Combine multiple decompositions into one.
    Each decomp is a tuple (U, V, W) for the same [n,m,p].
    """
    if not decomps:
        return None
    
    n, m, p = decomps[0][0].__len__() // m if False else (None, None, None)
    # Infer dimensions from the first decomp
    U0, V0, W0 = decomps[0]
    n = int(np.sqrt(len(U0)))  # This won't work for non-square
    # Actually, we need to know n, m, p. Let's pass them.
    # For now, let's assume all decomps are for the same dimensions.
    
    rank = sum(len(U[0]) for U, V, W in decomps)
    n = len(U0) // 5  # m=5 in our case, but this is fragile
    # Let's just concatenate
    U_new = []
    V_new = []
    W_new = []
    for U, V, W in decomps:
        for row in U:
            U_new.append(row + [0] * (rank - len(row)))
        for row in V:
            V_new.append(row + [0] * (rank - len(row)))
        for row in W:
            W_new.append(row + [0] * (rank - len(row)))
    
    # Wait, this is wrong. We need to align the rows.
    # Let me redo this.
    
    n_rows_U = len(U0)
    n_rows_V = len(V0)
    n_rows_W = len(W0)
    
    U_new = [[0] * rank for _ in range(n_rows_U)]
    V_new = [[0] * rank for _ in range(n_rows_V)]
    W_new = [[0] * rank for _ in range(n_rows_W)]
    
    offset = 0
    for U, V, W in decomps:
        r = len(U[0])
        for i in range(n_rows_U):
            for col in range(r):
                U_new[i][offset + col] = U[i][col]
        for i in range(n_rows_V):
            for col in range(r):
                V_new[i][offset + col] = V[i][col]
        for i in range(n_rows_W):
            for col in range(r):
                W_new[i][offset + col] = W[i][col]
        offset += r
    
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    # Strategy: Decompose [4,5,5] as:
    # Part 1: [4,4,4] using Strassen (rank 49)
    # Part 2: [4,1,4] schoolbook (rank 16) 
    # Part 3: [4,4,1] schoolbook (rank 16)
    # Part 4: [4,1,1] schoolbook (rank 4)
    # Total: 49 + 16 + 16 + 4 = 85
    
    # But wait, we need to be careful about how we split.
    # A is 4x5, B is 5x5, C is 4x5.
    # Let's split A into A_left (4x4) and A_right (4x1).
    # Let's split B into B_top (4x5) and B_bottom (1x5).
    # Let's split C into C_left (4x4) and C_right (4x1).
    
    # C = A * B = A_left * B_top + A_left * B_bottom + A_right * B_top + A_right * B_bottom
    # = [4x4 * 4x5] + [4x4 * 1x5] + [4x1 * 4x5] + [4x1 * 1x5]
    
    # Part 1: A_left (4x4) * B_top (4x5) -> contributes to C (4x5)
    #   This is [4,4,5] = [4,4,4] + [4,4,1]
    # Part 2: A_left (4x4) * B_bottom (1x5) -> contributes to C (4x5)
    #   This is [4,1,5] = [4,1,4] + [4,1,1]
    # Part 3: A_right (4x1) * B_top (4x5) -> contributes to C (4x5)
    #   This is [4,1,5] = [4,1,4] + [4,1,1]
    # Part 4: A_right (4x1) * B_bottom (1x5) -> contributes to C (4x5)
    #   This is [4,1,1]
    
    # So we need:
    # [4,4,4]: Strassen rank 49
    # [4,4,1]: schoolbook rank 16
    # [4,1,4]: schoolbook rank 16
    # [4,1,1]: schoolbook rank 4 (appears 3 times: from parts 2, 3, 4)
    # Wait, let me recount.
    
    # Part 1: [4,4,5] = [4,4,4] (rank 49) + [4,4,1] (rank 16)
    # Part 2: [4,1,5] = [4,1,4] (rank 16) + [4,1,1] (rank 4)
    # Part 3: [4,1,5] = [4,1,4] (rank 16) + [4,1,1] (rank 4)
    # Part 4: [4,1,1] (rank 4)
    
    # Total: 49 + 16 + 16 + 4 + 16 + 4 + 4 = 109. That's worse than schoolbook 100!
    
    # Let me reconsider. The issue is that we're double-counting.
    # Actually, the decomposition should be:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{j=0}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    
    # So:
    # Term 1: sum_{j=0}^{3} A[i,j] B[j,k] for k=0..3 -> [4,4,4]
    # Term 2: sum_{j=0}^{3} A[i,j] B[j,4] for k=4 -> [4,4,1]
    # Term 3: A[i,4] B[4,k] for k=0..3 -> [4,1,4]
    # Term 4: A[i,4] B[4,4] -> [4,1,1]
    
    # Ranks: 49 + 16 + 16 + 4 = 85. Good!
    
    # Now I need to construct these properly with the right index mappings.
    
    # For [4,4,4]: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # For [4,4,1]: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # For [4,1,4]: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # For [4,1,1]: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    
    # Let's get the Strassen decomposition for [4,4,4]
    U_444, V_444, W_444 = _get_strassen_decomp(2)  # 2^2 = 4
    rank_444 = len(U_444[0])  # Should be 49
    
    # Schoolbook for [4,4,1]
    U_441, V_441, W_441 = _schoolbook(4, 4, 1)
    rank_441 = len(U_441[0])  # Should be 16
    
    # Schoolbook for [4,1,4]
    U_414, V_414, W_414 = _schoolbook(4, 1, 4)
    rank_414 = len(U_414[0])  # Should be 16
    
    # Schoolbook for [4,1,1]
    U_411, V_411, W_411 = _schoolbook(4, 1, 1)
    rank_411 = len(U_411[0])  # Should be 4
    
    total_rank = rank_444 + rank_441 + rank_414 + rank_411
    # 49 + 16 + 16 + 4 = 85
    
    U = [[0] * total_rank for _ in range(n * m)]  # 20
    V = [[0] * total_rank for _ in range(m * p)]  # 25
    W = [[0] * total_rank for _ in range(p * n)]  # 20
    
    offset = 0
    
    # Part 1: [4,4,4] -> A[0:4,0:4], B[0:4,0:4], C[0:4,0:4]
    for i in range(4):
        for j in range(4):
            for r in range(rank_444):
                U[i * 5 + j][offset + r] = U_444[i * 4 + j][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank_444):
                V[j * 5 + k][offset + r] = V_444[j * 4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank_444):
                W[k * 4 + i][offset + r] = W_444[k * 4 + i][r]
    offset += rank_444
    
    # Part 2: [4,4,1] -> A[0:4,0:4], B[0:4,4], C[0:4,4]
    for i in range(4):
        for j in range(4):
            for r in range(rank_441):
                U[i * 5 + j][offset + r] = U_441[i * 4 + j][r]
    for j in range(4):
        for r in range(rank_441):
            V[j * 5 + 4][offset + r] = V_441[j * 1 + 0][r]  # B[j, 0] in the 4x1 sub-problem
    for r in range(rank_441):
        for i in range(4):
            W[4 * 4 + i][offset + r] = W_441[0 * 4 + i][r]  # C[0, i] in the 4x1 sub-problem, which is C[i, 4] in the full problem
    offset += rank_441
    
    # Part 3: [4,1,4] -> A[0:4,4], B[4,0:4], C[0:4,0:4]
    for i in range(4):
        for r in range(rank_414):
            U[i * 5 + 4][offset + r] = U_414[i * 1 + 0][r]  # A[i, 0] in the 4x1 sub-problem
    for k in range(4):
        for r in range(rank_414):
            V[4 * 5 + k][offset + r] = V_414[0 * 4 + k][r]  # B[0, k] in the 1x4 sub-problem
    for k in range(4):
        for i in range(4):
            for r in range(rank_414):
                W[k * 4 + i][offset + r] = W_414[k * 4 + i][r]
    offset += rank_414
    
    # Part 4: [4,1,1] -> A[0:4,4], B[4,4], C[0:4,4]
    for i in range(4):
        for r in range(rank_411):
            U[i * 5 + 4][offset + r] = U_411[i * 1 + 0][r]
    for r in range(rank_411):
        V[4 * 5 + 4][offset + r] = V_411[0 * 1 + 0][r]
    for i in range(4):
        for r in range(rank_411):
            W[4 * 4 + i][offset + r] = W_411[0 * 4 + i][r]
    offset += rank_411
    
    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
