import numpy as np
import time

# Deadline guard: 150s limit, use 145s to be safe
DEADLINE = 145.0
START_TIME = time.time()

DIMENSIONS = [4, 4, 8]

def _check_deadline():
    if time.time() - START_TIME > DEADLINE:
        return True
    return False

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (2, 2, 7)
    V shape: (2, 2, 7)
    W shape: (2, 2, 7)
    """
    U = np.zeros((2, 2, 7))
    V = np.zeros((2, 2, 7))
    W = np.zeros((2, 2, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0, 0] = 1; U[1, 1, 0] = 1
    V[0, 0, 0] = 1; V[1, 1, 0] = 1
    W[0, 0, 0] = 1; W[1, 1, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[1, 0, 1] = 1; U[1, 1, 1] = 1
    V[0, 0, 1] = 1
    W[0, 1, 1] = 1; W[1, 1, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 0, 2] = 1
    V[0, 1, 2] = 1; V[1, 1, 2] = -1
    W[1, 0, 2] = 1; W[1, 1, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[1, 1, 3] = 1
    V[1, 0, 3] = 1; V[0, 0, 3] = -1
    W[0, 0, 3] = 1; W[0, 1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 0, 4] = 1; U[0, 1, 4] = 1
    V[1, 1, 4] = 1
    W[0, 0, 4] = -1; W[1, 0, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[1, 0, 5] = 1; U[0, 0, 5] = -1
    V[0, 0, 5] = 1; V[0, 1, 5] = 1
    W[1, 1, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[0, 1, 6] = 1; U[1, 1, 6] = -1
    V[1, 0, 6] = 1; V[1, 1, 6] = 1
    W[0, 0, 6] = 1

    return U, V, W

def _build_4x4_factors():
    """
    Builds 4x4x4 factors using recursive Strassen.
    Returns U, V, W of shapes (4, 4, 49).
    """
    if _check_deadline():
        raise TimeoutError()

    U2, V2, W2 = _strassen_2x2_factors()
    
    rank_4 = 49
    U4 = np.zeros((4, 4, rank_4))
    V4 = np.zeros((4, 4, rank_4))
    W4 = np.zeros((4, 4, rank_4))
    
    # Recursive step:
    # The 4x4 multiplication is decomposed into 7 multiplications of 2x2 blocks.
    # The indices R for 4x4 are formed by R = 7*s + r, where s is the Strassen step (0..6) and r is the inner Strassen step (0..6).
    # Actually, the standard recursion for M_n uses 7 M_{n/2}.
    # For M_4, we use 7 M_2.
    # The factors are Kronecker products of the block selection and the inner factors.
    
    # U4[i, j, R]
    # i, j in 0..3. i = 2*i1 + i0, j = 2*j1 + j0.
    # R = 7*s + r.
    # U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2 * i1 + i0
                    j = 2 * j1 + j0
                    for s in range(7):
                        u_block = U2[i1, j1, s]
                        if u_block == 0: continue
                        for r in range(7):
                            u_inner = U2[i0, j0, r]
                            if u_inner == 0: continue
                            R = 7 * s + r
                            U4[i, j, R] = u_block * u_inner
    
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2 * j1 + j0
                    k = 2 * k1 + k0
                    for s in range(7):
                        v_block = V2[j1, k1, s]
                        if v_block == 0: continue
                        for r in range(7):
                            v_inner = V2[j0, k0, r]
                            if v_inner == 0: continue
                            R = 7 * s + r
                            V4[j, k, R] = v_block * v_inner
    
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2 * k1 + k0
                    i = 2 * i0 + i0 # Typo in thought, correct is i = 2*i1 + i0
                    i = 2 * i1 + i0
                    for s in range(7):
                        w_block = W2[k1, i1, s]
                        if w_block == 0: continue
                        for r in range(7):
                            w_inner = W2[k0, i0, r]
                            if w_inner == 0: continue
                            R = 7 * s + r
                            W4[k, i, R] = w_block * w_inner

    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Strategy: Use Kronecker product of 2x2x2 Strassen (rank 7) with a 2x2x4 tensor.
    # The 4x4x8 tensor = (2x2x2) ⊗ (2x2x4) where:
    #   - The first 2x2x2 handles the "outer" 2x2 block structure (rows/cols split in half)
    #   - The 2x2x4 handles: 2x2 A-block times 2x4 B-block -> 2x4 C-block
    #
    # For the inner 2x2x4: we can use Strassen on each column pair independently,
    # but that gives rank 14. Instead, we exploit that B is 2x4 = [B_left | B_right]
    # where B_left and B_right are 2x2. The product A*B = A*B_left + A*B_right (block-wise).
    # 
    # Actually, the simplest correct approach with rank 56:
    # Decompose 4x4x8 as: split the 8 columns into two groups of 4.
    # For each group, use 4x4x4 Strassen (rank 49). Total: 98.
    #
    # To get rank 56, we need a smarter decomposition. Let's think about it differently:
    # 4x4x8 = (2x2x2) ⊗ (2x2x4)
    # The 2x2x4 tensor: multiply 2x2 by 2x4. 
    # This can be done with rank 7 (Strassen-like) if we treat it as:
    #   Split B into two 2x2 blocks: B = [B0 | B1]
    #   C = [A*B0 | A*B1]
    #   Using Strassen for each: 7+7=14 for the inner.
    #   Total rank: 7 * 14 = 98. Still 98.
    #
    # Alternative: Use a direct 2x2x4 decomposition.
    # The 2x2x4 multiplication tensor has entries T[a,b,c,d] where a,b in {0,1}, c in {0,1}, d in {0,1,2,3}.
    # Actually, let's just use the known result: rank of 2x2x4 matrix mult is at most 7*2=14
    # via independent Strassen, but can it be less?
    #
    # For now, let's try a different structural approach:
    # Use the fact that 4x4x8 can be seen as 4x(4x8) or (4x4)x8.
    # 
    # Let me try: rank 56 = 7 * 8. 
    # If we can find a 2x2x4 tensor of rank 8, then 7*8=56.
    #
    # 2x2x4: A(2x2) * B(2x4) = C(2x4)
    # B = [[b00, b01, b02, b03],
    #      [b10, b11, b12, b13]]
    #
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[0,2] = a00*b02 + a01*b12
    # C[0,3] = a00*b03 + a01*b13
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # C[1,2] = a10*b02 + a11*b12
    # C[1,3] = a10*b03 + a11*b13
    #
    # This is just 8 independent dot products of length 2.
    # Rank of this tensor: each output entry is a bilinear form.
    # The tensor is the direct sum of 8 copies of the rank-2 "dot product" tensor.
    # So rank = 8.
    #
    # Therefore: 4x4x8 = (2x2x2 Strassen) ⊗ (2x2x4 rank-8) has rank 7*8 = 56.
    
    if _check_deadline():
        raise TimeoutError()
    
    # Get 2x2x2 Strassen factors
    U2, V2, W2 = _strassen_2x2_factors()
    
    # Define 2x2x4 factors (rank 8)
    # U24[i,j,r] for i,j in {0,1}, r in {0..7}
    # V24[j,k,r] for j in {0,1}, k in {0..3}, r in {0..7}
    # W24[k,i,r] for k in {0..3}, i in {0,1}, r in {0..7}
    
    U24 = np.zeros((2, 2, 8))
    V24 = np.zeros((2, 4, 8))
    W24 = np.zeros((4, 2, 8))
    
    # For r=0..3: handle column r of B (which is 2x1)
    # C[0, r] = A[0,:].B[:,r] = a00*b0r + a01*b1r
    # C[1, r] = A[1,:].B[:,r] = a10*b0r + a11*b1r
    # This is just the standard rank-4 decomposition for 2x2x2, but we only use the "diagonal" parts.
    # Actually, for a single column, it's two independent dot products of size 2.
    # We can encode each dot product as one rank-1 term.
    # a00*b0r: U[0,0]=1, V[0,r]=1, W[r,0]=1
    # a01*b1r: U[0,1]=1, V[1,r]=1, W[r,0]=1
    # a10*b0r: U[1,0]=1, V[0,r]=1, W[r,1]=1
    # a11*b1r: U[1,1]=1, V[1,r]=1, W[r,1]=1
    
    # So for each column r, we need 4 rank-1 terms.
    # Total: 4 columns * 4 terms = 16 terms. That's rank 16, not 8.
    #
    # Wait, I made an error. The 2x2x4 tensor has 8 output entries, each is a sum of 2 products.
    # The rank is the minimum number of rank-1 tensors whose sum gives the full tensor.
    # Each output C[i,k] = sum_j A[i,j]*B[j,k].
    # This is a direct sum of 8 "vector outer product" tensors (each of rank 2 for the A,B part, but...
    # 
    # Let me reconsider. The tensor T[i,j,k] for 2x2x2 has rank 4 (schoolbook) or 7 (Strassen).
    # For 2x2x4, it's like having 4 independent 2x2x2 tensors (one for each column of B).
    # Each 2x2x2 has rank at least 4 (actually, the minimal rank for 2x2x2 is 7 for Strassen, but the
    # "diagonal" tensor where only C[i,i] is nonzero has rank 4).
    #
    # Hmm, let me just use the straightforward construction:
    # For each of the 4 columns of B, use the 4 schoolbook terms.
    # Total rank for 2x2x4: 4*4 = 16.
    # Then 4x4x8 rank: 7*16 = 112. Worse than 98.
    #
    # Let me try another approach: use Strassen for the 2x2x4 by splitting B into two 2x2 blocks.
    # B = [B_left | B_right], each 2x2.
    # A*B = [A*B_left | A*B_right]
    # Use Strassen for each: rank 7 each, total 14.
    # Then 4x4x8 rank: 7*14 = 98. Same as before.
    #
    # OK so the Kronecker product approach doesn't immediately give < 98.
    # Let me just keep the current rank-98 construction but verify it's correct.
    
    # Use the existing 4x4x4 Strassen factors
    U4, V4, W4 = _build_4x4_factors()
    
    # Construct 4x4x8 by duplicating for two column halves
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First half: columns 0-3 of B, using ranks 0-48
    for i in range(n):
        for j in range(m):
            for R in range(49):
                U[i * m + j, R] = U4[i, j, R]
                
    for j in range(m):
        for k in range(4):
            for R in range(49):
                V[j * p + k, R] = V4[j, k, R]
                
    for k in range(4):
        for i in range(n):
            for R in range(49):
                W[k * n + i, R] = W4[k, i, R]
                
    # Second half: columns 4-7 of B, using ranks 49-97
    for i in range(n):
        for j in range(m):
            for R in range(49):
                U[i * m + j, 49 + R] = U4[i, j, R]
                
    for j in range(m):
        for kp in range(4):
            k = 4 + kp
            for R in range(49):
                V[j * p + k, 49 + R] = V4[j, kp, R]
                
    for kp in range(4):
        k = 4 + kp
        for i in range(n):
            for R in range(49):
                W[k * n + i, 49 + R] = W4[kp, i, R]

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
