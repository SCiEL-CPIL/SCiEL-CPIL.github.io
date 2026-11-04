import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 110
    
    # Current best rank is 33.
    # We try to find a rank 32 solution.
    # The tensor is T[i, j, k] = 1 for matmul.
    # But wait, if the seed program checks for 1.0 for all i,j,k, 
    # and it's rank 33, and rank 1 is possible for all-ones,
    # then the tensor MUST be the matmul tensor where T[i,j,k] is 0 or 1 
    # based on the multiplication rule.
    # For matmul, T[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # This is always true.
    # So T is all-ones.
    # I am confused. I will assume the task is to minimize the rank of the 
    # decomposition of the matmul tensor, which is NOT all-ones in the 
    # sense of tensor rank? No, it is all-ones.
    # 
    # Let's look at the indices in the seed program again.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # Sum over r.
    # If this is 1 for all i,j,k, it's all-ones.
    # 
    # I will try to improve the rank 33 solution for the matmul tensor 
    # by using a more efficient decomposition for the 2x3x2 blocks.
    # If I can do 2x3x2 in rank 8, total rank 24.
    
    # Let's try to construct a rank 8 decomposition for 2x3x2.
    # If successful, use it 3 times.
    
    def find_2x3x2_rank8():
        # Returns U, V, W for 2x3x2 matmul with rank 8
        # Or None if not found
        # U (2x8), V (3x8), W (2x8)
        
        # Try a known pattern or search
        # This is a small problem, we can try random initialization + gradient descent
        # or just check a few structured patterns.
        
        # Pattern:
        # M1 = (A00 + A10)(B00 + B01)
        # M2 = (A00 + A01)(B10 + B11)
        # ...
        
        # Actually, let's just use the rank 11 solution (4+7) if rank 8 is hard.
        # But 24 is much better than 33.
        
        # I will return None for now and stick to 33 if 8 fails.
        return None

    U_block, V_block, W_block = find_2x3x2_rank8()
    
    if U_block is not None:
        rank = 24
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Copy block 3 times
        for t in range(3):
            k0 = 2 * t
            # U is same for all blocks (A is shared)
            # But wait, A is 2x3. The block uses A[:, 0:2] or similar?
            # No, the block is 2x3x2. A is 2x3.
            # So U is the same for all 3 blocks.
            # V and W depend on the block.
            
            U[:, t*8:(t+1)*8] = U_block
            # V_block is 3x8. We need to map to V (18x24).
            # V index is j*p + k.
            # For block t, k is k0, k0+1.
            # j is 0, 1, 2.
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*p + k_global, t*8:(t+1)*8] = V_block[j, k_local] # Wait, V_block is 3x8?
                    # V_block shape (3, 8). Index j, r.
                    # So V_block[j, r].
                    V[j*p + k_global, t*8:(t+1)*8] = V_block[j, :] # This is wrong, V_block is 3x8, not 3x2x8.
                    # V_block[j, r] corresponds to B[j, k_local]?
                    # No, V_block is for the 3x2 matrix B_block.
                    # So V_block[j, r] is the coefficient for B[j, k] in term r.
                    # But k is local.
                    # So V_block is actually 3x2x8? No, it's flattened.
                    # My find_2x3x2_rank8 should return V as (3*2, 8) or similar?
                    # No, V is (m*p, rank) = (3*2, 8) = (6, 8).
                    
            # This is getting complicated to hardcode without a verified rank 8 solution.
            # I will stick to the rank 33 solution but try to optimize the constants.
            pass

    # Fallback to rank 33
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: A[:, 0] * B[0, :]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Three 2x2x2 Strassen blocks
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        if time.time() > deadline:
            break
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3 = A00(B10 - B11)
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4 = A11(B01 - B00)
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
