"""Improved program using Systematic Block-Structured Search."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Target rank range to search
    # We know 129 works. We want to find something better.
    # Theoretical lower bound is max(16, 16, 36) = 36? No, rank of multiplication tensor M_{n,m,p} >= max(n*m, m*p, p*n).
    # Here max(16, 16, 36) = 36.
    # But we know 4x4x4 has rank 49 (Strassen).
    # Let's try to find a rank between 49 and 129.
    
    # We will use a block-structured approach.
    # The tensor T is M_{4,4,9}.
    # We can view it as T = T_1 \oplus T_2 where T_1 is M_{4,4,4} and T_2 is M_{4,4,5}.
    # However, a general decomposition does not have to be a direct sum.
    # We can try to find a decomposition of rank R where R < 129.
    
    # Let's try a randomized search for a sparse decomposition.
    # We will represent the factors U, V, W as lists of lists.
    # To speed up verification, we will use numpy but with integer arithmetic to avoid floating point errors.
    
    # Since the problem size is small (16xR, 16xR, 36xR), we can verify quickly.
    
    # Let's try a few ranks in descending order of likelihood of success?
    # Or just try a fixed rank that is likely to work?
    # The Strassen rank for 4x4x4 is 49.
    # If we could reuse 49 terms for the 4x4x4 part and find ~20-30 terms for the 4x4x5 part, we'd be at ~70-80.
    
    # Strategy:
    # 1. Start with a known valid decomposition (e.g., the 129 rank one from the current code, or a simpler 144 schoolbook one).
    # 2. Use local search to reduce the rank by zeroing out entries and re-optimizing?
    #    This is hard because rank is discrete.
    
    # Alternative Strategy:
    # Construct a candidate decomposition with a specific rank R using a heuristic.
    # Let's try R = 80.
    # We can try to construct U, V, W such that they are sparse.
    
    # Let's implement a simple randomized local search on the entries of U, V, W for a fixed rank R.
    # But initializing R=80 randomly is hard to converge.
    
    # Better: Use the structure of the problem.
    # The 4x4x4 part is "hard" (rank 49).
    # The 4x4x5 part is "easier".
    # Let's assume the optimal solution has a block structure:
    # U = [U1  U2]
    # V = [V1  V2]
    # W = [W1  W2]
    # where U1, V1, W1 correspond to the 4x4x4 core (first 4 cols of B, first 4 rows of C).
    # and U2, V2, W2 handle the rest.
    
    # If we take the optimal 4x4x4 decomposition (Rank 49) for the first block,
    # and find a decomposition for the remaining part (4x4x5) with Rank R2,
    # and if we can make U2, V2, W2 sparse and non-overlapping in rank terms,
    # total rank = 49 + R2.
    
    # The current program does exactly this with R2 = 80 (Schoolbook for 4x4x5).
    # Can we find a better R2 for 4x4x5?
    # The rank of M_{4,4,5} is at least max(16, 20, 20) = 20.
    # Schoolbook is 80.
    # Strassen-like methods for rectangular matrices?
    
    # Let's try to find a decomposition for M_{4,4,5} with rank < 80.
    # We can use a randomized search for the factors of M_{4,4,5}.
    
    def find_decomp_4x4x5(R_target):
        """
        Find a decomposition of the 4x4x5 multiplication tensor with rank R_target.
        Returns U, V, W or None.
        U: 16 x R_target
        V: 20 x R_target
        W: 20 x R_target
        """
        if time.time() > _deadline - 5:
            return None
            
        # Initialize with random sparse matrices
        # We want to satisfy sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if (i,j,k) is in the tensor, else 0.
        # The tensor M_{4,4,5} has 1s at (i,j,k) for i in 0..3, j in 0..3, k in 0..4.
        
        # Let's try a specific structure:
        # Maybe we can use Strassen for the 4x4x4 part of the 4x4x5?
        # No, the 5th column is separate.
        
        # Let's try a simple heuristic:
        # Start with Schoolbook (Rank 80).
        # Try to remove terms.
        
        # Actually, let's just try to find *any* valid decomposition for 4x4x5 with rank R_target
        # using a randomized approach with restarts.
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(50):
            if time.time() > _deadline - 5:
                break
                
            # Random initialization
            # To make it work, we need to initialize close to a solution.
            # Let's initialize with Schoolbook and then zero out random columns?
            # No, that breaks the equation.
            
            # Let's try to construct U, V, W from scratch with sparsity.
            # This is a hard constraint satisfaction problem.
            
            # Alternative: Use the known 4x4x4 Strassen (Rank 49) for the first 4 columns of the 4x4x5?
            # If we split 4x4x5 into 4x4x4 and 4x4x1.
            # Rank(4x4x4) = 49.
            # Rank(4x4x1) = 16 (Schoolbook) or less?
            # M_{4,4,1} is just a vector of 16 ones? No, it's a tensor where C = A * b.
            # This is a linear map from 16-dim A to 4-dim C, for a fixed b? No, b is variable.
            # It's the tensor for multiplying a 4x4 matrix by a 4x1 vector.
            # Rank is 16.
            # So Rank(4x4x5) <= 49 + 16 = 65.
            
            # So if we can decompose 4x4x5 into 4x4x4 (Rank 49) + 4x4x1 (Rank 16),
            # and combine with the main 4x4x4 (Rank 49),
            # we might be able to share the 49 terms?
            
            # Let's try this structure:
            # Total Rank = 49 + 16 = 65?
            # Let's verify if we can construct this.
            
            if R_target >= 65:
                # Construct 4x4x4 Strassen
                U2 = [[0]*7 for _ in range(4)]
                V2 = [[0]*7 for _ in range(4)]
                W2 = [[0]*7 for _ in range(4)]
                
                U2[0][0] = 1; U2[3][0] = 1
                V2[0][0] = 1; V2[3][0] = 1
                W2[0][0] = 1; W2[3][0] = 1
                U2[3][1] = 1
                V2[2][1] = 1; V2[0][1] = -1
                W2[1][1] = 1
                U2[2][2] = 1; U2[3][2] = 1
                V2[1][2] = 1
                W2[2][2] = 1
                U2[0][3] = 1
                V2[1][3] = 1; V2[3][3] = -1
                W2[0][3] = 1
                U2[0][4] = 1; U2[1][4] = 1
                V2[3][4] = 1
                W2[1][4] = 1
                U2[2][5] = 1; U2[0][5] = -1
                V2[0][5] = 1; V2[2][5] = 1
                W2[0][5] = 1; W2[1][5] = 1
                U2[1][6] = 1; U2[3][6] = -1
                V2[1][6] = 1; V2[3][6] = 1
                W2[2][6] = 1; W2[3][6] = 1

                U4 = [[0]*49 for _ in range(16)]
                V4 = [[0]*49 for _ in range(16)]
                W4 = [[0]*49 for _ in range(16)]
                
                for i1 in range(2):
                    for j1 in range(2):
                        for c1 in range(7):
                            u1 = U2[i1*2+j1][c1]
                            if u1 == 0: continue
                            for i2 in range(2):
                                for j2 in range(2):
                                    for c2 in range(7):
                                        u2 = U2[i2*2+j2][c2]
                                        if u2 == 0: continue
                                        idx_u = (i1*2 + i2)*4 + (j1*2 + j2)
                                        idx_c = c1*7 + c2
                                        U4[idx_u][idx_c] = u1 * u2

                for j1 in range(2):
                    for k1 in range(2):
                        for c1 in range(7):
                            v1 = V2[j1*2+k1][c1]
                            if v1 == 0: continue
                            for j2 in range(2):
                                for k2 in range(2):
                                    for c2 in range(7):
                                        v2 = V2[j2*2+k2][c2]
                                        if v2 == 0: continue
                                        idx_v = (j1*2 + j2)*4 + (k1*2 + k2)
                                        idx_c = c1*7 + c2
                                        V4[idx_v][idx_c] = v1 * v2

                for k1 in range(2):
                    for i1 in range(2):
                        for c1 in range(7):
                            w1 = W2[k1*2+i1][c1]
                            if w1 == 0: continue
                            for k2 in range(2):
                                for i2 in range(2):
                                    for c2 in range(7):
                                        w2 = W2[k2*2+i2][c2]
                                        if w2 == 0: continue
                                        idx_w = (k1*2 + k2)*4 + (i1*2 + i2)
                                        idx_c = c1*7 + c2
                                        W4[idx_w][idx_c] = w1 * w2

                # Now add 4x4x1 part (Rank 16)
                # This part handles the 5th column of B (index 4) and 5th row of C (index 4).
                # Actually, 4x4x5 means B is 4x5, C is 4x5.
                # We split B into B1 (4x4) and B2 (4x1).
                # C into C1 (4x4) and C2 (4x1).
                # The 4x4x1 tensor has rank 16.
                # U_16: 16x16, V_16: 4x16 (j in 0..3, k=4), W_16: 4x16 (k=4, i in 0..3).
                
                U_ext = [[0]*16 for _ in range(16)]
                V_ext = [[0]*16 for _ in range(4)] # Only j=0..3, k=4? No, V is 4x5.
                # V indices: j*5+k. For k=4, indices are 4, 9, 14, 19.
                W_ext = [[0]*16 for _ in range(4)] # W indices: k*4+i. For k=4, indices are 16, 17, 18, 19.
                
                for i in range(4):
                    for j in range(4):
                        idx = i*4 + j
                        U_ext[idx][idx] = 1
                        V_ext[j*5 + 4][idx] = 1
                        W_ext[4*4 + i][idx] = 1

                # Combine
                U = [[0]*65 for _ in range(16)]
                V = [[0]*65 for _ in range(20)]
                W = [[0]*65 for _ in range(20)]
                
                for i in range(16):
                    for c in range(49):
                        U[i][c] = U4[i][c]
                for j in range(4):
                    for k in range(4):
                        for c in range(49):
                            V[j*5+k][c] = V4[j*4+k][c]
                for k in range(4):
                    for i in range(4):
                        for c in range(49):
                            W[k*4+i][c] = W4[k*4+i][c]
                            
                for i in range(16):
                    for c in range(16):
                        U[i][49+c] = U_ext[i][c]
                for j in range(4):
                    for c in range(16):
                        V[j*5+4][49+c] = V_ext[j][c]
                for i in range(4):
                    for c in range(16):
                        W[4*4+i][49+c] = W_ext[i][c]

                # Verify
                valid = True
                for i in range(4):
                    for j in range(4):
                        for k in range(5):
                            s = 0
                            for c in range(65):
                                s += U[i*4+j][c] * V[j*5+k][c] * W[k*4+i][c]
                            if s != 1:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return U, V, W

        return None

    # Try to find a 4x4x5 decomposition with rank 65
    U5, V5, W5 = find_decomp_4x4x5(65)
    
    if U5:
        # We have a 4x4x5 decomposition with rank 65.
        # But we need 4x4x9.
        # We can split 4x4x9 into 4x4x4 and 4x4x5.
        # Rank = 49 + 65 = 114?
        # Or can we share the 4x4x4 part?
        # If we use the same 49 terms for the first 4 columns of the 4x4x5 part,
        # and the same 49 terms for the first 4 columns of the 4x4x4 part...
        # Then we only need 49 (for the core) + 16 (for the extra column in the 4x4x5 part) + 49 (for the last 5 columns of the 4x4x9 part)?
        # No, that's not right.
        
        # Let's stick to the safe 129 if we can't find a better global structure quickly.
        # But 114 is better than 129.
        
        # Let's construct the 114 rank solution.
        # Part 1: 4x4x4 (cols 0-3 of B, rows 0-3 of C). Rank 49.
        # Part 2: 4x4x5 (cols 4-8 of B, rows 4-8 of C). Rank 65.
        
        # We need to map the 4x4x5 decomposition (which assumes B is 4x5, C is 4x5)
        # to the sub-block of the 4x4x9 tensor.
        
        total_rank = 49 + 65
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(36)]
        W = [[0]*total_rank for _ in range(36)]
        
        # Fill Part 1
        # We need U4, V4, W4 again.
        # Re-use the logic from find_decomp_4x4x5?
        # Let's just re-compute U4, V4, W4 here to be safe.
        
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        U2[3][1] = 1
        V2[2][1] = 1; V2[0][1] = -1
        W2[1][1] = 1
        U2[2][2] = 1; U2[3][2] = 1
        V2[1][2] = 1
        W2[2][2] = 1
        U2[0][3] = 1
        V2[1][3] = 1; V2[3][3] = -1
        W2[0][3] = 1
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[1][4] = 1
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[2][5] = 1
        W2[0][5] = 1; W2[1][5] = 1
        U2[1][6] = 1; U2[3][6] = -1
        V2[1][6] = 1; V2[3][6] = 1
        W2[2][6] = 1; W2[3][6] = 1

        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for i1 in range(2):
            for j1 in range(2):
                for c1 in range(7):
                    u1 = U2[i1*2+j1][c1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            for c2 in range(7):
                                u2 = U2[i2*2+j2][c2]
                                if u2 == 0: continue
                                idx_u = (i1*2 + i2)*4 + (j1*2 + j2)
                                idx_c = c1*7 + c2
                                U4[idx_u][idx_c] = u1 * u2

        for j1 in range(2):
            for k1 in range(2):
                for c1 in range(7):
                    v1 = V2[j1*2+k1][c1]
                    if v1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            for c2 in range(7):
                                v2 = V2[j2*2+k2][c2]
                                if v2 == 0: continue
                                idx_v = (j1*2 + j2)*4 + (k1*2 + k2)
                                idx_c = c1*7 + c2
                                V4[idx_v][idx_c] = v1 * v2

        for k1 in range(2):
            for i1 in range(2):
                for c1 in range(7):
                    w1 = W2[k1*2+i1][c1]
                    if w1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            for c2 in range(7):
                                w2 = W2[k2*2+i2][c2]
                                if w2 == 0: continue
                                idx_w = (k1*2 + k2)*4 + (i1*2 + i2)
                                idx_c = c1*7 + c2
                                W4[idx_w][idx_c] = w1 * w2

        for i in range(16):
            for c in range(49):
                U[i][c] = U4[i][c]
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V[j*9+k][c] = V4[j*4+k][c]
        for k in range(4):
            for i in range(4):
                for c in range(49):
                    W[k*4+i][c] = W4[k*4+i][c]

        # Fill Part 2 (Offset 49)
        offset = 49
        for i in range(16):
            for c in range(65):
                U[i][offset+c] = U5[i][c]
        for j in range(4):
            for k in range(5):
                for c in range(65):
                    V[j*9 + (4+k)][offset+c] = V5[j*5+k][c]
        for k in range(5):
            for i in range(4):
                for c in range(65):
                    W[(4+k)*4 + i][offset+c] = W5[k*4+i][c]

        # Final Verification
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    s = 0
                    for c in range(total_rank):
                        s += U[i*4+j][c] * V[j*9+k][c] * W[k*4+i][c]
                    if s != 1:
                        # Fallback to 129
                        rank = 129
                        U_f = [[0] * rank for _ in range(16)]
                        V_f = [[0] * rank for _ in range(36)]
                        W_f = [[0] * rank for _ in range(36)]
                        
                        # Re-use U4, V4, W4 for first 49
                        for i in range(16):
                            for c in range(49):
                                U_f[i][c] = U4[i][c]
                        for j in range(4):
                            for k in range(4):
                                for c in range(49):
                                    V_f[j*9+k][c] = V4[j*4+k][c]
                        for k in range(4):
                            for i in range(4):
                                for c in range(49):
                                    W_f[k*4+i][c] = W4[k*4+i][c]
                        
                        # Schoolbook for 4x4x5
                        col = 49
                        for i in range(4):
                            for j in range(4):
                                for k in range(5):
                                    U_f[i*4+j][col] = 1
                                    V_f[j*9 + (4+k)][col] = 1
                                    W_f[(4+k)*4 + i][col] = 1
                                    col += 1
                        return {"U": U_f, "V": V_f, "W": W_f}
                        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to 129
        rank = 129
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        
        # Re-use Strassen 4x4x4
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        U2[3][1] = 1
        V2[2][1] = 1; V2[0][1] = -1
        W2[1][1] = 1
        U2[2][2] = 1; U2[3][2] = 1
        V2[1][2] = 1
        W2[2][2] = 1
        U2[0][3] = 1
        V2[1][3] = 1; V2[3][3] = -1
        W2[0][3] = 1
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[1][4] = 1
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[2][5] = 1
        W2[0][5] = 1; W2[1][5] = 1
        U2[1][6] = 1; U2[3][6] = -1
        V2[1][6] = 1; V2[3][6] = 1
        W2[2][6] = 1; W2[3][6] = 1

        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for i1 in range(2):
            for j1 in range(2):
                for c1 in range(7):
                    u1 = U2[i1*2+j1][c1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            for c2 in range(7):
                                u2 = U2[i2*2+j2][c2]
                                if u2 == 0: continue
                                idx_u = (i1*2 + i2)*4 + (j1*2 + j2)
                                idx_c = c1*7 + c2
                                U4[idx_u][idx_c] = u1 * u2

        for j1 in range(2):
            for k1 in range(2):
                for c1 in range(7):
                    v1 = V2[j1*2+k1][c1]
                    if v1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            for c2 in range(7):
                                v2 = V2[j2*2+k2][c2]
                                if v2 == 0: continue
                                idx_v = (j1*2 + j2)*4 + (k1*2 + k2)
                                idx_c = c1*7 + c2
                                V4[idx_v][idx_c] = v1 * v2

        for k1 in range(2):
            for i1 in range(2):
                for c1 in range(7):
                    w1 = W2[k1*2+i1][c1]
                    if w1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            for c2 in range(7):
                                w2 = W2[k2*2+i2][c2]
                                if w2 == 0: continue
                                idx_w = (k1*2 + k2)*4 + (i1*2 + i2)
                                idx_c = c1*7 + c2
                                W4[idx_w][idx_c] = w1 * w2

        for i in range(16):
            for c in range(49):
                U[i][c] = U4[i][c]
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V[j*9+k][c] = V4[j*4+k][c]
        for k in range(4):
            for i in range(4):
                for c in range(49):
                    W[k*4+i][c] = W4[k*4+i][c]

        col = 49
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j][col] = 1
                    V[j*9 + (4+k)][col] = 1
                    W[(4+k)*4 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
