import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # The tensor T[i,j,k] = 1 for all i in [0,4), j in [0,5), k in [0,5)
    # Standard rank = 100
    
    # Strategy: Partition into 4 disjoint 2x2x2 sub-tensors (Strassen, rank 7 each)
    # and one remaining 2x3x3 sub-tensor.
    # The 2x3x3 sub-tensor covers i in {2,3}, j in {0,1,2}, k in {0,1,2}.
    # We use a rank-17 decomposition for the 2x3x3 case (or better if possible).
    # Actually, let's try to construct a valid decomposition for 2x3x3.
    
    # 2x3x3 tensor T[i,j,k]=1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # This is equivalent to multiplying a 2x3 matrix by a 3x3 matrix.
    # Rank of 2x3x3 multiplication tensor is known to be 17 (Landsberg, 2012? No, that's 3x3x3=23).
    # Wait, rank of 2x3x3 is 17?
    # Standard rank is 2*3*3 = 18.
    # Can we do better?
    # Strassen works for 2x2x2 (rank 7 vs 8).
    # For 2x3x3, we can view it as a 2x2x2 block (i=0,1; j=0,1; k=0,1) plus remaining terms.
    # The 2x2x2 block has rank 7.
    # The remaining terms involve j=2 or k=2.
    # Let's just use the standard rank-18 for now and see if we can optimize, or stick to a known decomposition.
    # Actually, a simple bound: rank(2x3x3) <= rank(2x2x2) + rank(2x1x3) + rank(2x3x1) + ...
    # Let's try to construct a rank-17 decomposition for 2x3x3.
    # Or simpler: Just use 4 Strassen blocks for the 4x4x4 part? No, we have 4x5x5.
    
    # Let's stick to the prompt's suggestion: "four disjoint 2x2x2 sub-tensors and one remaining 2x3x3 sub-tensor"
    # Partition:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 (Strassen, rank 7)
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1} -> 2x2x2 (Strassen, rank 7)
    # Block 3: i in {2,3}, j in {0,1}, k in {2,3} -> 2x2x2 (Strassen, rank 7)
    # Block 4: i in {2,3}, j in {2,3}, k in {2,3} -> 2x2x2 (Strassen, rank 7)
    # Remaining:
    # i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2? No, this is disjoint from Block 1.
    # Let's list all 8 2x2x2 blocks in a 4x4x4 grid (i,j,k in 0..3):
    # (0,0,0), (0,0,2), (0,2,0), (0,2,2) for i in 0,1
    # (2,0,0), (2,0,2), (2,2,0), (2,2,2) for i in 2,3
    # That's 8 blocks of 2x2x2.
    # Plus the "fringe" involving index 4 in j and k.
    # j=4: i in 0..3, k in 0..4 -> 4x1x5
    # k=4: i in 0..3, j in 0..3 -> 4x4x1
    # j=4, k=4: i in 0..3 -> 4x1x1
    
    # Let's use the 8 Strassen blocks for the 4x4x4 core.
    # Rank = 8 * 7 = 56.
    # Remaining terms:
    # 1. j=4, k in 0..4, i in 0..3. (4x1x5 tensor). Rank 4*1*5 = 20.
    #    But we can do better.
    #    For fixed j=4, T[i,4,k]=1.
    #    This is just the vector 1_i * 1_k.
    #    We can decompose this 4x1x5 tensor.
    #    Actually, we can combine terms.
    
    # Let's try a different partition.
    # Use Strassen for the 2x2x2 blocks where possible.
    # The 4x5x5 tensor can be split into:
    # - 4 blocks of 2x2x2 (covering i=0..3, j=0..3, k=0..3)? No, that's 8 blocks.
    # - The rest involves j=4 or k=4.
    
    # Let's implement the 8 Strassen blocks for the 4x4x4 part.
    # And handle the j=4 and k=4 parts with standard rank-1 terms, but optimized.
    
    # 8 Strassen blocks:
    # i_base in [0, 2], j_base in [0, 2], k_base in [0, 2]
    # Total rank so far: 8 * 7 = 56.
    
    # Remaining entries:
    # Any (i,j,k) where j=4 or k=4.
    # Case 1: j=4, k in 0..4, i in 0..3. (20 entries)
    # Case 2: j in 0..3, k=4, i in 0..3. (16 entries)
    # Note: (i,4,4) is in both.
    
    # We can use standard rank-1 terms for these.
    # Rank = 56 + 20 + 16 - 1 (overlap at 4,4) = 91?
    # Wait, if we use standard terms for the fringe, we are back to ~92.
    # Can we optimize the fringe?
    
    # For j=4, the tensor is T[i,4,k] = 1.
    # This is a 4x5 matrix of ones.
    # We need to represent sum_r U[i,4,r] V[4,k,r] W[k,i,r] = 1.
    # If we set W[k,i,r] = 1 for all r in this block, then we need sum_r U[i,4,r] V[4,k,r] = 1.
    # This is a rank-1 approximation of a 4x5 matrix of ones? No, exact.
    # We can use 1 term: U[i,4,0]=1, V[4,k,0]=1, W[k,i,0]=1.
    # But W[k,i,0] must be consistent with other terms.
    # If we use a dedicated set of terms for the fringe, we can isolate them.
    
    # Let's use dedicated terms for the fringe.
    # Fringe 1: j=4.
    # We need sum_r U[i,4,r] V[4,k,r] W[k,i,r] = 1 for all i,k.
    # If we use 4 terms (one for each i), we can set:
    # For each i, term r_i:
    # U[i,4,r_i] = 1
    # V[4,k,r_i] = 1 for all k
    # W[k,i,r_i] = 1 for all k
    # This works. Rank 4.
    
    # Fringe 2: k=4, j in 0..3.
    # We need sum_r U[i,j,r] V[j,4,r] W[4,i,r] = 1 for all i,j.
    # Similarly, use 4 terms (one for each i? or one for each j?).
    # If we use 4 terms (one for each j), we need to be careful not to conflict with Fringe 1 terms if they share indices.
    # But we can use new columns.
    # For each j in 0..3, term r_j:
    # U[i,j,r_j] = 1 for all i
    # V[j,4,r_j] = 1
    # W[4,i,r_j] = 1 for all i
    # This works. Rank 4.
    
    # What about (i,4,4)?
    # In Fringe 1, for i, k=4: U[i,4,r_i] * V[4,4,r_i] * W[4,i,r_i] = 1 * 1 * 1 = 1.
    # In Fringe 2, for j=4? No, Fringe 2 is j in 0..3.
    # So (i,4,4) is only covered by Fringe 1.
    # Wait, Fringe 2 is for k=4, j in 0..3. So (i,j,4) for j<4.
    # So (i,4,4) is covered by Fringe 1 (j=4, k=4).
    # Is it covered by Fringe 2? No, j=4 is not in 0..3.
    # So no double counting.
    
    # Total Rank = 56 (Strassen) + 4 (Fringe 1) + 4 (Fringe 2) = 64.
    # This is much better than 92.
    
    rank = 64
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    col = 0
    
    # 8 Strassen blocks for 4x4x4
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                i1, i2 = i_base, i_base + 1
                j1, j2 = j_base, j_base + 1
                k1, k2 = k_base, k_base + 1
                
                # M1 = (a11 + a22)(b11 + b22)
                U[i1 * m + j1, col] += 1
                U[i2 * m + j2, col] += 1
                V[j1 * p + k1, col] += 1
                V[j2 * p + k2, col] += 1
                W[k1 * n + i1, col] += 1
                W[k2 * n + i2, col] += 1
                col += 1
                
                # M2 = (a21 + a22)b11
                U[i2 * m + j1, col] += 1
                U[i2 * m + j2, col] += 1
                V[j1 * p + k1, col] += 1
                W[k1 * n + i2, col] += 1
                W[k2 * n + i2, col] += -1
                col += 1
                
                # M3 = a11(b12 - b22)
                U[i1 * m + j1, col] += 1
                V[j1 * p + k2, col] += 1
                V[j2 * p + k2, col] += -1
                W[k2 * n + i1, col] += 1
                W[k2 * n + i2, col] += 1
                col += 1
                
                # M4 = a22(b21 - b11)
                U[i2 * m + j2, col] += 1
                V[j2 * p + k1, col] += 1
                V[j1 * p + k1, col] += -1
                W[k1 * n + i1, col] += 1
                W[k1 * n + i2, col] += 1
                col += 1
                
                # M5 = (a11 + a12)b22
                U[i1 * m + j1, col] += 1
                U[i1 * m + j2, col] += 1
                V[j2 * p + k2, col] += 1
                W[k1 * n + i1, col] += -1
                W[k2 * n + i1, col] += 1
                col += 1
                
                # M6 = (a21 - a11)(b11 + b12)
                U[i2 * m + j1, col] += 1
                U[i1 * m + j1, col] += -1
                V[j1 * p + k1, col] += 1
                V[j1 * p + k2, col] += 1
                W[k2 * n + i2, col] += 1
                col += 1
                
                # M7 = (a12 - a22)(b21 + b22)
                U[i1 * m + j2, col] += 1
                U[i2 * m + j2, col] += -1
                V[j2 * p + k1, col] += 1
                V[j2 * p + k2, col] += 1
                W[k1 * n + i1, col] += 1
                col += 1
    
    # Fringe 1: j=4
    # For each i in 0..3, one term
    for i in range(n):
        # U[i, 4, col] = 1
        U[i * m + 4, col] = 1
        # V[4, k, col] = 1 for all k
        for k in range(p):
            V[4 * p + k, col] = 1
        # W[k, i, col] = 1 for all k
        for k in range(p):
            W[k * n + i, col] = 1
        col += 1
    
    # Fringe 2: k=4, j in 0..3
    # For each j in 0..3, one term
    for j in range(4):
        # U[i, j, col] = 1 for all i
        for i in range(n):
            U[i * m + j, col] = 1
        # V[j, 4, col] = 1
        V[j * p + 4, col] = 1
        # W[4, i, col] = 1 for all i
        for i in range(n):
            W[4 * n + i, col] = 1
        col += 1
    
    # Verify correctness
    if time.time() < deadline - 5:
        for test_i in range(n):
            for test_j in range(m):
                for test_k in range(p):
                    expected = 1
                    actual = 0
                    for r in range(rank):
                        actual += U[test_i * m + test_j, r] * V[test_j * p + test_k, r] * W[test_k * n + test_i, r]
                    if actual != expected:
                        # Fallback: return standard rank-100 decomposition
                        rank_std = 100
                        U_std = np.zeros((nm, rank_std), dtype=np.int64)
                        V_std = np.zeros((mp, rank_std), dtype=np.int64)
                        W_std = np.zeros((pn, rank_std), dtype=np.int64)
                        c = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    U_std[i*m+j, c] = 1
                                    V_std[j*p+k, c] = 1
                                    W_std[k*n+i, c] = 1
                                    c += 1
                        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
