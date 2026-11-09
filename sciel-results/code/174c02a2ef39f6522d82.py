import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # We want to find a rank-r decomposition:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Use Strassen's 7-term decomposition for 2x2x2 blocks.
    # For a 4x5x5 tensor, we can partition the i-dimension into two 2-blocks,
    # the j-dimension into two 2-blocks + one 1-block, and k-dimension into two 2-blocks + one 1-block.
    # 
    # Actually, let's think more carefully. The tensor is 4x5x5.
    # We can use the following partition:
    # - i: {0,1}, {2,3} (two blocks of size 2)
    # - j: {0,1}, {2,3}, {4} (two blocks of size 2, one block of size 1)
    # - k: {0,1}, {2,3}, {4} (two blocks of size 2, one block of size 1)
    #
    # For each combination of blocks (i_block, j_block, k_block), we have a sub-tensor.
    # If all three blocks have size 2, we can use Strassen (rank 7 instead of 8).
    # Otherwise, we use the standard method (rank = size_i * size_j * size_k).
    #
    # Let's enumerate:
    # i_blocks: [2, 2]
    # j_blocks: [2, 2, 1]
    # k_blocks: [2, 2, 1]
    #
    # Number of block combinations: 2 * 3 * 3 = 18
    # For each combination, the sub-tensor size is (si, sj, sk) where si in {2}, sj in {2,2,1}, sk in {2,2,1}
    # 
    # If si=2, sj=2, sk=2: rank 7 (Strassen)
    # If si=2, sj=2, sk=1: rank 4 (standard: 2*2*1)
    # If si=2, sj=1, sk=2: rank 4 (standard: 2*1*2)
    # If si=2, sj=1, sk=1: rank 2 (standard: 2*1*1)
    #
    # Count:
    # (2,2,2): 2 * 2 * 2 = 8 blocks, each rank 7 -> 56
    # (2,2,1): 2 * 2 * 1 = 4 blocks, each rank 4 -> 16
    # (2,1,2): 2 * 1 * 2 = 4 blocks, each rank 4 -> 16
    # (2,1,1): 2 * 1 * 1 = 2 blocks, each rank 2 -> 4
    # Total: 56 + 16 + 16 + 4 = 92
    #
    # Wait, that's the same as before. Let me recount.
    # i_blocks: 2 blocks of size 2
    # j_blocks: 2 blocks of size 2, 1 block of size 1
    # k_blocks: 2 blocks of size 2, 1 block of size 1
    #
    # (2,2,2): 2 * 2 * 2 = 8 combinations, each rank 7 -> 56
    # (2,2,1): 2 * 2 * 1 = 4 combinations, each rank 4 -> 16
    # (2,1,2): 2 * 1 * 2 = 4 combinations, each rank 4 -> 16
    # (2,1,1): 2 * 1 * 1 = 2 combinations, each rank 2 -> 4
    # Total: 56 + 16 + 16 + 4 = 92
    #
    # So the total rank is 92, same as before. But let's implement it properly.
    
    def build_decomposition():
        # Define blocks
        i_blocks = [[0, 1], [2, 3]]
        j_blocks = [[0, 1], [2, 3], [4]]
        k_blocks = [[0, 1], [2, 3], [4]]
        
        # Collect all terms
        terms = []  # Each term is (U_entry, V_entry, W_entry)
        
        for I in i_blocks:
            for J in j_blocks:
                for K in k_blocks:
                    si, sj, sk = len(I), len(J), len(K)
                    
                    if si == 2 and sj == 2 and sk == 2:
                        # Use Strassen
                        i1, i2 = I
                        j1, j2 = J
                        k1, k2 = K
                        
                        # M1 = (a11 + a22)(b11 + b22)
                        terms.append({
                            'U': [(i1*m+j1, 1), (i2*m+j2, 1)],
                            'V': [(j1*p+k1, 1), (j2*p+k2, 1)],
                            'W': [(k1*n+i1, 1), (k2*n+i2, 1)]
                        })
                        
                        # M2 = (a21 + a22)b11
                        terms.append({
                            'U': [(i2*m+j1, 1), (i2*m+j2, 1)],
                            'V': [(j1*p+k1, 1)],
                            'W': [(k1*n+i2, 1), (k2*n+i2, -1)]
                        })
                        
                        # M3 = a11(b12 - b22)
                        terms.append({
                            'U': [(i1*m+j1, 1)],
                            'V': [(j1*p+k2, 1), (j2*p+k2, -1)],
                            'W': [(k2*n+i1, 1), (k2*n+i2, 1)]
                        })
                        
                        # M4 = a22(b21 - b11)
                        terms.append({
                            'U': [(i2*m+j2, 1)],
                            'V': [(j2*p+k1, 1), (j1*p+k1, -1)],
                            'W': [(k1*n+i1, 1), (k1*n+i2, 1)]
                        })
                        
                        # M5 = (a11 + a12)b22
                        terms.append({
                            'U': [(i1*m+j1, 1), (i1*m+j2, 1)],
                            'V': [(j2*p+k2, 1)],
                            'W': [(k1*n+i1, -1), (k2*n+i1, 1)]
                        })
                        
                        # M6 = (a21 - a11)(b11 + b12)
                        terms.append({
                            'U': [(i2*m+j1, 1), (i1*m+j1, -1)],
                            'V': [(j1*p+k1, 1), (j1*p+k2, 1)],
                            'W': [(k2*n+i2, 1)]
                        })
                        
                        # M7 = (a12 - a22)(b21 + b22)
                        terms.append({
                            'U': [(i1*m+j2, 1), (i2*m+j2, -1)],
                            'V': [(j2*p+k1, 1), (j2*p+k2, 1)],
                            'W': [(k1*n+i1, 1)]
                        })
                    else:
                        # Standard method
                        for i in I:
                            for j in J:
                                for k in K:
                                    terms.append({
                                        'U': [(i*m+j, 1)],
                                        'V': [(j*p+k, 1)],
                                        'W': [(k*n+i, 1)]
                                    })
        
        # Build matrices
        rank = len(terms)
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        for col, term in enumerate(terms):
            for idx, val in term['U']:
                U[idx][col] += val
            for idx, val in term['V']:
                V[idx][col] += val
            for idx, val in term['W']:
                W[idx][col] += val
        
        return U, V, W, rank
    
    U, V, W, rank = build_decomposition()
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is correct."""
        nm, mp, pn = n*m, m*p, p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to standard decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for potential further optimization
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_rank = rank
    best_rank = current_rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try to find reducible terms using linear dependency search
    # For each column, check if it's a linear combination of others in the
    # tensor-product sense. This is expensive, so limit to a few attempts.
    
    # Quick check: are there any all-zero columns?
    for r in range(current_rank):
        if all(U_np[i][r] == 0 for i in range(nm)) and \
           all(V_np[i][r] == 0 for i in range(mp)) and \
           all(W_np[i][r] == 0 for i in range(pn)):
            # Remove this column
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            current_rank -= 1
            r -= 1
    
    if time.time() > deadline:
        # Return what we have
        U_final = [[int(round(U_np[i][r])) if abs(U_np[i][r] - round(U_np[i][r])) < 1e-9 else U_np[i][r] 
                    for r in range(current_rank)] for i in range(nm)]
        V_final = [[int(round(V_np[i][r])) if abs(V_np[i][r] - round(V_np[i][r])) < 1e-9 else V_np[i][r] 
                    for r in range(current_rank)] for i in range(mp)]
        W_final = [[int(round(W_np[i][r])) if abs(W_np[i][r] - round(W_np[i][r])) < 1e-9 else W_np[i][r] 
                    for r in range(current_rank)] for i in range(pn)]
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Convert back to lists with integer/half-integer entries
    U_final = []
    for i in range(nm):
        row = []
        for r in range(current_rank):
            val = U_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        U_final.append(row)
    
    V_final = []
    for i in range(mp):
        row = []
        for r in range(current_rank):
            val = V_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        V_final.append(row)
    
    W_final = []
    for i in range(pn):
        row = []
        for r in range(current_rank):
            val = W_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        W_final.append(row)
    
    return {"U": U_final, "V": V_final, "W": W_final}
