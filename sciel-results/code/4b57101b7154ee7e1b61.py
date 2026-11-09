"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing a rank-r decomposition
    of the matrix multiplication tensor for 4x4 * 4x5.
    
    The decomposition satisfies:
      For all i in [0,n), j in [0,m), k in [0,p):
        sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    where n=4, m=4, p=5.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Strategy: Use Winograd-style block decomposition.
    # For 2x2 * 2x2 matrix multiplication, Winograd's algorithm achieves rank 7.
    # For 4x4 * 4x5, we can decompose into blocks and apply Winograd where possible.
    
    # Approach: Split the 4x4 * 4x5 multiplication into:
    #   - 4x4 * 4x4 part: use 4 blocks of 2x2*2x2, each with rank 7 → 4*7 = 28
    #   - 4x4 * 4x1 part (5th column): use schoolbook → 16
    #   Total: 28 + 16 = 44
    #
    # But we can do better by overlapping blocks or using a more global approach.
    # Let's try a direct construction.
    
    # Actually, let's use a systematic approach:
    # We'll construct the decomposition using the known Winograd 2x2 algorithm
    # and extend it to 4x4 * 4x5.
    
    # Winograd 2x2 * 2x2:
    # Given A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # Compute:
    #   m0 = (a0+a1)*(b0+b2)
    #   m1 = (a0-a1)*b0
    #   m2 = a1*b2
    #   m3 = (a2+a3)*(b1+b3)
    #   m4 = (a2-a3)*b1
    #   m5 = a3*b3
    #   m6 = (a0+a3)*b2
    # Then:
    #   C00 = m0 + m1 + m2
    #   C01 = m3 + m4 - m6
    #   C10 = m3 + m5 + m6
    #   C11 = m0 - m4 - m5
    #
    # In tensor form, the 7 rank-1 terms have coefficients:
    # For term t, U entries (for A indices), V entries (for B indices), W entries (for C indices)
    
    # Let's build the full 4x4 * 4x5 decomposition.
    # We'll index A as A[i,j] for i=0..3, j=0..3
    # B as B[j,k] for j=0..3, k=0..4
    # C as C[i,k] for i=0..3, k=0..4
    
    # Split into 2x2 blocks:
    # A has blocks: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # B has blocks: B[0:2,0:2], B[0:2,2:4], B[0:2,4:5], B[2:4,0:2], B[2:4,2:4], B[2:4,4:5]
    # C has blocks: C[0:2,0:2], C[0:2,2:4], C[0:2,4:5], C[2:4,0:2], C[2:4,2:4], C[2:4,4:5]
    
    # For the 4x4 * 4x4 part (k=0..3), we have 4 blocks of 2x2*2x2, each rank 7 → 28
    # For the 4x4 * 4x1 part (k=4), we need to handle it separately.
    
    # Let's use a cleaner approach: direct enumeration with Winograd on 2x2 blocks.
    
    # Initialize factor matrices
    # We'll determine the rank first by building the decomposition
    
    # Build the decomposition using Winograd for 2x2 blocks
    # For each 2x2 * 2x2 block, we add 7 rank-1 terms
    
    # Let's track the columns (rank terms)
    U_rows = n * m  # 16
    V_rows = m * p  # 20
    W_rows = p * n  # 20
    
    # Start with schoolbook to have a baseline, then try to optimize
    rank_schoolbook = n * m * p  # 80
    
    # Try to build a better decomposition
    # Using Winograd 2x2 blocks:
    
    # For the 4x4 * 4x4 sub-problem:
    # Blocks of A: (i0,i1) x (j0,j1) where i0 in {0,2}, j0 in {0,2}
    # Blocks of B: (j0,j1) x (k0,k1) where j0 in {0,2}, k0 in {0,2}
    # Blocks of C: (i0,i1) x (k0,k1) where i0 in {0,2}, k0 in {0,2}
    
    # For each combination of (i_block, j_block, k_block) where:
    #   i_block in {0, 2}, j_block in {0, 2}, k_block in {0, 2}
    # We have a 2x2 * 2x2 multiplication with rank 7.
    
    # That's 2*2*2 = 8 blocks, each rank 7 → 56 for the 4x4*4x4 part.
    # For the 4x4 * 4x1 part (k=4), we need 16 terms (schoolbook).
    # Total: 56 + 16 = 72.
    
    # Hmm, that's still less than 80 but not by much. Let me reconsider.
    
    # Actually, the 4x4*4x4 part has 4x4x4 = 64 scalar multiplications in schoolbook.
    # With Winograd 2x2, we have 4 blocks (in i-k plane) each with 2x2*2x2.
    # Wait, I need to be more careful.
    
    # The multiplication C = A * B where A is 4x4, B is 4x5.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's partition:
    # i into {0,1} and {2,3}
    # j into {0,1} and {2,3}  
    # k into {0,1}, {2,3}, {4}
    
    # For k in {0,1}: B[:, 0:2] is 4x2
    # For k in {2,3}: B[:, 2:4] is 4x2
    # For k = 4: B[:, 4] is 4x1
    
    # For the k in {0,1} part:
    #   C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    #   C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    # Each of these is a sum of two 2x2*2x2 products.
    # Using Winograd, each 2x2*2x2 has rank 7, so each sum has rank at most 14.
    # But we can share terms? Not easily.
    
    # Let me just build it directly.
    
    # I'll construct the decomposition term by term.
    # Each term t has:
    #   U[i*m+j, t] for i in 0..3, j in 0..3
    #   V[j*p+k, t] for j in 0..3, k in 0..4
    #   W[k*n+i, t] for k in 0..4, i in 0..3
    
    # Winograd 2x2 * 2x2 rank-7 terms:
    # For A block [[a00, a01], [a10, a11]] and B block [[b00, b01], [b10, b11]]:
    # Term 0: U: a00+a01 → positions (0,0),(0,1) get 1; V: b00+b10 → positions (0,0),(1,0) get 1; W: C00+=1, C11+=1 → (0,0),(1,1) get 1
    # Wait, let me be precise.
    
    # Winograd 2x2:
    # m0 = (a00+a01)*(b00+b10) → contributes to C00 and C11
    # m1 = (a00-a01)*b00 → contributes to C00
    # m2 = a01*b10 → contributes to C00
    # m3 = (a10+a11)*(b01+b11) → contributes to C01 and C10
    # m4 = (a10-a11)*b01 → contributes to C01
    # m5 = a11*b11 → contributes to C11
    # m6 = (a00+a11)*b10 → contributes to C01 (negative) and C10 (positive)
    
    # More carefully:
    # C00 = m0 + m1 + m2
    # C01 = m3 + m4 - m6
    # C10 = m3 + m5 + m6
    # C11 = m0 - m4 - m5
    
    # So the 7 rank-1 terms for a 2x2*2x2 block are:
    # Term 0: U = [1,1,0,0] (for a00,a01,a10,a11), V = [1,0,1,0] (for b00,b01,b10,b11), W = [1,0,0,1] (for C00,C01,C10,C11)
    # Term 1: U = [1,-1,0,0], V = [1,0,0,0], W = [1,0,0,0]
    # Term 2: U = [0,1,0,0], V = [0,0,1,0], W = [1,0,0,0]
    # Term 3: U = [0,0,1,1], V = [0,1,0,1], W = [0,1,1,0]
    # Term 4: U = [0,0,1,-1], V = [0,1,0,0], W = [0,1,0,0]
    # Term 5: U = [0,0,0,1], V = [0,0,0,1], W = [0,0,0,1]
    # Term 6: U = [1,0,0,1], V = [0,0,1,0], W = [0,-1,1,0]
    
    # Let me verify term 0: (a00+a01)*(b00+b10) = a00*b00 + a00*b10 + a01*b00 + a01*b10
    # C00 gets: a00*b00 (correct), a01*b00 (should be part of m1+m2), a00*b10 (part of m6), a01*b10 (m2)
    # This is getting complicated. Let me just use the standard formulation.
    
    # Standard Winograd 2x2*2x2:
    # m0 = (a00+a01)*(b00+b10)
    # m1 = (a00-a01)*b00  
    # m2 = a01*b10
    # m3 = (a10+a11)*(b01+b11)
    # m4 = (a10-a11)*b01
    # m5 = a11*b11
    # m6 = (a00+a11)*b10
    
    # C00 = m0 + m1 + m2
    # C01 = m3 + m4 - m6
    # C10 = m3 + m5 + m6
    # C11 = m0 - m4 - m5
    
    # Rank-1 terms (U, V, W) for each m:
    # m0: U=[1,1,0,0], V=[1,0,1,0], W=[1,0,0,1]
    # m1: U=[1,-1,0,0], V=[1,0,0,0], W=[1,0,0,0]
    # m2: U=[0,1,0,0], V=[0,0,1,0], W=[1,0,0,0]
    # m3: U=[0,0,1,1], V=[0,1,0,1], W=[0,1,1,0]
    # m4: U=[0,0,1,-1], V=[0,1,0,0], W=[0,1,0,-1]
    # m5: U=[0,0,0,1], V=[0,0,0,1], W=[0,0,0,1]
    # m6: U=[1,0,0,1], V=[0,0,1,0], W=[0,-1,1,0]
    
    # Now for the full 4x4 * 4x5:
    # We'll partition into 2x2 blocks in i and j, and handle k separately.
    
    # Let's build the decomposition:
    # For k in {0,1,2,3}: use Winograd 2x2 blocks
    # For k = 4: use schoolbook
    
    # For k in {0,1}: 
    #   C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    #   C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    # Each 2x2*2x2 product uses 7 terms, so 4 products * 7 = 28 terms for k in {0,1}
    
    # Similarly for k in {2,3}: another 28 terms
    # For k = 4: 16 terms (schoolbook)
    # Total: 28 + 28 + 16 = 72
    
    # Let's build this.
    
    U = [[0] * 72 for _ in range(U_rows)]
    V = [[0] * 72 for _ in range(V_rows)]
    W = [[0] * 72 for _ in range(W_rows)]
    
    col = 0
    
    # Winograd 2x2*2x2 terms (U, V, W) as lists of 4 elements each
    winograd_terms = [
        ([1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]),   # m0
        ([1, -1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),   # m1
        ([0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]),    # m2
        ([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]),    # m3
        ([0, 0, 1, -1], [0, 1, 0, 0], [0, 1, 0, -1]),  # m4
        ([0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]),    # m5
        ([1, 0, 0, 1], [0, 0, 1, 0], [0, -1, 1, 0]),   # m6
    ]
    
    # For each block combination:
    # i_block in {0, 2}, j_block in {0, 2}, k_block in {0, 2}
    # This gives 8 blocks of 2x2*2x2 for the 4x4*4x4 part
    
    for i_block in [0, 2]:
        for j_block in [0, 2]:
            for k_block in [0, 2]:
                for (u_vals, v_vals, w_vals) in winograd_terms:
                    if col >= 72:
                        break
                    # U: A[i_block: i_block+2, j_block: j_block+2]
                    for di in range(2):
                        for dj in range(2):
                            i = i_block + di
                            j = j_block + dj
                            U[i * m + j][col] += u_vals[di * 2 + dj]
                    
                    # V: B[j_block: j_block+2, k_block: k_block+2]
                    for dj in range(2):
                        for dk in range(2):
                            j = j_block + dj
                            k = k_block + dk
                            V[j * p + k][col] += v_vals[dj * 2 + dk]
                    
                    # W: C[i_block: i_block+2, k_block: k_block+2]
                    for di in range(2):
                        for dk in range(2):
                            i = i_block + di
                            k = k_block + dk
                            W[k * n + i][col] += w_vals[di * 2 + dk]
                    
                    col += 1
    
    # Now handle k = 4 (the 5th column) with schoolbook
    for i in range(n):
        for j in range(m):
            if col >= 72:
                break
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Verify the decomposition
    # Check that for all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape for easier checking
    U_reshaped = U_np.reshape(n, m, -1)  # [i, j, t]
    V_reshaped = V_np.reshape(m, p, -1)  # [j, k, t]
    W_reshaped = W_np.reshape(p, n, -1)  # [k, i, t]
    
    # Check each (i, j, k)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
