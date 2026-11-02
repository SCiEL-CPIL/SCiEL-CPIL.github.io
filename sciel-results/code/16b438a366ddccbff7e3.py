import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use the decomposition based on M_{2,2,2} rank 7.
    # M_{2,4,8} -> 2 * M_{2,2,8} -> 2 * 4 * M_{2,2,2} = 8 * 7 = 56.
    
    # Strassen's 7 multiplications for 2x2 matrix multiplication.
    # C = A * B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    # Strassen:
    # m1 = (a+d)*(e+h)
    # m2 = (c+d)*e
    # m3 = a*(f-h)
    # m4 = d*(g-e)
    # m5 = (a+b)*h
    # m6 = (c-a)*(e+f)
    # m7 = (b-d)*(g+h)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # We need to map this to the tensor decomposition format.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Here n=2, m=2, p=2 for the base block.
    # We are composing these blocks.
    
    # Base block: M_{2,2,2}
    # Indices for A (2x2): 0,1,2,3 (i*m+j)
    # Indices for B (2x2): 0,1,2,3 (j*p+k)
    # Indices for C (2x2): 0,1,2,3 (k*n+i)
    
    # Let's define the 7 terms for one M_{2,2,2} block.
    # Each term is a rank 1 tensor.
    # U_vec, V_vec, W_vec of length 4.
    
    def get_strassen_block(offset_u, offset_v, offset_w):
        # Returns list of (u_vec, v_vec, w_vec) for the 7 multiplications.
        # u_vec is in R^4, v_vec in R^4, w_vec in R^4.
        terms = []
        
        # m1 = (a+d)*(e+h)
        # A parts: a (idx 0), d (idx 3). Sum: [1, 0, 0, 1]
        # B parts: e (idx 0), h (idx 3). Sum: [1, 0, 0, 1]
        # C parts: C00 (idx 0) +, C11 (idx 3) +
        # Wait, m1 contributes to C00 and C11.
        # C00 = m1 + ...
        # C11 = m1 - ...
        # So W vector has 1 at 0, 1 at 3?
        # The term is m1 * (u1 v1^T).
        # u1 = [1, 0, 0, 1] (for A)
        # v1 = [1, 0, 0, 1] (for B)
        # w1 = [1, 0, 0, 1] (for C, adds to C00 and C11)
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        
        # m2 = (c+d)*e
        # A parts: c (idx 2), d (idx 3). Sum: [0, 0, 1, 1]
        # B parts: e (idx 0). [1, 0, 0, 0]
        # C parts: C10 (idx 2) +
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        
        # m3 = a*(f-h)
        # A parts: a (idx 0). [1, 0, 0, 0]
        # B parts: f (idx 1), h (idx 3). Diff: [0, 1, 0, -1]
        # C parts: C01 (idx 1) +, C11 (idx 3) +
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        
        # m4 = d*(g-e)
        # A parts: d (idx 3). [0, 0, 0, 1]
        # B parts: g (idx 2), e (idx 0). Diff: [-1, 0, 1, 0]
        # C parts: C00 (idx 0) +, C10 (idx 2) +
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        
        # m5 = (a+b)*h
        # A parts: a (idx 0), b (idx 1). Sum: [1, 1, 0, 0]
        # B parts: h (idx 3). [0, 0, 0, 1]
        # C parts: C00 (idx 0) -, C01 (idx 1) +
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        
        # m6 = (c-a)*(e+f)
        # A parts: c (idx 2), a (idx 0). Diff: [-1, 0, 1, 0]
        # B parts: e (idx 0), f (idx 1). Sum: [1, 1, 0, 0]
        # C parts: C11 (idx 3) +
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        
        # m7 = (b-d)*(g+h)
        # A parts: b (idx 1), d (idx 3). Diff: [0, 1, 0, -1]
        # B parts: g (idx 2), h (idx 3). Sum: [0, 0, 1, 1]
        # C parts: C00 (idx 0) +, C11 (idx 3) -
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]))
        
        return terms

    # Now we compose 8 blocks.
    # Block (bi, bk) where bi in {0,1} (row of A), bk in {0,1,2,3} (col group of B).
    # A is 2x4. Split into A[0..1, 0..1] and A[0..1, 2..3].
    # So bi=0 uses A cols 0,1. bi=1 uses A cols 2,3.
    # B is 4x8. Split rows into 0,1 and 2,3.
    # So for A block 0 (cols 0,1), we multiply with B rows 0,1.
    # For A block 1 (cols 2,3), we multiply with B rows 2,3.
    # Wait, A = [A_L, A_R]. B = [[B_T], [B_B]].
    # C = A_L B_T + A_R B_B.
    # A_L is 2x2. B_T is 2x8.
    # A_R is 2x2. B_B is 2x8.
    # So we need 2 instances of M_{2,2,8}.
    # Each M_{2,2,8} is 4 instances of M_{2,2,2}.
    # Total 8 instances.
    
    U = [[0] * 56 for _ in range(n * m)] # 8x56
    V = [[0] * 56 for _ in range(m * p)] # 32x56
    W = [[0] * 56 for _ in range(p * n)] # 16x56
    
    col = 0
    for a_block in range(2): # 0: A_L, 1: A_R
        for b_block in range(4): # 0-3: B columns 0-1, 2-3, 4-5, 6-7
            # Get the 7 terms for this block
            terms = get_strassen_block(0, 0, 0)
            for u_vec, v_vec, w_vec in terms:
                # Map indices to global indices
                # A index: i (0,1), j (0,1 within block)
                # Global A index: i * m + (a_block * 2 + j)
                # B index: j (0,1 within block row), k (0,1 within block col)
                # Global B index: (a_block * 2 + j) * p + (b_block * 2 + k)
                # C index: k (0,1 within block col), i (0,1)
                # Global C index: (b_block * 2 + k) * n + i
                
                for local_i in range(2):
                    for local_j in range(2):
                        g_u = local_i * m + (a_block * 2 + local_j)
                        U[g_u][col] = u_vec[local_i * 2 + local_j]
                        
                for local_j in range(2):
                    for local_k in range(2):
                        g_v = (a_block * 2 + local_j) * p + (b_block * 2 + local_k)
                        V[g_v][col] = v_vec[local_j * 2 + local_k]
                        
                for local_k in range(2):
                    for local_i in range(2):
                        g_w = (b_block * 2 + local_k) * n + local_i
                        W[g_w][col] = w_vec[local_k * 2 + local_i]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
