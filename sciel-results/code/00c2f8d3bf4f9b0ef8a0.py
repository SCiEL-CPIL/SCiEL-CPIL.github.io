import numpy as np
import time

DEADLINE_SECONDS = 145.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def construct():
    n, m, p = 2, 3, 9
    # Use block structure: 9 = 3*3 columns of B
    # For each 3-column block t (t=0,1,2), compute A (2x3) * B[:, 3t:3t+3] (3x3)
    # The 2x3 x 3x3 multiplication tensor has rank 12
    # Total rank = 3 * 12 = 36
    
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 27 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 18 rows
    
    # For 2x3 * 3x3 matmul: C[i,k] = sum_j A[i,j]*B[j,k]
    # Rank-12 decomposition using the following construction:
    # We use 12 bilinear multiplications.
    # 
    # Construction: For the 2x3 x 3x3 tensor, we can achieve rank 12 by:
    # For each pair (i,k), we need sum_j A[i,j]*B[j,k].
    # Instead of 18 terms, we use the following 12-term decomposition:
    #
    # Let's define 12 terms. The key is that we can share some structure.
    # Actually, let me use a verified rank-12 construction.
    #
    # One approach: note that 2x3 * 3x3 = (2x3 * 3x1) repeated 3 times for columns.
    # 2x3 * 3x1 has rank 4 (not 3, because we need 2 outputs each being a dot product).
    # Wait: 2x3 * 3x1: C[0]=A[0,0]B[0]+A[0,1]B[1]+A[0,2]B[2], C[1]=A[1,0]B[0]+A[1,1]B[1]+A[1,2]B[2]
    # This is 2 outputs, 3 inputs each side. Bilinear rank: at most 6, but can we do 4?
    # 
    # Actually, the bilinear rank of 2x3 * 3x1 is 4. Here's how:
    # m0 = A[0,0]*B[0]
    # m1 = A[0,1]*B[1]  
    # m2 = A[0,2]*B[2]
    # m3 = (A[1,0]-A[0,0])*B[0] + (A[1,1]-A[0,1])*B[1] + (A[1,2]-A[0,2])*B[2]
    # Then C[0] = m0+m1+m2, C[1] = m0+m1+m2+m3
    # But m3 is a single bilinear form? No, it's a sum of 3 products, so it's not rank 1.
    #
    # I think I'm overcomplicating this. Let me just use rank 18 per block (schoolbook)
    # giving total 54, but with a cleaner implementation. Or better, let me try to find
    # the actual minimum.
    #
    # After reflection: the rank of M_{2,3,3} is actually 12. Here's a construction:
    # Use the fact that we can write the 3x3 part using 4 multiplications (Strassen-like)
    # but adapted for 2x3 * 3x3.
    #
    # Let me just use a safe rank and verify numerically.
    # I'll use rank 36 with a specific construction.
    
    # Construction for 2x3 * 3x3 with rank 12:
    # We have A (2x3), B (3x3), C (2x3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # 12-term decomposition:
    # For j=0: use 4 terms to handle B[0,:] contributions to both rows
    # For j=1: use 4 terms
    # For j=2: use 4 terms
    # 
    # For each j, we need to compute:
    #   C[0,:] += A[0,j] * B[j,:]
    #   C[1,:] += A[1,j] * B[j,:]
    # 
    # This is: for fixed j, we have a 2x3 * 1x3 "product" (outer product of A[:,j] and B[j,:])
    # The rank of this 2x1 * 1x3 = 2x3 outer product is 2 (it's just a[0]*b[0], a[0]*b[1], a[0]*b[2], a[1]*b[0], a[1]*b[1], a[1]*b[2])
    # Wait, that's 6 entries but only 2 independent bilinear forms: a[0]*b[k] and a[1]*b[k] for k=0,1,2.
    # So for each j, we need 2*3=6 rank-1 terms? No, we need 2 bilinear multiplications:
    # m_j0 = A[0,j] * B[j,k] for each k... but B[j,k] varies with k.
    #
    # I think the issue is that B[j,k] is a different input for each k, so we can't share.
    # For fixed j: we have inputs A[0,j], A[1,j] (2 values) and B[j,0], B[j,1], B[j,2] (3 values).
    # Outputs: C[0,k] gets A[0,j]*B[j,k], C[1,k] gets A[1,j]*B[j,k] for k=0,1,2.
    # That's 6 outputs, each a product of one A-value and one B-value.
    # The bilinear rank of this sub-problem is 6 (each output is a distinct product).
    # So for 3 values of j, total = 18. That's schoolbook.
    #
    # I don't see an easy way to reduce below 54 without a clever construction.
    # Let me just implement schoolbook rank 54 cleanly.
    
    R = n * m * p  # 54
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][idx] = 1.0
                V[v_idx][idx] = 1.0
                W[w_idx][idx] = 1.0
                idx += 1
    
    return {"U": U, "V": V, "W": W}
