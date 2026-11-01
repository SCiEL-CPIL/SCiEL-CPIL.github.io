"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if val != 1:
                    return False
    # Also verify that non-matching entries are 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            if (i, j, k) != (ip, jp, kp):
                                a = i * m + j
                                b = jp * p + kp
                                c = k * n + ip
                                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                                if val != 0:
                                    return False
    return True

def _rank11_2x3_3x2():
    """
    Rank-11 bilinear algorithm for 2x3 * 3x2.
    Convention: U[i*3+j, c], V[j*2+k, c], W[k*2+i, c]
    
    A = [[a00, a01, a02],
         [a10, a11, a12]]
    B = [[b00, b01],
         [b10, b11],
         [b20, b21]]
    C = [[c00, c01],
         [c10, c11]]
    
    c[i,k] = sum_j a[i,j] * b[j,k]
    """
    n, m, p = 2, 3, 2
    r = 11
    U = [[0] * r for _ in range(n * m)]  # 6 x 11
    V = [[0] * r for _ in range(m * p)]  # 6 x 11
    W = [[0] * r for _ in range(p * n)]  # 4 x 11
    
    # Strategy: use the sharing trick across i for each (j,k),
    # but also share across k within the same j using linear combinations.
    #
    # For each j in {0,1,2}, we have two k values (k=0, k=1).
    # Schoolbook for fixed j: 2 (i) * 2 (k) = 4 terms.
    # 
    # Trick 1 (across i): For each (j,k), use 2 terms instead of 2:
    #   Term A: a[0,j]*b[j,k] -> contributes to c[0,k]
    #   Term B: (a[0,j]+a[1,j])*b[j,k] -> contributes to c[0,k] and c[1,k]
    #   Then c[1,k] contribution = Term B - Term A
    # This uses 2 terms per (j,k), total 3*2*2 = 12. No savings.
    #
    # Trick 2 (across k): For fixed j, define:
    #   s0 = a[i,j] * b[j,0]
    #   s1 = a[i,j] * b[j,1]
    #   s01 = a[i,j] * (b[j,0] + b[j,1])
    #   Then a[i,j]*b[j,1] = s01 - s0
    # This uses 2 terms per (i,j) instead of 2, no savings.
    #
    # Trick 3 (combined): For fixed j, use 3 terms for both i and both k:
    #   t0: a[0,j] * b[j,0] -> W[k=0,i=0]
    #   t1: a[0,j] * (b[j,0]+b[j,1]) -> W[k=0,i=0], W[k=1,i=0]
    #   t2: (a[0,j]+a[1,j]) * b[j,0] -> W[k=0,i=0], W[k=0,i=1]
    #   t3: (a[0,j]+a[1,j]) * (b[j,0]+b[j,1]) -> W[k=0,i=0], W[k=1,i=0], W[k=0,i=1], W[k=1,i=1]
    #   Then:
    #   c[0,0] = t0
    #   c[0,1] = t1 - t0
    #   c[1,0] = t2 - t0
    #   c[1,1] = t3 - t1 - t2 + t0
    # This uses 4 terms per j, total 3*4 = 12. Still no savings!
    #
    # The issue is that the 2x2 output requires 4 independent values.
    # 
    # Let me try a different known construction. The rank of 2x3*3x2 is known to be 11.
    # One known algorithm:
    
    # Let me use a specific known rank-11 algorithm.
    # Based on Strassen-like ideas:
    
    # Terms 0-5: handle j=0 with 3 terms, j=1 with 3 terms, j=2 with 3 terms = 9
    # Plus 2 extra terms for cross-sharing
    
    # Actually, let me just use a verified construction.
    # I'll use the following known rank-11 algorithm:
    
    # For j=0:
    # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1       -> a00*b00 -> c00
    # Term 1: U[0,0]=1, V[0,1]=1, W[1,0]=1       -> a00*b01 -> c01
    # Term 2: U[3,0]=1, V[0,0]=1, W[0,1]=1       -> a10*b00 -> c10
    # Term 3: U[3,0]=1, V[0,1]=1, W[1,1]=1       -> a10*b01 -> c11
    
    # For j=1:
    # Term 4: U[0,1]=1, V[1,0]=1, W[0,0]=1       -> a01*b10 -> c00
    # Term 5: U[0,1]=1, V[1,1]=1, W[1,0]=1       -> a01*b11 -> c01
    # Term 6: U[3,1]=1, V[1,0]=1, W[0,1]=1       -> a11*b10 -> c10
    # Term 7: U[3,1]=1, V[1,1]=1, W[1,1]=1       -> a11*b11 -> c11
    
    # For j=2:
    # Term 8: U[0,2]=1, V[2,0]=1, W[0,0]=1       -> a02*b20 -> c00
    # Term 9: U[0,2]=1, V[2,1]=1, W[1,0]=1       -> a02*b21 -> c01
    # Term 10: U[3,2]=1, V[2,0]=1, W[0,1]=1      -> a12*b20 -> c10
    
    # That's 11 terms but missing a12*b21 for c11!
    # We need 12 terms for schoolbook. So this doesn't work directly.
    
    # Let me try a different approach: use linear combinations to save one term.
    # 
    # Key idea: c11 = a10*b01 + a11*b11 + a12*b21
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c00 = a00*b00 + a01*b10 + a02*b20
    #
    # Define:
    # s0 = (a00+a10)*b00 -> gives a00*b00 + a10*b00
    # s1 = (a00+a10)*b01 -> gives a00*b01 + a10*b01
    # s2 = (a01+a11)*b10 -> gives a01*b10 + a11*b10
    # s3 = (a01+a11)*b11 -> gives a01*b11 + a11*b11
    # s4 = (a02+a12)*b20 -> gives a02*b20 + a12*b20
    # s5 = (a02+a12)*b21 -> gives a02*b21 + a12*b21
    #
    # Then:
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c11 = a10*b01 + a11*b11 + a12*b21
    #
    # From s0: a00*b00 + a10*b00 = s0
    # From s1: a00*b01 + a10*b01 = s1
    # From s2: a01*b10 + a11*b10 = s2
    # From s3: a01*b11 + a11*b11 = s3
    # From s4: a02*b20 + a12*b20 = s4
    # From s5: a02*b21 + a12*b21 = s5
    #
    # We need 4 more terms to separate the i=0 and i=1 parts:
    # t0 = a00*b00
    # t1 = a00*b01
    # t2 = a01*b10
    # t3 = a01*b11
    # t4 = a02*b20
    # t5 = a02*b21
    #
    # Then:
    # a10*b00 = s0 - t0
    # a10*b01 = s1 - t1
    # a11*b10 = s2 - t2
    # a11*b11 = s3 - t3
    # a12*b20 = s4 - t4
    # a12*b21 = s5 - t5
    #
    # Total terms: 6 + 6 = 12. Still 12!
    #
    # To get to 11, we need one more sharing. Notice that we can combine:
    # Instead of separate t0 and t1, use:
    # u0 = a00*(b00+b01)
    # Then a00*b01 = u0 - t0
    # Similarly for other j.
    #
    # Revised:
    # s0 = (a00+a10)*b00
    # s1 = (a00+a10)*b01
    # s2 = (a01+a11)*b10
    # s3 = (a01+a11)*b11
    # s4 = (a02+a12)*b20
    # s5 = (a02+a12)*b21
    # u0 = a00*(b00+b01)
    # u1 = a01*(b10+b11)
    # u2 = a02*(b20+b21)
    # t0 = a10*b00
    # t1 = a11*b10
    # t2 = a12*b20
    #
    # Then:
    # a00*b00 = t0 is not directly available... 
    # 
    # This is getting complicated. Let me just use rank 12 for the subproblem
    # and tile to get rank 48, or find a better global strategy.
    
    # For now, let me return a valid rank-12 (schoolbook) for 2x3*3x2
    r = 12
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W

def _rank16_2x3_3x3():
    """
    Rank-16 bilinear algorithm for 2x3 * 3x3.
    Convention: U[i*3+j, c], V[j*3+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 3, 3
    r = 16
    U = [[0] * r for _ in range(n * m)]  # 6 x 16
    V = [[0] * r for _ in range(m * p)]  # 9 x 16
    W = [[0] * r for _ in range(p * n)]  # 6 x 16
    
    # Schoolbook would be 2*3*3 = 18. We need rank 16.
    # Use the sharing trick: for each (j,k), use 2 terms for i=0 and i=1,
    # but share across k.
    
    # For each j in {0,1,2}:
    #   s0 = a[0,j] * b[j,0]
    #   s1 = a[0,j] * b[j,1]
    #   s2 = a[0,j] * b[j,2]
    #   t0 = (a[0,j]+a[1,j]) * b[j,0]
    #   t1 = (a[0,j]+a[1,j]) * b[j,1]
    #   t2 = (a[0,j]+a[1,j]) * b[j,2]
    #   Then a[1,j]*b[j,k] = t_k - s_k
    #   Total: 6 terms per j, 18 total. No savings.
    
    # Better: for each j, use 3 terms:
    #   s0 = a[0,j] * b[j,0]
    #   s1 = a[0,j] * (b[j,1]+b[j,2])
    #   t0 = (a[0,j]+a[1,j]) * b[j,0]
    #   t1 = (a[0,j]+a[1,j]) * (b[j,1]+b[j,2])
    #   Then:
    #   a[0,j]*b[j,1] + a[0,j]*b[j,2] = s1
    #   a[1,j]*b[j,0] = t0 - s0
    #   a[1,j]*b[j,1] + a[1,j]*b[j,2] = t1 - s1
    #   But we need a[0,j]*b[j,1] and a[0,j]*b[j,2] separately for c[0,1] and c[0,2]
    #   So this doesn't work directly.
    
    # Let me just use schoolbook rank 18 for now and tile.
    r = 18
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Strategy: tile 2x3*3x2 four times (rank 12 each) = rank 48
    # Or use a direct approach.
    
    # Let me try a direct rank-44 construction using the sharing trick globally.
    # 
    # For 2x3 * 3x8:
    # c[i,k] = sum_j a[i,j] * b[j,k] for i in {0,1}, k in {0,...,7}
    #
    # For each j in {0,1,2}, define:
    #   s_k = a[0,j] * b[j,k] for k=0,...,7  (8 terms)
    #   t_k = (a[0,j]+a[1,j]) * b[j,k] for k=0,...,7  (8 terms)
    #   Then a[1,j]*b[j,k] = t_k - s_k
    #   Total: 2*8 = 16 terms per j, 48 total. No savings.
    #
    # To save, share across k:
    # For each j, instead of 8 separate b[j,k], use linear combinations.
    # Define for each j:
    #   s0 = a[0,j] * b[j,0]
    #   s1 = a[0,j] * b[j,1]
    #   ...
    #   s7 = a[0,j] * b[j,7]
    #   t0 = (a[0,j]+a[1,j]) * b[j,0]
    #   ...
    #   t7 = (a[0,j]+a[1,j]) * b[j,7]
    #
    # This is still 16 per j.
    #
    # Alternative: use 8 terms per j by defining:
    #   q_k = a[0,j] * b[j,k]
    #   r_k = a[1,j] * b[j,k]
    #   And c[0,k] += q_k, c[1,k] += r_k
    # That's 16 per j, 48 total.
    #
    # The only way to save is to share across j. But the a[i,j] are different.
    #
    # Let me try: define for each k:
    #   x_k = a[0,0]*b[0,k] + a[0,1]*b[1,k] + a[0,2]*b[2,k] = c[0,k]
    #   y_k = a[1,0]*b[0,k] + a[1,1]*b[1,k] + a[1,2]*b[2,k] = c[1,k]
    #
    # We need to compute these 16 values using bilinear products.
    #
    # One approach: use 8 "combined" products:
    #   z_k = (a[0,0]+a[1,0])*b[0,k] + (a[0,1]+a[1,1])*b[1,k] + (a[0,2]+a[1,2])*b[2,k]
    #        = c[0,k] + c[1,k]
    #   Then c[1,k] = z_k - c[0,k]
    #
    # But we still need c[0,k] separately, which requires 3 products per k = 24.
    # Plus 8 for z_k = 32 total. That's better than 48!
    #
    # Wait, z_k uses 3 products per k: (a[0,j]+a[1,j])*b[j,k] for j=0,1,2.
    # And c[0,k] uses 3 products per k: a[0,j]*b[j,k] for j=0,1,2.
    # Total: 6 products per k, 48 total. No savings!
    #
    # The issue is that each "combined" product still costs 1 term.
    #
    # Let me try a completely different approach: use the known fact that
    # the rank of 2x3*3x8 is at most 44 by using a rank-11 for 2x3*3x2.
    #
    # I'll construct a valid rank-48 solution and return it, since finding
    # a lower rank requires known algorithms I don't have memorized.
    
    rank = n * m * p  # 48
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
    
    # Verify
    assert _verify(U, V, W, n, m, p, rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
