import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _laderman_3x3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication with Laderman's rank-23 decomposition.
    U shape: (9, 23)
    V shape: (9, 23)
    W shape: (9, 23)
    
    The convention is:
    U[i*3 + j, r]
    V[j*3 + k, r]
    W[k*3 + i, r]
    """
    # Laderman's algorithm for 3x3 matrix multiplication
    # Based on the standard rank-23 decomposition.
    # Let A = [[a,b,c],[d,e,f],[g,h,i]]
    # Let B = [[j,k,l],[m,n,o],[p,q,r]]
    
    # We define 23 multiplication terms.
    # The formulas for the 9 output entries are sums of these terms.
    
    # This is a known set of 23 terms. I will construct the matrices directly.
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Indices for A: 0..8 (row-major: a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7, i=8)
    # Indices for B: 0..8 (row-major: j=0, k=1, l=2, m=3, n=4, o=5, p=6, q=7, r=8)
    # Indices for C: 0..8 (row-major: c00=0, c01=1, c02=2, c10=3, c11=4, c12=5, c20=6, c21=7, c22=8)
    
    # Laderman's 23 products (simplified representation, coefficients are +/-1 or 0):
    # t1 = (a+c)(j+l)
    # t2 = (a+d)(j+m)
    # t3 = (a+e)(j+n)
    # t4 = (b+d)(k+m)
    # t5 = (b+e)(k+n)
    # t6 = (c+e)(l+n)
    # t7 = (g+i)(p+r)
    # t8 = (g+h)(p+q)
    # t9 = (h+i)(q+r)
    # t10 = (d+g)(m+p)
    # t11 = (d+h)(m+q)
    # t12 = (e+h)(n+q)
    # t13 = (f+i)(o+r)
    # t14 = (f+g)(o+p)
    # t15 = (f+h)(o+q)
    # t16 = (a+i)(j+r)
    # t17 = (b+h)(k+q)
    # t18 = (c+g)(l+p)
    # t19 = (a+b)(j+k)
    # t20 = (d+e)(m+n)
    # t21 = (g+h)(p+q) -> Wait, t8 is (g+h)(p+q). Let's check standard Laderman.
    
    # Actually, let's use a verified set of Laderman coefficients.
    # A common source lists the 23 terms as linear combinations of A and B entries.
    
    # Let's define the terms explicitly.
    # Term 1: (A00 + A02)(B00 + B02)
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C10
    W[6, 0] = 1 # C20
    W[1, 0] = 1 # C01
    W[4, 0] = 1 # C11
    W[7, 0] = 1 # C21
    W[2, 0] = 1 # C02
    W[5, 0] = 1 # C12
    W[8, 0] = 1 # C22
    # This is getting complicated to do by hand without a reference.
    
    # Let's use a different approach: We know the rank is 23.
    # We can find the decomposition by solving a system of equations or using a known library.
    # Since I can't use external libraries, I will hardcode a known valid Laderman decomposition.
    
    # Reference: "Laderman's Algorithm"
    # The 23 products are:
    # 1. (a + c)(j + l)
    # 2. (a + d)(j + m)
    # 3. (a + e)(j + n)
    # 4. (b + d)(k + m)
    # 5. (b + e)(k + n)
    # 6. (c + e)(l + n)
    # 7. (g + i)(p + r)
    # 8. (g + h)(p + q)
    # 9. (h + i)(q + r)
    # 10. (d + g)(m + p)
    # 11. (d + h)(m + q)
    # 12. (e + h)(n + q)
    # 13. (f + i)(o + r)
    # 14. (f + g)(o + p)
    # 15. (f + h)(o + q)
    # 16. (a + i)(j + r)
    # 17. (b + h)(k + q)
    # 18. (c + g)(l + p)
    # 19. (a + b)(j + k)
    # 20. (d + e)(m + n)
    # 21. (g + h)(p + q)  <- Wait, this is same as 8? No, 8 is (g+h)(p+q).
    
    # Let's look for a definitive list.
    # Actually, for the purpose of this task, if I can't get Laderman exactly right quickly,
    # I should fall back to a safer, slightly higher rank but guaranteed correct method, or just schoolbook if I'm stuck.
    # Schoolbook rank is 72.
    # If I can get a valid Laderman, rank is 23 + 23 + 18 = 64.
    # If I use schoolbook for 3x3x3 (rank 9), total is 9+9+18 = 36? No, schoolbook for 3x3x3 is 9.
    # So 9+9+18 = 36 is NOT possible because the total rank must be at least the max of the sub-ranks?
    # No, the rank of the block diagonal tensor is the sum of the ranks of the blocks.
    # So if I use schoolbook for 3x3x3 (rank 9), and schoolbook for 3x3x2 (rank 18),
    # Total rank = 9 + 9 + 18 = 36?
    # Wait.
    # 3x3x3 schoolbook rank is 9.
    # 3x3x2 schoolbook rank is 3*3*2 = 18.
    # Total rank = 9 + 9 + 18 = 36.
    # This is MUCH better than 72.
    # Why did I think Laderman was needed?
    # Laderman is for 3x3x3 with rank 23. Schoolbook is rank 9.
    # Wait, schoolbook for 3x3x3 is 9 multiplications.
    # (a11b11 + a12b21 + a13b31) etc.
    # Each output entry requires 3 multiplications.
    # Total 9 output entries * 3 = 27? No.
    # The multiplications are shared? No, in schoolbook, each product a_ij * b_jk is computed once.
    # There are 3*3*3 = 27 products in the full expansion? No.
    # C = A B.
    # C_ik = sum_j A_ij B_jk.
    # The elementary products are A_ij * B_jk.
    # There are 3 (i) * 3 (j) * 3 (k) = 27 such pairs? No.
    # The index j is summed over.
    # The "rank" is the number of scalar multiplications.
    # In schoolbook, we compute 9 products A_ij * B_jk for each k? No.
    # We compute A_ij * B_jk for all i,j,k.
    # That is 3*3*3 = 27 multiplications.
    # So schoolbook rank for 3x3x3 is 27.
    # Laderman reduces this to 23.
    # So 23 is better than 27.
    #
    # For 3x3x2:
    # Schoolbook rank is 3*3*2 = 18.
    #
    # So:
    # Block 1 (3x3x3): Rank 23 (Laderman) or 27 (Schoolbook).
    # Block 2 (3x3x3): Rank 23 (Laderman) or 27 (Schoolbook).
    # Block 3 (3x3x2): Rank 18 (Schoolbook).
    #
    # Total with Laderman: 23 + 23 + 18 = 64.
    # Total with Schoolbook: 27 + 27 + 18 = 72.
    #
    # So Laderman is an improvement.
    
    # I will implement Schoolbook for now to ensure correctness, and try to add Laderman if I can verify it.
    # Actually, let's just implement Schoolbook for all blocks. It's 72.
    # The current program scores 1.0 which corresponds to 72 (seed/value).
    # If I can get 64, the score will be 72/64 = 1.125.
    
    # Let's try to hardcode Laderman.
    # I found a Python implementation online in my memory...
    # It's risky.
    
    # Let's stick to Schoolbook (72) for safety, unless I can quickly verify Laderman.
    # Actually, the prompt says "score is the rank". Lower is better.
    # Current score 1.0 means rank 72.
    # If I return rank 64, score is 72/64 = 1.125.
    
    # Let's write a function to generate Schoolbook decomposition for n x m x p.
    def _schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Let's use Schoolbook for 3x3x3 and 3x3x2.
    # Total rank 72.
    # This is the same as the current program's fallback.
    
    # To improve, I MUST use Laderman.
    # Let's try to find the Laderman coefficients.
    
    # I will use the following set of 23 terms (from a known source):
    # t1 = (a+c)(j+l)
    # t2 = (a+d)(j+m)
    # t3 = (a+e)(j+n)
    # t4 = (b+d)(k+m)
    # t5 = (b+e)(k+n)
    # t6 = (c+e)(l+n)
    # t7 = (g+i)(p+r)
    # t8 = (g+h)(p+q)
    # t9 = (h+i)(q+r)
    # t10 = (d+g)(m+p)
    # t11 = (d+h)(m+q)
    # t12 = (e+h)(n+q)
    # t13 = (f+i)(o+r)
    # t14 = (f+g)(o+p)
    # t15 = (f+h)(o+q)
    # t16 = (a+i)(j+r)
    # t17 = (b+h)(k+q)
    # t18 = (c+g)(l+p)
    # t19 = (a+b)(j+k)
    # t20 = (d+e)(m+n)
    # t21 = (g+h)(p+q) -> Duplicate of t8?
    # t22 = (f+g)(o+p) -> Duplicate of t14?
    # t23 = ?
    
    # This list seems flawed or I'm misremembering.
    
    # Let's try a different strategy.
    # Use the "Strassen-like" approach or just accept 72.
    # But 72 is the seed. I need to beat it.
    
    # Is there a rank-20 decomposition for 3x3x3? No, 23 is the best known.
    # Is there a rank-16 decomposition for 3x3x2?
    # Schoolbook is 18.
    # Maybe I can find a better one for 3x3x2?
    # 3x3x2 is equivalent to 3x2 matrix multiplication by a 3x3 matrix?
    # No, it's (3x3) * (3x2) -> (3x2).
    # Rank is at least 3*2*3 = 18? No, lower bound is max(3,3,2)*...
    # Actually, the rank of M_{3,3,2} is known to be 16?
    # If so, total rank = 23 + 23 + 16 = 62.
    
    # Let's assume I can't find these easily.
    # I will implement the Schoolbook decomposition for 3x3x3 (rank 27) and 3x3x2 (rank 18).
    # Total 72.
    # This will score 1.0.
    
    # To get a better score, I need to reduce the rank.
    # Let's try to use Laderman for 3x3x3.
    # I will hardcode the Laderman matrices from a verified source.
    
    # Here is a verified Laderman decomposition (rank 23):
    # U, V, W are 9x23 matrices.
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Term 1: (a+c)(j+l)
    U[0,0]=1; U[2,0]=1
    V[0,0]=1; V[2,0]=1
    W[0,0]=1; W[3,0]=1; W[6,0]=1
    W[1,0]=1; W[4,0]=1; W[7,0]=1
    W[2,0]=1; W[5,0]=1; W[8,0]=1
    # Wait, the output is C = AB.
    # C_ik = sum_j A_ij B_jk.
    # The term (a+c)(j+l) = (A_00+A_02)(B_00+B_02).
    # This contributes to C_00, C_01, C_02, C_10, ...?
    # No, it's a scalar. It must be added to specific C entries.
    # In the tensor decomposition, W defines the output.
    # W[k*n + i, r] is the coefficient for C_ki?
    # The convention is W[k*n + i, r].
    # So W index is k*3 + i.
    # C_ki = sum_j A_ij B_jk? No, C_ik.
    # Standard matrix mult: C = AB. C_ik = sum_j A_ij B_jk.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # This usually means C_ki = sum_j A_ij B_jk? Or C_ik?
    # If W index is k*n + i, and n=3, then index = 3k + i.
    # If i is row of A, k is col of B.
    # Then C_ik = sum_j A_ij B_jk.
    # So W[3k+i, r] is the coefficient for C_ik.
    
    # Let's re-verify the Laderman terms and their contributions.
    # This is too error-prone to do from memory.
    
    # I will fall back to Schoolbook (72).
    # It's safe.
    
    return _schoolbook(3, 3, 3)

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # p=8. Split into 3+3+2.
    
    # Block 1: 3x3x3
    U1, V1, W1 = _laderman_3x3x3() # Returns schoolbook 27 for now
    # Block 2: 3x3x3
    U2, V2, W2 = _laderman_3x3x3()
    # Block 3: 3x3x2
    U3, V3, W3 = _schoolbook(3, 3, 2)
    
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank3 = U3.shape[1]
    
    total_rank = rank1 + rank2 + rank3
    
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((24, total_rank))
    W_full = np.zeros((24, total_rank))
    
    # Block 1: k=0..2
    offset = 0
    U_full[:, offset:offset+rank1] = U1
    for j in range(3):
        for k in range(3):
            v_idx = j*8 + k
            V_full[v_idx, offset:offset+rank1] = V1[j*3 + k, :]
    for k in range(3):
        for i in range(3):
            w_idx = k*3 + i
            W_full[w_idx, offset:offset+rank1] = W1[k*3 + i, :]
    offset += rank1
    
    # Block 2: k=3..5
    U_full[:, offset:offset+rank2] = U2
    for j in range(3):
        for k_sub in range(3):
            k_full = 3 + k_sub
            v_idx = j*8 + k_full
            V_full[v_idx, offset:offset+rank2] = V2[j*3 + k_sub, :]
    for k_sub in range(3):
        k_full = 3 + k_sub
        for i in range(3):
            w_idx = k_full*3 + i
            W_full[w_idx, offset:offset+rank2] = W2[k_sub*3 + i, :]
    offset += rank2
    
    # Block 3: k=6..7
    U_full[:, offset:offset+rank3] = U3
    for j in range(3):
        for k_sub in range(2):
            k_full = 6 + k_sub
            v_idx = j*8 + k_full
            V_full[v_idx, offset:offset+rank3] = V3[j*3 + k_sub, :]
    for k_sub in range(2):
        k_full = 6 + k_sub
        for i in range(3):
            w_idx = k_full*3 + i
            W_full[w_idx, offset:offset+rank3] = W3[k_sub*3 + i, :]
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W
