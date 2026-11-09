import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# Known optimal 2x2x2 matrix multiplication tensor (rank 7)
# This is the Strassen decomposition for 2x2 matrices.
# Indices: A (2x2), B (2x2), C = A*B (2x2)
# U indices: i*2 + j for A[i,j] -> 0,1,2,3
# V indices: j*2 + k for B[j,k] -> 0,1,2,3
# W indices: k*2 + i for C[k,i] -> 0,1,2,3

# Standard Strassen rank-7 factors
# M1 = A00*B00
# M2 = (A00+A11)*B11
# M3 = (A10+A11)*B01
# M4 = A00*(B01-B11)
# M5 = (A01+A10)*(B00-B10)
# M6 = A10*(B01+B11)
# M7 = A11*(B10-B01)

# C00 = M1 + M4 - M6 + M7
# C01 = M3 + M5
# C10 = M2 - M4 + M5
# C11 = M1 - M2 + M6 + M7

# Let's verify the structure. The tensor T[i,j,k] = 1 if we compute C[i,k] = sum_j A[i,j]B[j,k]
# In our convention, the decomposition is sum_r U[i,j,r] V[j,k,r] W[i,k,r] = T[i,j,k] for all i,j,k?
# No, the tensor for matrix multiplication is T[i,j,k] where the output is C[i,k] = sum_j A[i,j]B[j,k].
# The tensor is usually defined on indices (i, j, k) such that the coefficient of A[i,j]B[j,k] in the computation of C[i,k] is 1.
# Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]".
# Let's look at the seed program:
# U[i*m+j][column] = 1
# V[j*p+k][column] = 1
# W[k*n+i][column] = 1
# This implies: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,k} ? No.
# For schoolbook, for a specific (i,j,k), U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1.
# So the product sum is 1 for (i,j,k) and 0 otherwise?
# No, matrix multiplication C[i,k] = sum_j A[i,j]B[j,k].
# The tensor is T_{i,j,k} = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
# In the seed, for a fixed column, it picks a specific (i,j,k) triple.
# So T[i,j,k] = 1 for all i,j,k? No, that's the identity tensor?
# Actually, the matrix multiplication tensor is defined such that the coefficient of A[i,j]B[j,k] is 1 for the component C[i,k].
# So the tensor is sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
# The seed sets U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
# The product is sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c].
# This equals 1 if there is a c such that U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
# In schoolbook, each c corresponds to a unique (i,j,k). So the sum is 1 for the specific (i,j,k) that generated c, and 0 otherwise.
# So the tensor being decomposed is T[i,j,k] = 1 for all i,j,k.
# Wait, if T[i,j,k]=1 for all i,j,k, then C[i,k] = sum_{j} A[i,j] B[j,k] * T[i,j,k]?
# If T[i,j,k]=1 for all j, then C[i,k] = sum_j A[i,j]B[j,k]. This is correct.
# So the tensor is the "all ones" tensor in the indices (i,j,k) for the relevant ranges?
# No, the tensor is defined on the input spaces A (n*m), B (m*p), and output space C (p*n)?
# Let's check the shapes.
# U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
# The product U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] sums over r.
# The result is a tensor of shape (n*m, m*p, p*n)? No, it's a contraction.
# Actually, the condition for validity is:
# For all i, j, k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
# This is because A[i,j] and B[j,k] are independent variables, and their product A[i,j]B[j,k] must appear with coefficient 1 in the computation of C[i,k].
# And C[i,k] is the variable at index k*n+i in W.
# So yes, the target tensor T has T[i*m+j, j*p+k, k*n+i] = 1? No.
# The tensor is T[a, b, c] where a is index in A, b in B, c in C.
# T[a, b, c] = 1 if A[a] and B[b] are multiplied to contribute to C[c].
# A[a] is A[i,j] where a = i*m+j.
# B[b] is B[j,k] where b = j*p+k.
# C[c] is C[i,k] where c = k*n+i.
# For A[i,j] and B[j,k] to contribute to C[i,k], the indices must match:
# a = i*m+j
# b = j'*p+k'
# c = k''*n+i''
# They match if j=j', k=k', i=i'', i''=i.
# So T[a, b, c] = 1 if b - (b mod p) == a mod m? No.
# a = i*m+j. b = j*p+k. c = k*n+i.
# Condition: (a mod m) == (b div p) AND (b mod p) == (c div n) AND (a div m) == (c mod n).
# If these hold, T[a,b,c]=1. Else 0.

# So the goal is to find U, V, W such that sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].

# Now, back to the 2x2x2 case.
# n=2, m=2, p=2.
# a = i*2+j (0..3)
# b = j*2+k (0..3)
# c = k*2+i (0..3)
# T[a,b,c]=1 if (a%2 == b//2) and (b%2 == c//2) and (a//2 == c%2).

# Strassen's algorithm computes C = AB with 7 multiplications.
# This corresponds to a rank-7 decomposition.
# We need the explicit U, V, W matrices for Strassen.

# Let's derive Strassen factors.
# M1 = A00 B00
# M2 = (A00+A11) B11
# M3 = (A10+A11) B01
# M4 = A00 (B01-B11)
# M5 = (A01+A10) (B00-B10)
# M6 = A10 (B01+B11)
# M7 = A11 (B10-B01)

# C00 = M1 + M4 - M6 + M7
# C01 = M3 + M5
# C10 = M2 - M4 + M5
# C11 = M1 - M2 + M6 + M7

# The terms are linear combinations of A and B.
# U corresponds to A coefficients, V to B coefficients, W to C coefficients.
# For each multiplication M_r, we have:
# M_r = (sum_{a} u_{a,r} A[a]) * (sum_{b} v_{b,r} B[b])
# And C[c] = sum_{r} w_{c,r} M_r.

# Let's map indices:
# A00 -> a=0, A01 -> a=1, A10 -> a=2, A11 -> a=3
# B00 -> b=0, B01 -> b=1, B10 -> b=2, B11 -> b=3
# C00 -> c=0 (k=0,i=0 -> 0*2+0=0)
# C01 -> c=1 (k=1,i=0 -> 1*2+0=1) -- Wait, W index is k*n+i.
# n=2.
# C00: i=0, k=0 -> c = 0*2+0 = 0.
# C01: i=0, k=1 -> c = 1*2+0 = 2.
# C10: i=1, k=0 -> c = 0*2+1 = 1.
# C11: i=1, k=1 -> c = 1*2+1 = 3.

# So W index mapping:
# c=0 -> C00
# c=1 -> C10
# c=2 -> C01
# c=3 -> C11

# Let's write down U, V, W for the 7 multiplications.

# M1 = A00 B00
# u: A00=1 -> [1,0,0,0]
# v: B00=1 -> [1,0,0,0]
# w: contributes to C00 (+1) and C11 (+1) -> [1,0,1,1] (c0=1, c3=1)

# M2 = (A00+A11) B11
# u: A00=1, A11=1 -> [1,0,0,1]
# v: B11=1 -> [0,0,0,1]
# w: contributes to C10 (+1) and C11 (-1) -> [0,1,0,-1] (c1=1, c3=-1)

# M3 = (A10+A11) B01
# u: A10=1, A11=1 -> [0,0,1,1]
# v: B01=1 -> [0,1,0,0]
# w: contributes to C01 (+1) -> [0,0,1,0] (c2=1)

# M4 = A00 (B01-B11)
# u: A00=1 -> [1,0,0,0]
# v: B01=1, B11=-1 -> [0,1,0,-1]
# w: contributes to C00 (+1) and C10 (-1) -> [1,-1,0,0] (c0=1, c1=-1)

# M5 = (A01+A10) (B00-B10)
# u: A01=1, A10=1 -> [0,1,1,0]
# v: B00=1, B10=-1 -> [1,0,-1,0]
# w: contributes to C01 (+1) and C10 (+1) -> [0,1,1,0] (c1=1, c2=1)

# M6 = A10 (B01+B11)
# u: A10=1 -> [0,0,1,0]
# v: B01=1, B11=1 -> [0,1,0,1]
# w: contributes to C00 (-1) and C11 (+1) -> [-1,0,0,1] (c0=-1, c3=1)

# M7 = A11 (B10-B01)
# u: A11=1 -> [0,0,0,1]
# v: B10=1, B01=-1 -> [0,-1,1,0]
# w: contributes to C00 (+1) and C11 (+1) -> [1,0,0,1] (c0=1, c3=1)

# Let's check M7 w: C00 = M1 + M4 - M6 + M7. So +1. C11 = M1 - M2 + M6 + M7. So +1. Correct.

# So U, V, W for 2x2x2 are:
U_222 = [
    [1, 1, 0, 1, 0, 0, 0], # A00
    [0, 0, 0, 0, 1, 0, 0], # A01
    [0, 0, 1, 0, 1, 1, 0], # A10
    [0, 1, 1, 0, 0, 0, 1]  # A11
]
V_222 = [
    [1, 0, 0, 0, 1, 0, 0], # B00
    [0, 0, 1, 1, 0, 1, -1],# B01
    [0, 0, 0, 0, -1, 0, 1],# B10
    [0, 1, 0, -1, 0, 1, 0] # B11
]
W_222 = [
    [1, 0, 0, 1, -1, 0, 1], # C00 (c=0)
    [0, 1, 0, -1, 1, 0, 0], # C10 (c=1)
    [0, 0, 1, 0, 1, 0, 0],  # C01 (c=2)
    [1, -1, 0, 0, 0, 1, 1]  # C11 (c=3)
]

def _build_222():
    return U_222, V_222, W_222

def construct():
    n, m, p = DIMENSIONS # 3, 4, 4
    
    # We want to decompose M_{3,4,4}.
    # Strategy:
    # Split A (3x4) into A_top (2x4) and A_bot (1x4).
    # C (3x4) = [A_top B; A_bot B].
    # C_bot (1x4) = A_bot B. This is a 1x4x4 multiplication. Rank 16.
    # C_top (2x4) = A_top B. This is a 2x4x4 multiplication.
    # We can further split A_top (2x4) into A_TL (2x2) and A_TR (2x2).
    # And B (4x4) into B_L (4x2) and B_R (4x2).
    # Then C_TL = A_TL B_L (2x2x2, rank 7).
    # And C_TR = A_TR B_R (2x2x2, rank 7).
    # Total rank = 16 + 7 + 7 = 30.
    
    rank = 30
    U = [[0]*rank for _ in range(n*m)] # 12x30
    V = [[0]*rank for _ in range(m*p)] # 16x30
    W = [[0]*rank for _ in range(p*n)] # 12x30
    
    # 1. C_bot = A_bot B (1x4 x 4x4 -> 1x4)
    # A_bot is rows 2, cols 0-3. A indices: 2*4+0=8, 9, 10, 11.
    # B is rows 0-3, cols 0-3. B indices: 0-15.
    # C_bot is row 2, cols 0-3. C indices: k*n+i. i=2.
    # k=0 -> c=0*3+2=2.
    # k=1 -> c=1*3+2=5.
    # k=2 -> c=2*3+2=8.
    # k=3 -> c=3*3+2=11.
    
    # Schoolbook for 1x4x4.
    # For each j (0..3) and k (0..3), we have a term A_bot[j] B[j,k].
    # A index: 8+j.
    # B index: j*4+k.
    # C index: k*3+2.
    
    col = 0
    for j in range(4):
        for k in range(4):
            a_idx = 8 + j
            b_idx = j * 4 + k
            c_idx = k * 3 + 2
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
    # col is now 16.
    
    # 2. C_TL = A_TL B_L (2x2 x 4x2 -> 2x2? No, A_TL is 2x2, B_L is 4x2. Wait.)
    # A_top is rows 0,1.
    # A_TL: rows 0,1, cols 0,1. A indices: 0,1, 4,5.
    # A_TR: rows 0,1, cols 2,3. A indices: 2,3, 6,7.
    # B_L: rows 0-3, cols 0,1. B indices: 0,1, 4,5, 8,9, 12,13.
    # B_R: rows 0-3, cols 2,3. B indices: 2,3, 6,7, 10,11, 14,15.
    
    # C_TL is C[0:2, 0:2].
    # C indices:
    # i=0,k=0 -> 0. i=0,k=1 -> 3.
    # i=1,k=0 -> 1. i=1,k=1 -> 4.
    
    # C_TR is C[0:2, 2:4].
    # i=0,k=2 -> 6. i=0,k=3 -> 7.
    # i=1,k=2 -> 9. i=1,k=3 -> 10.
    
    # We use Strassen for 2x2x2.
    # But Strassen computes C (2x2) = A (2x2) B (2x2).
    # Here, for C_TL, A is 2x2, but B is 4x2.
    # Wait. C_TL (2x2) = A_TL (2x2) * B_L (4x2)?
    # No. A_TL is 2x2. B_L is 4x2.
    # A_TL * B_L is not defined (2x2 times 4x2).
    # Ah, matrix multiplication C = AB. A is n x m, B is m x p.
    # A is 3x4. B is 4x4.
    # A_TL is rows 0-1, cols 0-1 of A. So 2x2.
    # B_L is rows 0-3, cols 0-1 of B. So 4x2.
    # The product C_TL = sum_{j=0}^{1} A_TL[:,j] B_L[j,:].
    # This is (2x2) * (2x2) if we only take the first 2 rows of B_L?
    # No, B_L has 4 rows. A_TL has 2 columns.
    # So we sum over j=0,1.
    # The relevant part of B is B[0:2, 0:2]. Let's call it B_TL.
    # So C_TL = A_TL * B_TL. (2x2 * 2x2).
    
    # What about the contribution from j=2,3?
    # A_TL has zeros in cols 2,3. So A_TL[:,2]=0, A_TL[:,3]=0.
    # So A_TL * B[2:4, 0:2] is 0.
    # So C_TL is indeed A_TL (2x2) * B_TL (2x2).
    
    # Similarly, C_TR = A_TR * B_TR.
    # A_TR is 2x2 (cols 2,3). B_TR is 2x2 (rows 0,1, cols 2,3)?
    # No, sum over j=2,3.
    # A_TR has non-zeros in cols 2,3.
    # So C_TR = A_TR * B[2:4, 2:4]. Let's call it B_BR.
    
    # So we have two independent 2x2 matrix multiplications.
    # 1. C_TL = A_TL * B_TL.
    # A_TL indices in A: 0,1,4,5.
    # B_TL indices in B: 0,1,4,5. (Rows 0,1, Cols 0,1).
    # C_TL indices in C: 0,3,1,4. (i=0,k=0; i=0,k=1; i=1,k=0; i=1,k=1).
    
    # 2. C_TR = A_TR * B_BR.
    # A_TR indices in A: 2,3,6,7.
    # B_BR indices in B: 10,11,14,15. (Rows 2,3, Cols 2,3).
    # C_TR indices in C: 6,7,9,10.
    
    # We use the Strassen factors for 2x2x2.
    # We need to map the local 2x2 indices (0..3) to the global indices.
    
    # Local indices for 2x2:
    # a_loc: 0=A00, 1=A01, 2=A10, 3=A11.
    # b_loc: 0=B00, 1=B01, 2=B10, 3=B11.
    # c_loc: 0=C00, 1=C10, 2=C01, 3=C11.
    
    # Mapping for C_TL:
    # A_TL[0,0] -> a_glob 0. A_TL[0,1] -> a_glob 1.
    # A_TL[1,0] -> a_glob 4. A_TL[1,1] -> a_glob 5.
    # So a_map_TL = [0, 1, 4, 5].
    
    # B_TL[0,0] -> b_glob 0. B_TL[0,1] -> b_glob 1.
    # B_TL[1,0] -> b_glob 4. B_TL[1,1] -> b_glob 5.
    # So b_map_TL = [0, 1, 4, 5].
    
    # C_TL[0,0] -> c_glob 0. C_TL[0,1] -> c_glob 3.
    # C_TL[1,0] -> c_glob 1. C_TL[1,1] -> c_glob 4.
    # So c_map_TL = [0, 1, 3, 4]. (c_loc 0->0, 1->1, 2->3, 3->4).
    
    # Mapping for C_TR:
    # A_TR[0,0] -> a_glob 2. A_TR[0,1] -> a_glob 3.
    # A_TR[1,0] -> a_glob 6. A_TR[1,1] -> a_glob 7.
    # So a_map_TR = [2, 3, 6, 7].
    
    # B_BR[0,0] -> b_glob 10. B_BR[0,1] -> b_glob 11.
    # B_BR[1,0] -> b_glob 14. B_BR[1,1] -> b_glob 15.
    # So b_map_TR = [10, 11, 14, 15].
    
    # C_TR[0,0] -> c_glob 6. C_TR[0,1] -> c_glob 7.
    # C_TR[1,0] -> c_glob 9. C_TR[1,1] -> c_glob 10.
    # So c_map_TR = [6, 7, 9, 10].
    
    U_s, V_s, W_s = _build_222()
    
    # Add C_TL factors
    for r in range(7):
        for a_loc in range(4):
            a_glob = a_map_TL[a_loc]
            U[a_glob][col + r] += U_s[a_loc][r]
        for b_loc in range(4):
            b_glob = b_map_TL[b_loc]
            V[b_glob][col + r] += V_s[b_loc][r]
        for c_loc in range(4):
            c_glob = c_map_TL[c_loc]
            W[c_glob][col + r] += W_s[c_loc][r]
    
    col += 7
    # col is now 23.
    
    # Add C_TR factors
    for r in range(7):
        for a_loc in range(4):
            a_glob = a_map_TR[a_loc]
            U[a_glob][col + r] += U_s[a_loc][r]
        for b_loc in range(4):
            b_glob = b_map_TR[b_loc]
            V[b_glob][col + r] += V_s[b_loc][r]
        for c_loc in range(4):
            c_glob = c_map_TR[c_loc]
            W[c_glob][col + r] += W_s[c_loc][r]
    
    col += 7
    # col is now 30.
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    # Quick verification
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # We want T[a,b,c] = 1 if indices match, else 0.
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                # Target is 1
                val = np.dot(U[a], V[b, :]) # This is not right, V is (m*p, rank).
                # val = sum_r U[a,r] V[b,r]
                # Then result = sum_c ... No.
                # The condition is sum_r U[a,r] V[b,r] W[c,r] = 1.
                # Let's compute the full tensor and check.
                
    # Efficient check
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Let M = U @ V.T. M[a,b] = sum_r U[a,r] V[b,r].
    # Then T[a,b,c] = sum_c ... No, W is (p*n, rank).
    # T[a,b,c] = M[a,b] * W[c].T? No.
    # T[a,b,c] = sum_r (U @ V.T)[a,b] * W[c,r]? No.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    # This is element-wise product?
    # Let X = U @ V.T. X[a,b] = sum_r U[a,r] V[b,r].
    # Then T[a,b,c] is not directly X.
    # We need to check sum_r U[a,r] V[b,r] W[c,r] = 1.
    
    # Let's just do it explicitly for small size.
    rank = len(U[0])
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        # Outer product of U[:,r], V[:,r], W[:,r]
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if T[a,b,c] != 1:
                    max_err = max(max_err, abs(T[a,b,c]-1))
                # Also check other c?
                # For fixed a,b, only one c should be 1?
                # a = i*m+j, b = j*p+k.
                # c must be k*n+i.
                # Are there other (i', j', k') that give same a,b?
                # a=i*m+j. b=j*p+k.
                # i = a//m, j = a%m.
                # j' = b//p, k = b%p.
                # So i,j,k are uniquely determined by a,b.
                # So T[a,b,c] should be 1 if c = k*n+i, else 0.
                for c_test in range(p*n):
                    if c_test != c:
                        if T[a,b,c_test] != 0:
                            max_err = max(max_err, abs(T[a,b,c_test]))
    
    print("Max error:", max_err)
    print("Rank:", rank)
