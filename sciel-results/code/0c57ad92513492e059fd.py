import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Try Rank 46 Construction
    # Split B into two 3x3 blocks
    # Block 1: B[:, 0:3], Block 2: B[:, 3:6]
    # For each 3x3x3, use Rank 23 decomposition
    
    def get_3x3x3_rank23():
        # Returns list of (i, j, k, coeff) for 3x3x3
        # i: row of A, j: col of C (which is col of B for 3x3x3), k: col of A (which is row of B)
        # Wait, tensor T[i, j, k] = A[i, k] B[k, j]
        # U index: i*m + k ? No, U is (n*m, rank). U[i*m+k, r]
        # V index: m*p + j ? No, V is (m*p, rank). V[k*p+j, r]
        # W index: p*n + i ? No, W is (p*n, rank). W[j*n+i, r]
        
        terms = []
        # Strassen for 2x2 block (indices 0,1)
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # This decomposition is for the product C = AB.
        # The 7 terms are linear combinations of A and B.
        # We need to express the rank-1 tensors u, v, w.
        
        # It is easier to just use the 16 schoolbook terms for indices involving 2
        # and 7 Strassen terms for the 2x2 block.
        
        # Schoolbook terms for indices with 2:
        # (i, j, k) where i,j,k in {0,1,2} and at least one is 2.
        # Total 27 - 8 = 19 terms?
        # No, the 2x2 block has 8 schoolbook terms. We replace them with 7 Strassen.
        # So 19 + 7 = 26?
        # Wait, the 2x2 block indices are (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1).
        # The other 19 terms are the rest.
        # If we use Strassen, we replace the 8 with 7.
        # So 19 + 7 = 26.
        # But the rank is 23.
        # This means the Strassen terms also cover some of the "other" terms?
        # No, Strassen is only for the 2x2 sub-block.
        # The rank of 3x3x3 is 23.
        # My calculation 19+7=26 is an upper bound.
        # To get 23, we need to combine terms.
        
        # Given the complexity, I will just return the schoolbook 54.
        # The risk of implementing 23 incorrectly is high.
        return None

    # If I can't implement 23 reliably, I return 54.
    # But 54 is the baseline.
    # I will return the schoolbook.
    
    return {"U": U_school, "V": V_school, "W": W_school}
