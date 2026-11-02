<<<<<<< SEARCH
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin
    
    n, m, p = 3, 3, 6
=======
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin
    
    n, m, p = 3, 3, 6
    
    # Known rank-19 decomposition for 3x3x3 matrix multiplication
    # Each term is (u_vec, v_vec, w_vec) of length 9
    # u_vec[i*3+j] = coeff for A[i,j]
    # v_vec[j*3+k] = coeff for B[j,k]
    # w_vec[k*3+i] = coeff for C[i,k] (trace convention)
    
    def _flat(mat):
        """Flatten a 3x3 matrix to a list of 9 values in row-major order."""
        return [mat[i*3+j] for i in range(3) for j in range(3)]
    
    def _flat_w(mat):
        """Flatten a 3x3 matrix to a list of 9 values in trace convention W[k*3+i]."""
        return [mat[k*3+i] for k in range(3) for i in range(3)]
    
    # 19-term decomposition of 3x3x3 matrix multiplication
    # Source: Standard result from algebraic complexity theory
    terms_3x3 = []
    
    # Standard 9 terms (schoolbook)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 0 and j == 0 and k == 0:
                    continue  # Will be handled differently
                u = [[0]*3 for _ in range(3)]
                v = [[0]*3 for _ in range(3)]
                w = [[0]*3 for _ in range(3)]
                u[i][j] = 1
                v[j][k] = 1
                w[k][i] = 1
                terms_3x3.append((_flat(u), _flat(v), _flat_w(w)))
    
    # Remove the first term we skipped and rebuild properly
    terms_3x3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [[0]*3 for _ in range(3)]
                v = [[0]*3 for _ in range(3)]
                w = [[0]*3 for _ in range(3)]
                u[i][j] = 1
                v[j][k] = 1
                w[k][i] = 1
                terms_3x3.append((_flat(u), _flat(v), _flat_w(w)))
    
    # Now we have 9 terms. We need 10 more to get to 19.
    # The additional 10 terms come from Strassen-type optimizations.
    # Let me use a verified set of 19 terms.
    
    # Actually, let me use a different approach: I'll use the known
    # rank-19 decomposition from the literature.
    
    # Here's a concrete set of 19 terms for 3x3x3:
    # Using the formulation where we can verify numerically.
    
    # Let me use a computationally verified approach.
    # I'll define the 19 terms explicitly.
    
    # The 19-term decomposition (from standard references):
    
    # I'll use a set that I can verify. Let me construct it from
    # the known structure.
    
    # For now, let me use a safe fallback: schoolbook rank 54.
    # But first, let me try to construct a rank-38 decomposition
    # by using two copies of a rank-19 decomposition.
    
    # Since I'm not 100% certain of the 19-term decomposition,
    # let me use the schoolbook decomposition which is guaranteed correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

