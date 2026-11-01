import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    # A is n x m (2x2), B is m x p (2x10), C is n x p (2x10)
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Decomposition: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly computes matrix multiplication."""
        n, m, p = 2, 2, 10
        R = len(U[0])
        # Generate random test cases
        np.random.seed(42)
        for trial in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(R):
                # u_A: for each i, sum_j U[i*m+j, r] * A[i,j]
                u_A = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        u_A[i] += U[i*m + j][r] * A[i, j]
                
                # v_B: for each k, sum_j V[j*p+k, r] * B[j,k]
                v_B = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        v_B[k] += V[j*p + k][r] * B[j, k]
                
                # C[i,k] += W[k*n+i, r] * u_A[i] * v_B[k]
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W[k*n + i][r] * u_A[i] * v_B[k]
            
            if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
                return False
        return True

    def build_strassen_solution():
        """
        Constructs a rank-35 decomposition for 2x2 * 2x10 matrix multiplication.
        Uses Strassen (rank 7) for each of the 5 blocks of 2x2 * 2x2.
        """
        n, m, p = 2, 2, 10
        R = 35
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        # Strassen coefficients for 2x2 * 2x2
        # Indices: A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        # U indices: 0:a, 1:b, 2:c, 3:d
        # V indices: 0:e, 1:g, 2:f, 3:h (Note: V is indexed by j*p+k. 
        #   For a block q, V[2q] is B[0, 2q], V[2q+1] is B[1, 2q], 
        #   V[10+2q] is B[0, 2q+1], V[10+2q+1] is B[1, 2q+1].
        #   Wait, V shape is (m*p, R) = (20, R).
        #   V[j*p+k]. j is 0 or 1. k is 0..9.
        #   For block q (columns 2q, 2q+1):
        #   V[0*10 + 2q] = V[2q] corresponds to B[0, 2q]
        #   V[1*10 + 2q] = V[10+2q] corresponds to B[1, 2q]
        #   V[0*10 + 2q+1] = V[2q+1] corresponds to B[0, 2q+1]
        #   V[1*10 + 2q+1] = V[10+2q+1] corresponds to B[1, 2q+1]
        #   
        #   Let's map Strassen variables to these indices.
        #   Strassen usually takes B as [[e, f], [g, h]] where rows are 0,1 and cols 0,1.
        #   e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1].
        #   So for block q:
        #   e -> B[0, 2q] -> V[2q]
        #   g -> B[1, 2q] -> V[10+2q]
        #   f -> B[0, 2q+1] -> V[2q+1]
        #   h -> B[1, 2q+1] -> V[10+2q+1]
        #
        #   W shape is (p*n, R) = (20, R).
        #   W[k*n+i]. k is 0..9, i is 0,1.
        #   For block q:
        #   i -> C[0, 2q] -> W[2q*2 + 0] = W[4q]
        #   j -> C[0, 2q+1] -> W[2q*2 + 1] = W[4q+1]
        #   k -> C[1, 2q] -> W[2q*2 + 2] = W[4q+2]
        #   l -> C[1, 2q+1] -> W[2q*2 + 3] = W[4q+3]
        
        terms = [
            # (U_vec, V_vec, W_vec)
            # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1, C01 -= M1, C10 -= M1 ? No.
            # Standard Strassen:
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B10 - B11)
            # M4 = A11(B01 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            #
            # Let's verify M1:
            # U: [1, 0, 0, 1] (a, b, c, d) -> a+d
            # V: [1, 0, 0, 1] (e, g, f, h) -> e+h
            # W: [1, 0, 0, 1] (i, j, k, l) -> i+l
            # Contribution: (a+d)(e+h)(i+l).
            # This adds (a+d)(e+h) to i and l.
            # My formula says C00 += M1, C11 += M1.
            # So W should be [1, 0, 0, 1]. Correct.
            
            # M2 = (c+d)e
            # U: [0, 0, 1, 1]
            # V: [1, 0, 0, 0] (e)
            # C10 += M2, C11 -= M2.
            # W: [0, 0, 1, -1]
            
            # M3 = a(g-h)
            # U: [1, 0, 0, 0]
            # V: [0, 1, 0, -1] (g-h)
            # C01 += M3, C11 += M3.
            # W: [0, 1, 0, 1]
            
            # M4 = d(f-e)
            # U: [0, 0, 0, 1]
            # V: [0, 0, 1, -1] (f-e)
            # C00 += M4, C10 += M4.
            # W: [1, 0, 1, 0]
            
            # M5 = (a+b)h
            # U: [1, 1, 0, 0]
            # V: [0, 0, 0, 1] (h)
            # C00 -= M5, C01 += M5.
            # W: [-1, 1, 0, 0]
            
            # M6 = (c-a)(e+f)
            # U: [-1, 0, 1, 0]
            # V: [1, 0, 1, 0] (e+f)
            # C11 += M6.
            # W: [0, 0, 0, 1]
            
            # M7 = (b-d)(g+h)
            # U: [0, 1, 0, -1]
            # V: [0, 1, 0, 1] (g+h)
            # C00 += M7, C11 += M7.
            # W: [1, 0, 0, 1]
            
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
            ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 1])  # M7
        ]
        
        col = 0
        for q in range(5):
            for a_v, b_v, c_v in terms:
                for i in range(4):
                    U[i][col] = a_v[i]
                
                # b_v order: e, g, f, h
                # V[2q] = e
                # V[10+2q] = g
                # V[2q+1] = f
                # V[10+2q+1] = h
                V[2*q][col] = b_v[0]
                V[10+2*q][col] = b_v[1]
                V[2*q+1][col] = b_v[2]
                V[10+2*q+1][col] = b_v[3]
                
                # c_v order: i, j, k, l
                # W[4q] = i
                # W[4q+1] = j
                # W[4q+2] = k
                # W[4q+3] = l
                W[4*q][col] = c_v[0]
                W[4*q+1][col] = c_v[1]
                W[4*q+2][col] = c_v[2]
                W[4*q+3][col] = c_v[3]
                
                col += 1
                
        return U, V, W

    U, V, W = build_strassen_solution()
    
    if not verify_decomposition(U, V, W):
        raise ValueError("Decomposition failed")

    return {"U": U, "V": V, "W": W}
