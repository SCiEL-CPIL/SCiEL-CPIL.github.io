import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

    # Try Rank 46 (2 * Laderman 23)
    try:
        if time.time() - start_time > 100:
            return get_schoolbook()
            
        # Hardcoded Laderman 23-product algorithm for 3x3x3
        # U_23[i*3+j][l] = coefficient of A[i,j] in product l
        # V_23[j*3+k][l] = coefficient of B[j,k] in product l
        # W_23[k*3+i][l] = coefficient of product l in C[i,k]
        
        # Using the standard Laderman algorithm with explicit coefficients
        # Each product l has: P_l = (sum_ij u_ij^l * A_ij) * (sum_jk v_jk^l * B_jk)
        # And C_ik = sum_l w_ik^l * P_l
        
        # Define as lists of (i, j, coeff) for A and (j, k, coeff) for B
        products = []
        
        def add_product(a_terms, b_terms, w_terms):
            products.append((a_terms, b_terms, w_terms))
        
        # Laderman's 23 products - standard formulation
        # a_terms: [(i, j, coeff), ...]
        # b_terms: [(j, k, coeff), ...]
        # w_terms: [(i, k, coeff), ...]
        
        # P0: A00 * B00
        add_product([(0,0,1)], [(0,0,1)], [(0,0,1)])
        # P1: A01 * B11
        add_product([(0,1,1)], [(1,1,1)], [(0,1,1)])
        # P2: A10 * B00
        add_product([(1,0,1)], [(0,0,1)], [(1,0,1)])
        # P3: A11 * B11
        add_product([(1,1,1)], [(1,1,1)], [(1,1,1)])
        # P4: A22 * B22
        add_product([(2,2,1)], [(2,2,1)], [(2,2,1)])
        # P5: (A00+A11-A22) * B22
        add_product([(0,0,1),(1,1,1),(2,2,-1)], [(2,2,1)], [(2,0,1),(2,1,1),(2,2,-1)])
        # P6: (A00+A11-A22) * B01
        add_product([(0,0,1),(1,1,1),(2,2,-1)], [(0,1,1)], [(0,0,1),(1,0,1),(2,0,-1)])
        # P7: (A01+A10) * B01
        add_product([(0,1,1),(1,0,1)], [(0,1,1)], [(0,0,1),(1,0,1)])
        # P8: (A01+A10) * B10
        add_product([(0,1,1),(1,0,1)], [(1,0,1)], [(0,1,1),(1,1,1)])
        # P9: (A00+A11+A22-A01-A10) * B10
        add_product([(0,0,1),(1,1,1),(2,2,1),(0,1,-1),(1,0,-1)], [(1,0,1)], [(0,1,1),(1,1,1),(2,1,1)])
        # P10: (A00+A11+A22+A01+A10) * B00
        add_product([(0,0,1),(1,1,1),(2,2,1),(0,1,1),(1,0,1)], [(0,0,1)], [(0,0,1),(1,0,1),(2,0,1)])
        # P11: (A00-A11) * B01
        add_product([(0,0,1),(1,1,-1)], [(0,1,1)], [(0,0,1),(1,0,-1)])
        # P12: (A00-A11) * B10
        add_product([(0,0,1),(1,1,-1)], [(1,0,1)], [(0,1,1),(1,1,-1)])
        # P13: (A01-A10) * B01
        add_product([(0,1,1),(1,0,-1)], [(0,1,1)], [(0,0,1),(1,0,-1)])
        # P14: (A01-A10) * B10
        add_product([(0,1,1),(1,0,-1)], [(1,0,1)], [(0,1,1),(1,1,-1)])
        # P15: (A00+A01+A10+A11) * B11
        add_product([(0,0,1),(0,1,1),(1,0,1),(1,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)])
        # P16: (A00+A01+A10+A11) * B00
        add_product([(0,0,1),(0,1,1),(1,0,1),(1,1,1)], [(0,0,1)], [(0,0,1),(1,0,1)])
        # P17: (A00+A10+A22) * B22
        add_product([(0,0,1),(1,0,1),(2,2,1)], [(2,2,1)], [(2,0,1),(2,1,1)])
        # P18: (A01+A11+A22) * B22
        add_product([(0,1,1),(1,1,1),(2,2,1)], [(2,2,1)], [(2,0,1),(2,1,1)])
        # P19: (A00+A10+A22) * B01
        add_product([(0,0,1),(1,0,1),(2,2,1)], [(0,1,1)], [(0,0,1),(1,0,1)])
        # P20: (A01+A11+A22) * B10
        add_product([(0,1,1),(1,1,1),(2,2,1)], [(1,0,1)], [(0,1,1),(1,1,1)])
        # P21: (A00+A01+A10+A11+A22) * B00
        add_product([(0,0,1),(0,1,1),(1,0,1),(1,1,1),(2,2,1)], [(0,0,1)], [(0,0,1),(1,0,1),(2,0,1)])
        # P22: (A00+A01+A10+A11+A22) * B11
        add_product([(0,0,1),(0,1,1),(1,0,1),(1,1,1),(2,2,1)], [(1,1,1)], [(0,1,1),(1,1,1),(2,1,1)])

        # Build U, V, W matrices for the 23 products
        U_23 = [[0] * 23 for _ in range(9)]
        V_23 = [[0] * 23 for _ in range(9)]
        W_23 = [[0] * 23 for _ in range(9)]
        
        for l in range(23):
            a_terms, b_terms, w_terms = products[l]
            for i, j, coeff in a_terms:
                U_23[i * 3 + j][l] = coeff
            for j, k, coeff in b_terms:
                V_23[j * 3 + k][l] = coeff
            for i, k, coeff in w_terms:
                W_23[k * 3 + i][l] = coeff

        # Verify the 3x3x3 decomposition before using it
        rng = np.random.default_rng(42)
        A_test = rng.integers(-5, 5, size=(3, 3))
        B_test = rng.integers(-5, 5, size=(3, 3))
        C_test = np.matmul(A_test, B_test)
        
        C_calc = np.zeros((3, 3))
        for l in range(23):
            ua = 0
            for i in range(3):
                for j in range(3):
                    ua += U_23[i*3+j][l] * A_test[i,j]
            vb = 0
            for j in range(3):
                for k in range(3):
                    vb += V_23[j*3+k][l] * B_test[j,k]
            prod = ua * vb
            for k in range(3):
                for i in range(3):
                    C_calc[i,k] += prod * W_23[k*3+i][l]
        
        if not np.allclose(C_test, C_calc, atol=1e-6):
            # Laderman verification failed, fall back to schoolbook
            return get_schoolbook()
            
        # Construct full U, V, W for rank 46
        rank = 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 1: k in 0..2
        for l in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*3+j][l] = U_23[i*3+j][l]
            for j in range(3):
                for k in range(3):
                    V[j*3+k][l] = V_23[j*3+k][l]
            for k in range(3):
                for i in range(3):
                    W[k*3+i][l] = W_23[k*3+i][l]
                    
        # Block 2: k in 3..5
        for l in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*3+j][l+23] = U_23[i*3+j][l]
            for j in range(3):
                for k in range(3):
                    V[j*3+(k+3)][l+23] = V_23[j*3+k][l]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i][l+23] = W_23[k*3+i][l]
                    
        # Verify full decomposition
        if time.time() - start_time > 110:
            return get_schoolbook()
            
        A_test2 = rng.integers(-5, 5, size=(3, 3))
        B_test2 = rng.integers(-5, 5, size=(3, 6))
        C_test2 = np.matmul(A_test2, B_test2)
        
        C_calc2 = np.zeros((3, 6))
        for r in range(rank):
            ua = 0
            for i in range(3):
                for j in range(3):
                    ua += U[i*3+j][r] * A_test2[i,j]
            vb = 0
            for j in range(3):
                for k in range(6):
                    vb += V[j*3+k][r] * B_test2[j,k]
            prod = ua * vb
            for k in range(6):
                for i in range(3):
                    C_calc2[i,k] += prod * W[k*3+i][r]
                    
        if not np.allclose(C_test2, C_calc2, atol=1e-6):
            return get_schoolbook()
            
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        return get_schoolbook()
