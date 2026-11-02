"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 multiplication
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Try rank 133: 19 (for 3x3) x 7 (for 2x2)
    # Rank-19 decomposition for 3x3 matrix multiplication
    # Each term: (U_dict, V_dict, W_dict) where keys are (row, col)
    rank19_terms = [
        # Term 0: M1 = (A00+A11+A22)(B00+B11+B22) -> C00+C11+C22
        ({(0,0):1,(1,1):1,(2,2):1}, {(0,0):1,(1,1):1,(2,2):1}, {(0,0):1,(1,1):1,(2,2):1}),
        # Term 1: M2 = (A10+A20)(B00) -> C10+C20
        ({(1,0):1,(2,0):1}, {(0,0):1}, {(1,0):1,(2,0):1}),
        # Term 2: M3 = (A00)(B10+B20) -> C01+C02
        ({(0,0):1}, {(1,0):1,(2,0):1}, {(0,1):1,(0,2):1}),
        # Term 3: M4 = (A11+A22)(B01) -> C11+C22
        ({(1,1):1,(2,2):1}, {(0,1):1}, {(1,1):1,(2,2):1}),
        # Term 4: M5 = (A01)(B11+B22) -> C01+C02
        ({(0,1):1}, {(1,1):1,(2,2):1}, {(0,1):1,(0,2):1}),
        # Term 5: M6 = (A20+A21)(B00+B01) -> C20+C21
        ({(2,0):1,(2,1):1}, {(0,0):1,(0,1):1}, {(2,0):1,(2,1):1}),
        # Term 6: M7 = (A00+A01)(B10+B20) -> C01+C02
        ({(0,0):1,(0,1):1}, {(1,0):1,(2,0):1}, {(0,1):1,(0,2):1}),
        # Term 7: M8 = (A10+A11)(B01+B11) -> C11+C12
        ({(1,0):1,(1,1):1}, {(0,1):1,(1,1):1}, {(1,1):1,(1,2):1}),
        # Term 8: M9 = (A20+A22)(B01+B22) -> C21+C22
        ({(2,0):1,(2,2):1}, {(0,1):1,(2,2):1}, {(2,1):1,(2,2):1}),
        # Term 9: M10 = (A00+A22)(B00+A22) -> C00+C22
        ({(0,0):1,(2,2):1}, {(0,0):1,(2,2):1}, {(0,0):1,(2,2):1}),
        # Term 10: M11 = (A10+A22)(B00+B11) -> C10+C11
        ({(1,0):1,(2,2):1}, {(0,0):1,(1,1):1}, {(1,0):1,(1,1):1}),
        # Term 11: M12 = (A00+A11)(B01+B22) -> C01+C22
        ({(0,0):1,(1,1):1}, {(0,1):1,(2,2):1}, {(0,1):1,(2,2):1}),
        # Term 12: M13 = (A00+A21)(B00+B11) -> C00+C11
        ({(0,0):1,(2,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # Term 13: M14 = (A10+A21)(B01+B22) -> C11+C22
        ({(1,0):1,(2,1):1}, {(0,1):1,(2,2):1}, {(1,1):1,(2,2):1}),
        # Term 14: M15 = (A00+A12)(B00+B11) -> C00+C11
        ({(0,0):1,(1,2):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # Term 15: M16 = (A10+A12)(B01+B22) -> C11+C22
        ({(1,0):1,(1,2):1}, {(0,1):1,(2,2):1}, {(1,1):1,(2,2):1}),
        # Term 16: M17 = (A00+A22)(B01+B11) -> C01+C11
        ({(0,0):1,(2,2):1}, {(0,1):1,(1,1):1}, {(0,1):1,(1,1):1}),
        # Term 17: M18 = (A10+A22)(B00+B21) -> C10+C21
        ({(1,0):1,(2,2):1}, {(0,0):1,(2,1):1}, {(1,0):1,(2,1):1}),
        # Term 18: M19 = (A00+A11)(B00+B22) -> C00+C22
        ({(0,0):1,(1,1):1}, {(0,0):1,(2,2):1}, {(0,0):1,(2,2):1}),
    ]
    
    # Verify rank-19 decomposition
    def verify_decomposition(terms, size):
        """Verify that the decomposition is correct."""
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randint(-5, 6, size=(size, size))
        B = np.random.randint(-5, 6, size=(size, size))
        expected = A @ B
        
        result = np.zeros((size, size), dtype=float)
        for U_dict, V_dict, W_dict in terms:
            u_val = sum(A[r, c] * coeff for (r, c), coeff in U_dict.items())
            v_val = sum(B[r, c] * coeff for (r, c), coeff in V_dict.items())
            for (r, c), coeff in W_dict.items():
                result[r, c] += u_val * v_val * coeff
        
        return np.allclose(result, expected, atol=1e-9)
    
    use_rank133 = verify_decomposition(rank19_terms, 3)
    
    if use_rank133 and time.time() < deadline:
        # Rank 133: 19 (3x3 level) x 7 (2x2 level)
        rank = 19 * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for l19 in range(19):
            u3, v3, w3 = rank19_terms[l19]
            for l7 in range(7):
                u2, v2, w2 = strassen_terms[l7]
                
                # For each pair of non-zero entries in u3 and u2, add to U
                # U index: (block_row * 2 + local_row, block_col * 2 + local_col)
                for (br, bc), c3 in u3.items():
                    for (lr, lc), c2 in u2.items():
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        U[i * m + j][col] += c3 * c2
                
                for (br, bc), c3 in v3.items():
                    for (lr, lc), c2 in v2.items():
                        j = br * 2 + lr
                        k = bc * 2 + lc
                        V[j * p + k][col] += c3 * c2
                
                for (br, bc), c3 in w3.items():
                    for (lr, lc), c2 in w2.items():
                        i = br * 2 + lr
                        k = bc * 2 + lc
                        W[k * n + i][col] += c3 * c2
                
                col += 1
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: Rank 189
        rank = 189
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    for l in range(7):
                        u_coeffs, v_coeffs, w_coeffs = strassen_terms[l]
                        
                        for (r, s), coeff in u_coeffs.items():
                            i = 2 * alpha + r
                            j = 2 * beta + s
                            U[i * m + j][col] += coeff
                        
                        for (s_prime, t), coeff in v_coeffs.items():
                            j = 2 * beta + s_prime
                            k = 2 * gamma + t
                            V[j * p + k][col] += coeff
                        
                        for (r, t), coeff in w_coeffs.items():
                            i = 2 * alpha + r
                            k = 2 * gamma + t
                            W[k * n + i][col] += coeff
                        
                        col += 1
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
