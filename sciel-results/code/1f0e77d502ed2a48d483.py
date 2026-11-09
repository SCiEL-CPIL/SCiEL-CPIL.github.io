import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define Schoolbook (Rank 27) as fallback
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
    
    # 2. Define Laderman (Rank 23) candidate
    # Based on Laderman (1976)
    # Indices: A[0..8], B[0..8], C[0..8]
    # A: [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # B: [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C: [[c00, c01, c02], [c10, c11, c12], [c20, c21, c22]]
    
    rank_laderman = 23
    U_lad = [[0] * rank_laderman for _ in range(n * m)]
    V_lad = [[0] * rank_laderman for _ in range(m * p)]
    W_lad = [[0] * rank_laderman for _ in range(p * n)]
    
    # Helper to add coefficients
    def add_term(u_coeffs, v_coeffs, w_coeffs, r):
        # u_coeffs: dict {idx: val} for A
        # v_coeffs: dict {idx: val} for B
        # w_coeffs: dict {idx: val} for C
        for idx, val in u_coeffs.items():
            U_lad[idx][r] += val
        for idx, val in v_coeffs.items():
            V_lad[idx][r] += val
        for idx, val in w_coeffs.items():
            W_lad[idx][r] += val

    # Laderman's 23 terms
    # 1. a00 b00
    add_term({0:1}, {0:1}, {0:1}, 0)
    
    # 2. a00 (b10 + b20 - b00)
    add_term({0:1}, {3:1, 6:1, 0:-1}, {0:1, 3:1}, 1)
    
    # 3. (a00 + a01) b00
    add_term({0:1, 1:1}, {0:1}, {0:1, 3:1}, 2)
    
    # 4. a11 (b01 + b11 - b00)
    add_term({4:1}, {1:1, 4:1, 0:-1}, {1:1, 4:1}, 3)
    
    # 5. (a11 + a12) (b11 + b21 - b10)
    add_term({4:1, 5:1}, {4:1, 7:1, 3:-1}, {1:1, 4:1, 7:1}, 4)
    
    # 6. a22 (b12 + b22 - b11)
    add_term({8:1}, {5:1, 8:1, 4:-1}, {2:1, 5:1}, 5)
    
    # 7. (a22 + a02) (b22 + b02 - b11)
    add_term({8:1, 2:1}, {8:1, 2:1, 4:-1}, {2:1, 5:1, 8:1}, 6)
    
    # 8. a01 (b20 - b00)
    add_term({1:1}, {6:1, 0:-1}, {3:1}, 7)
    
    # 9. a12 (b01 - b00)
    add_term({5:1}, {1:1, 0:-1}, {4:1}, 8)
    
    # 10. a20 (b12 - b11)
    add_term({6:1}, {5:1, 4:-1}, {5:1}, 9)
    
    # 11. a02 (b21 - b11)
    add_term({2:1}, {7:1, 4:-1}, {8:1}, 10)
    
    # 12. (a00 + a11) (b10 + b20 - b00)
    add_term({0:1, 4:1}, {3:1, 6:1, 0:-1}, {0:1, 1:1}, 11)
    
    # 13. (a11 + a22) (b01 + b11 - b00)
    add_term({4:1, 8:1}, {1:1, 4:1, 0:-1}, {1:1, 2:1}, 12)
    
    # 14. (a22 + a00) (b12 + b22 - b11)
    add_term({8:1, 0:1}, {5:1, 8:1, 4:-1}, {2:1, 5:1}, 13)
    
    # 15. (a01 + a12) (b20 + b01 - b00)
    add_term({1:1, 5:1}, {6:1, 1:1, 0:-1}, {0:1, 3:1}, 14)
    
    # 16. (a12 + a20) (b01 + b12 - b11)
    add_term({5:1, 6:1}, {1:1, 5:1, 4:-1}, {4:1, 5:1}, 15)
    
    # 17. (a20 + a01) (b12 + b20 - b11)
    add_term({6:1, 1:1}, {5:1, 6:1, 4:-1}, {5:1, 8:1}, 16)
    
    # 18. (a00 + a22) (b10 + b02 - b00)
    add_term({0:1, 8:1}, {3:1, 2:1, 0:-1}, {0:1, 2:1}, 17)
    
    # 19. (a01 + a22) (b20 + b11 - b00)
    add_term({1:1, 8:1}, {6:1, 4:1, 0:-1}, {3:1, 5:1}, 18)
    
    # 20. (a12 + a00) (b01 + b22 - b00)
    add_term({5:1, 0:1}, {1:1, 8:1, 0:-1}, {1:1, 2:1}, 19)
    
    # 21. (a20 + a12) (b12 + b02 - b11)
    add_term({6:1, 5:1}, {5:1, 2:1, 4:-1}, {2:1, 5:1}, 20)
    
    # 22. (a01 + a20) (b20 + b12 - b00)
    add_term({1:1, 6:1}, {6:1, 5:1, 0:-1}, {3:1, 5:1}, 21)
    
    # 23. (a00 + a12) (b10 + b22 - b00)
    add_term({0:1, 5:1}, {3:1, 8:1, 0:-1}, {0:1, 5:1}, 22)

    # 3. Verify Laderman
    def verify(U, V, W, rank):
        # Generate random matrices
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_comp = np.zeros((n, p))
        for r in range(rank):
            # Compute scalar factors
            u_val = 0
            for i in range(n):
                for j in range(m):
                    u_val += U[i*m + j][r] * A[i, j]
            
            v_val = 0
            for j in range(m):
                for k in range(p):
                    v_val += V[j*p + k][r] * B[j, k]
            
            # Add to C
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k*n + i][r] * u_val * v_val
                    
        return np.allclose(C_true, C_comp, atol=1e-9)

    if verify(U_lad, V_lad, W_lad, rank_laderman):
        return {"U": U_lad, "V": V_lad, "W": W_lad}
    else:
        # Fallback to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
