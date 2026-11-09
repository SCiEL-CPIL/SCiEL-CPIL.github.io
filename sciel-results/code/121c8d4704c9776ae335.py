import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Try Laderman's algorithm for 3x3, applied twice
    # Rank 23 for each 3x3, total 46
    
    # Define Laderman's 23 products for 3x3 matrix multiplication
    # A is 3x3, B is 3x3
    # Products are of the form L(A) * N(B)
    # We define the coefficients for A and B for each product
    
    # List of (coeffs_A, coeffs_B) for each of the 23 products
    # coeffs are lists of length 9 (flattened 3x3)
    # We use 1, -1, 0
    
    products = []
    
    # 1. a00 b00
    products.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # 2. a01 (b10+b20)
    products.append(([0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,1,0,0]))
    # 3. (a00+a10) b11
    products.append(([1,0,0,1,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # 4. (a01+a11) b12
    products.append(([0,1,0,0,1,0,0,0,0], [0,0,0,0,0,1,0,0,0]))
    # 5. a11 (b00+b01)
    products.append(([0,0,0,0,1,0,0,0,0], [1,1,0,0,0,0,0,0,0]))
    # 6. (a00+a01) (b10+b11)
    products.append(([1,1,0,0,0,0,0,0,0], [0,0,0,1,1,0,0,0,0]))
    # 7. a10 (b01-b00)
    products.append(([0,0,0,1,0,0,0,0,0], [-1,1,0,0,0,0,0,0,0]))
    # 8. a20 (b00+b01)
    products.append(([0,0,0,0,0,0,1,0,0], [1,1,0,0,0,0,0,0,0]))
    # 9. a21 (b10+b11)
    products.append(([0,0,0,0,0,0,0,1,0], [0,0,0,1,1,0,0,0,0]))
    # 10. (a20+a21) (b11+b12)
    products.append(([0,0,0,0,0,0,1,1,0], [0,0,0,0,1,1,0,0,0]))
    # 11. a22 (b00+b01)
    products.append(([0,0,0,0,0,0,0,0,1], [1,1,0,0,0,0,0,0,0]))
    # 12. a22 (b10+b11)
    products.append(([0,0,0,0,0,0,0,0,1], [0,0,0,1,1,0,0,0,0]))
    # 13. (a20+a22) (b11+b12)
    products.append(([0,0,0,0,0,0,1,0,1], [0,0,0,0,1,1,0,0,0]))
    # 14. (a21+a22) (b10+b12)
    products.append(([0,0,0,0,0,0,0,1,1], [0,0,0,1,0,1,0,0,0]))
    # 15. a02 (b00+b01)
    products.append(([0,0,1,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0]))
    # 16. a02 (b10+b11)
    products.append(([0,0,1,0,0,0,0,0,0], [0,0,0,1,1,0,0,0,0]))
    # 17. (a02+a12) (b11+b12)
    products.append(([0,0,1,0,0,1,0,0,0], [0,0,0,0,1,1,0,0,0]))
    # 18. (a02+a12) (b10+b12)
    products.append(([0,0,1,0,0,1,0,0,0], [0,0,0,1,0,1,0,0,0]))
    # 19. a12 (b00+b01)
    products.append(([0,0,0,0,0,1,0,0,0], [1,1,0,0,0,0,0,0,0]))
    # 20. a12 (b10+b11)
    products.append(([0,0,0,0,0,1,0,0,0], [0,0,0,1,1,0,0,0,0]))
    # 21. (a00+a02) (b20+b21)
    products.append(([1,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,1,0]))
    # 22. (a01+a02) (b20+b22)
    products.append(([0,1,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,1]))
    # 23. (a10+a12) (b21+b22)
    products.append(([0,0,0,1,0,1,0,0,0], [0,0,0,0,0,0,0,1,1]))
    
    rank = 46
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(18)]
    W = [[0]*rank for _ in range(18)]
    
    # We need to map the Laderman products to the C matrix
    # C_ij = sum_k P_k M_k
    # The P_k coefficients are not stored in U, V, W directly, but in W
    # Wait, the decomposition is C = sum_r U[:,r] V[:,r] W[:,r]^T ? No.
    # The tensor is T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # Here i is row of A, j is col of A (row of B), k is col of B (col of C)
    # So U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # For each of the 23 Laderman products, we have a linear form in A and a linear form in B.
    # Let L_A be the coeffs for A, L_B be the coeffs for B.
    # The product is M = L_A . A * L_B . B
    # This M contributes to C via the coefficients P_{ij}
    # C_{ij} += P_{ij} M
    
    # I need the P_{ij} coefficients for each of the 23 products.
    # This is the hard part. I don't have them memorized.
    # However, I can compute them by solving a linear system?
    # Or I can just use the schoolbook algorithm if I can't find them.
    
    # Since I don't have the P_{ij} coefficients, I cannot construct W correctly.
    # I will fall back to schoolbook.
    
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
