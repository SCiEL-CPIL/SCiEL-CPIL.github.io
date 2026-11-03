import numpy as np

def construct():
    n, m, p = 3, 3, 3
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
        col += 1

    # Group 1: Strassen for C[0,0], C[0,1], C[1,0], C[1,1]
    # M1 = (A00+A11)(B00+B11)
    # A00 is U[0], A11 is U[4]
    # B00 is V[0], B11 is V[4]
    # C00 is W[0], C01 is W[3], C10 is W[1], C11 is W[4]
    
    # We need to handle the linear combinations.
    # Since U, V, W are matrices of coefficients, we can just add the coefficients.
    # But each "multiplication" corresponds to a column.
    # So for M1, we need a column where U has 1 at A00 and A11, V has 1 at B00 and B11, and W has the appropriate signs for C00, C01, etc.
    # Wait, the decomposition is T = sum u_r v_r w_r.
    # So for each multiplication, we define a vector u, v, w.
    
    # Let's re-implement add_term to take lists of indices and coefficients.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_coeffs:
            U[idx][col] += val
        for idx, val in v_coeffs:
            V[idx][col] += val
        for idx, val in w_coeffs:
            W[idx][col] += val
        col += 1

    # Indices:
    # A[i,j] -> i*m+j
    # B[j,k] -> j*p+k
    # C[i,k] -> k*n+i (Note: task says W[k*n+i])
    
    # Strassen Terms
    # 1. M1 = (A00+A11)(B00+B11)
    # Contributes to C00 (+1), C01 (0), C10 (0), C11 (-1)
    # Wait, C00 = M1 + M4 - M5 + M7. C11 = M1 - M2 + M3 + M6.
    # So M1 contributes +1 to C00 and +1 to C11.
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (4, 1)])
    
    # 2. M2 = (A10+A11)B00
    # A10 is U[3], A11 is U[4]. B00 is V[0].
    # Contributes to C10 (+1), C11 (-1).
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
    
    # 3. M3 = A00(B10-B11)
    # A00 is U[0]. B10 is V[3], B11 is V[4].
    # Contributes to C01 (+1), C11 (+1).
    add_term([(0, 1)], [(3, 1), (4, -1)], [(3, 1), (4, 1)])
    
    # 4. M4 = A11(B01-B00)
    # A11 is U[4]. B01 is V[1], B00 is V[0].
    # Contributes to C00 (+1), C10 (+1).
    add_term([(4, 1)], [(1, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # 5. M5 = (A00+A01)B11
    # A00 is U[0], A01 is U[1]. B11 is V[4].
    # Contributes to C00 (-1), C01 (+1).
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (3, 1)])
    
    # 6. M6 = (A10-A00)(B00+B01)
    # A10 is U[3], A00 is U[0]. B00 is V[0], B01 is V[1].
    # Contributes to C11 (+1).
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)])
    
    # 7. M7 = (A01-A11)(B10+B11)
    # A01 is U[1], A11 is U[4]. B10 is V[3], B11 is V[4].
    # Contributes to C00 (+1).
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)])
    
    # Group 2: A22 B21 (A[0..1, 0..1] * B[0..1, 2]) -> C[0..1, 2]
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    # A00=U[0], A01=U[1], A10=U[3], A11=U[4]
    # B02=V[2], B12=V[5]
    # C02=W[6], C12=W[7]
    
    # 8. A00 B02 -> C02
    add_term([(0, 1)], [(2, 1)], [(6, 1)])
    # 9. A01 B12 -> C02
    add_term([(1, 1)], [(5, 1)], [(6, 1)])
    # 10. A10 B02 -> C12
    add_term([(3, 1)], [(2, 1)], [(7, 1)])
    # 11. A11 B12 -> C12
    add_term([(4, 1)], [(5, 1)], [(7, 1)])
    
    # Group 3: A12 B22 (A[2, 0..1] * B[0..1, 0..1]) -> C[2, 0..1]
    # C20 = A20 B00 + A21 B10
    # C21 = A20 B01 + A21 B11
    # A20=U[6], A21=U[7]
    # B00=V[0], B10=V[3], B01=V[1], B11=V[4]
    # C20=W[2], C21=W[5]
    
    # 12. A20 B00 -> C20
    add_term([(6, 1)], [(0, 1)], [(2, 1)])
    # 13. A21 B10 -> C20
    add_term([(7, 1)], [(3, 1)], [(2, 1)])
    # 14. A20 B01 -> C21
    add_term([(6, 1)], [(1, 1)], [(5, 1)])
    # 15. A21 B11 -> C21
    add_term([(7, 1)], [(4, 1)], [(5, 1)])
    
    # Group 4: A21 B11 (A[0..1, 2] * B[2, 2]) -> C[0..1, 2]
    # C02 += A02 B22
    # C12 += A12 B22
    # A02=U[2], A12=U[5]
    # B22=V[8]
    # C02=W[6], C12=W[7]
    
    # 16. A02 B22 -> C02
    add_term([(2, 1)], [(8, 1)], [(6, 1)])
    # 17. A12 B22 -> C12
    add_term([(5, 1)], [(8, 1)], [(7, 1)])
    
    # Group 5: A11 B12 (A[2, 2] * B[2, 0..1]) -> C[2, 0..1]
    # C20 += A22 B20
    # C21 += A22 B21
    # A22=U[8]
    # B20=V[6], B21=V[7]
    # C20=W[2], C21=W[5]
    
    # 18. A22 B20 -> C20
    add_term([(8, 1)], [(6, 1)], [(2, 1)])
    # 19. A22 B21 -> C21
    add_term([(8, 1)], [(7, 1)], [(5, 1)])
    
    # Group 6: A12 B21 (A[2, 0..1] * B[0..1, 2]) -> C[2, 2]
    # C22 = A20 B02 + A21 B12
    # A20=U[6], A21=U[7]
    # B02=V[2], B12=V[5]
    # C22=W[8]
    
    # 20. A20 B02 -> C22
    add_term([(6, 1)], [(2, 1)], [(8, 1)])
    # 21. A21 B12 -> C22
    add_term([(7, 1)], [(5, 1)], [(8, 1)])
    
    # Group 7: A11 B11 (A[2, 2] * B[2, 2]) -> C[2, 2]
    # C22 += A22 B22
    # A22=U[8], B22=V[8]
    # C22=W[8]
    
    # 22. A22 B22 -> C22
    add_term([(8, 1)], [(8, 1)], [(8, 1)])
    
    return {"U": U, "V": V, "W": W}
