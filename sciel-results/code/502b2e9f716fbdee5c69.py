# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 120

    # We will attempt to construct a rank-23 decomposition for 3x3 matrix multiplication.
    # This is the Laderman algorithm.
    # The tensor T_{ijk} = delta_{jk} (where i is row of A, j is col of A / row of B, k is col of B)
    # We seek U, V, W such that T_{ijk} = sum_{r} U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
    # Note the W index convention: W[k*n + i, r].
    # So the term for output C[i, k] is indexed by idx = k*n + i.
    
    # Laderman's algorithm has 23 products.
    # Let's define the 23 products (U, V) and the corresponding W coefficients.
    
    # The products are linear combinations of entries of A and B.
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    
    # Let's use the explicit Laderman formulas.
    # Reference: "A 23-multiplication algorithm for 3x3 matrix multiplication" by Laderman (1976)
    # Or similar sources providing the explicit 23 terms.
    
    # One known set of 23 terms:
    # Let's try to construct them.
    # Actually, finding the exact 23 terms from memory is error-prone.
    # A safer approach might be to use a known correct rank-23 decomposition if I can recall it.
    # Or, I can try to use a rank-26 or rank-27 if I can't get 23 right.
    # But the goal is to improve on 27. Rank 23 is the known minimum.
    
    # Let's try to implement the Laderman algorithm.
    # The 23 products are:
    # M1 = (a00 + a11 + a22) * (b00 + b11 + b22)
    # M2 = (a01 + a10 + a22) * (b01 + b10 + b22)
    # ... this is getting complicated.
    
    # Let's look for a simpler, verified rank-23 decomposition.
    # Actually, I can use the "Laderman 23" algorithm which is well-documented.
    
    # Here is a set of 23 terms from a reliable source (e.g., Wikipedia or a paper):
    # Let's denote the entries of A as A_{ij} and B as B_{jk}.
    # The 23 products P_r are:
    
    # I will define the terms as dictionaries.
    # U: {(i,j): coeff}
    # V: {(j,k): coeff}
    # W: {(i,k): coeff} -> stored at index k*n+i
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
        
    # Laderman's 23 products:
    # 1. P1 = (A00 + A11 + A22)(B00 + B11 + B22)
    #    U: A00:1, A11:1, A22:1
    #    V: B00:1, B11:1, B22:1
    #    W: C00:1, C11:1, C22:1, C01:-1, C10:-1, C21:-1, C12:-1, C20:-1, C20:-1? No.
    #    This term contributes to the diagonal and off-diagonals.
    #    Actually, P1 is not a simple product in the tensor decomposition sense unless it's the only term.
    #    In tensor decomposition, each term is a product of a linear form in A and a linear form in B.
    #    The coefficient in W is 1 for the specific (i,k) it contributes to.
    #    Wait, the W matrix entries can be -1, 0, 1.
    #    So a single term P_r = L_r(A) * M_r(B) contributes to multiple C_{ik} entries if W_{ik, r} is non-zero.
    #    But usually, each term is designed to contribute to specific entries.
    
    # Let's use a known correct set of 23 terms.
    # I will use the terms from the "Laderman" algorithm as described in many optimization papers.
    
    # Term 1:
    # U: A00, A11, A22 (coeff 1)
    # V: B00, B11, B22 (coeff 1)
    # W: C00:1, C11:1, C22:1, C01:-1, C10:-1, C12:-1, C20:-1, C21:-1, C02:-1? 
    # This is too risky to guess.
    
    # Alternative: Use a rank-27 schoolbook decomposition which is definitely correct, 
    # and then try to optimize it to rank 23 using a solver? 
    # No, that's too complex for 120s.
    
    # Let's try to find a simple rank-23 decomposition online in my training data.
    # One common set is:
    # 
    # P1 = (A00+A11+A22)(B00+B11+B22)
    # P2 = (A01+A10+A22)(B01+B10+B22)
    # P3 = (A02+A12+A20)(B02+B12+B20)
    # P4 = (A00+A11+A22)(B01+B10+B22)
    # P5 = (A00+A11+A22)(B02+B12+B20)
    # P6 = (A01+A10+A22)(B00+B11+B22)
    # P7 = (A01+A10+A22)(B02+B12+B20)
    # P8 = (A02+A12+A20)(B00+B11+B22)
    # P9 = (A02+A12+A20)(B01+B10+B22)
    # P10 = (A00+A11)(B00+B11)
    # P11 = (A00+A11)(B02+B12)
    # P12 = (A00+A11)(B20+B21)
    # P13 = (A01+A10)(B00+B11)
    # P14 = (A01+A10)(B02+B12)
    # P15 = (A01+A10)(B20+B21)
    # P16 = (A02+A20)(B00+B22)
    # P17 = (A02+A20)(B01+B21)
    # P18 = (A02+A20)(B10+B12)
    # P19 = (A12+A21)(B00+B22)
    # P20 = (A12+A21)(B01+B21)
    # P21 = (A12+A21)(B10+B12)
    # P22 = A22*B22
    # P23 = A12*B21? No.
    
    # This is also not matching the standard Laderman.
    
    # Let's go back to the safe rank-27.
    # The current program has rank 27.
    # If I can't find a correct rank 23, I should stick with 27.
    # But the task asks for "rank-minimal".
    # The known minimum is 23.
    
    # I will try to implement the Laderman algorithm by looking up the specific 23 terms.
    # From "Matrix Multiplication Algorithms" by Laderman:
    # 
    # Let's use the terms from this source:
    # https://www.cs.cornell.edu/courses/cs682/2005fa/lectures/lec11.pdf (or similar)
    # 
    # Actually, I can use the "Laderman 23" algorithm from the Wikipedia page on "Strassen algorithm" or "Matrix multiplication".
    # 
    # Let's try to construct it using the following 23 products:
    # 
    # M1 = (A00 + A11 + A22) * (B00 + B11 + B22)
    # M2 = (A01 + A10 + A22) * (B01 + B10 + B22)
    # M3 = (A02 + A12 + A20) * (B02 + B12 + B20)
    # M4 = (A00 + A11) * (B00 + B11)
    # M5 = (A00 + A11) * (B02 + B12)
    # M6 = (A00 + A11) * (B20 + B21)
    # M7 = (A01 + A10) * (B00 + B11)
    # M8 = (A01 + A10) * (B02 + B12)
    # M9 = (A01 + A10) * (B20 + B21)
    # M10 = (A02 + A20) * (B00 + B22)
    # M11 = (A02 + A20) * (B01 + B21)
    # M12 = (A02 + A20) * (B10 + B12)
    # M13 = (A12 + A21) * (B00 + B22)
    # M14 = (A12 + A21) * (B01 + B21)
    # M15 = (A12 + A21) * (B10 + B12)
    # M16 = A22 * B22
    # M17 = A12 * B21
    # M18 = A21 * B12
    # M19 = A02 * B20
    # M20 = A20 * B02
    # M21 = A10 * B01
    # M22 = A01 * B10
    # M23 = A00 * B00? No, that's schoolbook.
    
    # This is not Laderman.
    
    # Let's try a different approach.
    # I will use the rank-27 schoolbook decomposition, which is guaranteed to be correct.
    # This ensures a score of 1.0 (or better if the seed was worse, but the seed is 1.0).
    # If I can't find a correct rank-23, I shouldn't risk an incorrect one.
    
    # However, I can try to use a "Laderman" decomposition if I can verify it.
    # Since I can't verify it easily, I will stick to the schoolbook.
    
    # Wait, the current program IS the schoolbook.
    # So I am not improving.
    
    # Let's try to find a rank-23 decomposition.
    # I will use the one from the paper "A 23-Multiplication Algorithm for 3x3 Matrix Multiplication" by Laderman.
    # 
    # The 23 products are:
    # 1. (A00+A11+A22)(B00+B11+B22)
    # 2. (A01+A10+A22)(B01+B10+B22)
    # 3. (A02+A12+A20)(B02+B12+B20)
    # 4. (A00+A11)(B00+B11)
    # 5. (A00+A11)(B02+B12)
    # 6. (A00+A11)(B20+B21)
    # 7. (A01+A10)(B00+B11)
    # 8. (A01+A10)(B02+B12)
    # 9. (A01+A10)(B20+B21)
    # 10. (A02+A20)(B00+B22)
    # 11. (A02+A20)(B01+B21)
    # 12. (A02+A20)(B10+B12)
    # 13. (A12+A21)(B00+B22)
    # 14. (A12+A21)(B01+B21)
    # 15. (A12+A21)(B10+B12)
    # 16. A22*B22
    # 17. A12*B21
    # 18. A21*B12
    # 19. A02*B20
    # 20. A20*B02
    # 21. A10*B01
    # 22. A01*B10
    # 23. A00*B00
    
    # Let's check if this works.
    # C00 = A00B00 + A01B10 + A02B20
    # From M1: A00B00 + ...
    # From M4: A00B00 + ...
    # From M23: A00B00
    # This is getting messy.
    
    # I will stick to the schoolbook rank-27.
    # It is the safest bet.
    
    terms = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {k * n + i: 1}
                terms.append((u, v, w))
                
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] = val
        for (j, k), val in v.items():
            V[j * p + k][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
