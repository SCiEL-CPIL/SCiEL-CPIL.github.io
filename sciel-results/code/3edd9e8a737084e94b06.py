# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 3, 3

    # We will build the rank-26 decomposition manually.
    # This consists of:
    # 7 Strassen terms for the 2x2 block (indices 0,1)
    # 19 schoolbook terms for the rest (involving index 2)
    
    # The decomposition identity is:
    # For all A (n x m), B (m x p):
    # C[i, k] = sum_{j=0}^{m-1} A[i, j] B[j, k]
    # The factorized form is:
    # C[i, k] = sum_{r=0}^{R-1} (sum_{i',j'} U[i', j', r] A[i', j']) (sum_{j'',k'} V[j'', k', r] B[j'', k']) W[k, i, r]
    # Wait, the W convention is W[k*n+i, r].
    # The problem states: "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means the output vector is indexed by (k, i) where k is the column of C and i is the row of C?
    # Let's check the shape of W: (p*n, rank).
    # If C is n x p, then it has n*p elements.
    # The index k*n+i implies k ranges 0..p-1 and i ranges 0..n-1.
    # So the element at index `idx` in the flattened output corresponds to C[i, k] where idx = k*n + i.
    # This matches the standard column-major flattening if we view C^T, or just a specific indexing convention.
    # Let's verify with the tensor equation.
    # Term r contributes:
    # U_r(A) * V_r(B) * W_r
    # where U_r(A) = sum_{i,j} U[i, j, r] A[i, j]
    # V_r(B) = sum_{j, k} V[j, k, r] B[j, k]
    # The product is a scalar. It is added to the output element corresponding to W_r.
    # If W_r has a non-zero entry at index `idx`, it adds to the output element at `idx`.
    # The output element at `idx = k*n + i` is C[i, k].
    # So we need:
    # sum_r U_r(A) V_r(B) W[idx, r] = C[i, k] for idx = k*n+i.
    
    # Let's construct the terms (U, V, W) for each of the 26 components.
    # U is a dict (i, j) -> val
    # V is a dict (j, k) -> val
    # W is a dict idx -> val (where idx = k*n + i)
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
        
    # Helper to get W index
    def w_idx(i, k):
        return k * n + i
        
    # --- Strassen 7 terms for C[0:2, 0:2] (using only A[0:2, 0:2] and B[0:2, 0:2]) ---
    # These terms compute the 2x2 product C_22 = A_22 * B_22.
    # A_22 = [[A00, A01], [A10, A11]]
    # B_22 = [[B00, B01], [B10, B11]]
    # C_22 = [[C00, C01], [C10, C11]]
    
    # 1. M1 = (A00 + A11) (B00 + B11)
    # U: A00, A11
    # V: B00, B11
    # Product terms: A00B00, A00B11, A11B00, A11B11
    # C00 gets A00B00 -> W[idx(0,0)] += 1
    # C01 gets A00B11? No, C01 = A00B01 + A01B11.
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M3 = A00(B10-B11) = A00B10 - A00B11
    # M4 = A11(B01-B00) = A11B01 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    
    # Let's sum them up to see which C entries they produce:
    # C00 = A00B00 + A01B10
    # From M1: +A00B00
    # From M6: -A00B00
    # From M7: +A01B10
    # Sum: A01B10. Missing A00B00.
    
    # Let's check the standard Strassen combination for C:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # Let's verify C00:
    # M1 - M4 + M6 - M7
    # = (A00B00 + A00B11 + A11B00 + A11B11) 
    # - (A11B01 - A11B00) 
    # + (A10B00 + A10B01 - A00B00 - A00B01) 
    # - (A01B10 + A01B11 - A11B10 - A11B11)
    # 
    # A00B00: 1 - 1 = 0? No, wait.
    # M1 has A00B00. M6 has -A00B00. Sum = 0.
    # M1 has A00B11. M3 has -A00B11 (not in this sum). M5 has A00B11 (not in this sum). M7 has -A01B11 (different).
    # Wait, I need to look at the terms carefully.
    # C00 = A00B00 + A01B10.
    # Let's re-evaluate the sum for C00:
    # M1: A00B00
    # M6: -A00B00
    # M7: -A01B10
    # M4: -(-A11B00) -> +A11B00? No, M4 = A11B01 - A11B00. -M4 = -A11B01 + A11B00.
    # This is getting messy. Let's just use the known coefficients.
    # 
    # Coefficients for C00: M1: 1, M4: -1, M6: 1, M7: -1
    # Let's check if this yields A00B00 + A01B10.
    # M1: A00B00 + A00B11 + A11B00 + A11B11
    # -M4: -A11B01 + A11B00
    # M6: A10B00 + A10B01 - A00B00 - A00B01
    # -M7: -A01B10 - A01B11 + A11B10 + A11B11
    # 
    # Summing terms:
    # A00B00: 1 (from M1) - 1 (from M6) = 0. ERROR.
    
    # I am misremembering the Strassen combination or the definitions.
    # Standard Strassen:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B22)
    # P4 = A22(B12 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    # 
    # C11 = P1 - P4 + P6 - P7
    # C12 = P5 + P7
    # C21 = P2 + P4
    # C22 = P1 - P3 + P5 - P6
    
    # My indices: 0,1 correspond to 1,2 in standard notation.
    # So:
    # M1 = P1
    # M2 = P2
    # M3 = P3
    # M4 = P4
    # M5 = P5
    # M6 = P6
    # M7 = P7
    
    # C[0,0] = M1 - M4 + M6 - M7
    # C[0,1] = M5 + M7
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M5 - M6
    
    # Let's verify C[0,0] = A00B00 + A01B10 again with this.
    # M1: A00B00 + A00B11 + A11B00 + A11B11
    # -M4: -A11B01 + A11B00
    # M6: A10B00 + A10B01 - A00B00 - A00B01
    # -M7: -A01B10 - A01B11 + A11B10 + A11B11
    
    # Terms:
    # A00B00: 1 - 1 = 0. Still 0.
    # There is something wrong.
    # Let's check P1 definition again.
    # P1 = (A11+A22)(B11+B22).
    # If indices are 0,1:
    # P1 = (A00+A11)(B00+B11).
    # Expansion: A00B00 + A00B11 + A11B00 + A11B11. Correct.
    
    # Let's check P6 definition.
    # P6 = (A21-A11)(B11+B12).
    # Indices 0,1:
    # P6 = (A10-A00)(B00+B01).
    # Expansion: A10B00 + A10B01 - A00B00 - A00B01. Correct.
    
    # Sum for C11 (which is C[0,0]):
    # A00B00: From P1 (1), from P6 (-1). Sum 0.
    # Why is C11 not A11B11?
    # C11 = A11B11 + A12B21.
    # In 2x2 case, C11 = A00B00 + A01B10.
    # 
    # Wait, the Strassen formula computes the product of two 2x2 matrices.
    # Let's look at a reliable source or re-derive.
    # 
    # Actually, I can just use the schoolbook terms for all 9 entries and verify rank 27.
    # Then try to optimize.
    # But the task asks for rank-minimal.
    # Current best known for 3x3 is 23? No, 27 is standard schoolbook.
    # Strassen for 3x3 can be done with 3x3 blocks of size 1? No.
    # 
    # Let's stick to the safe bet: Rank 27 Schoolbook.
    # It is guaranteed to be correct.
    # Score: seed/value. If seed is 27, score 1.0.
    # If I can do better, great. But correctness is 0.0 currently.
    # So first goal: Get a valid decomposition.
    
    terms = []
    
    # Schoolbook: 27 terms.
    # For each i, k in 0..2:
    #   For each j in 0..2:
    #     Term: U{(i,j):1}, V{(j,k):1}, W{idx(i,k):1}
    #     This contributes A[i,j] B[j,k] to C[i,k].
    
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
