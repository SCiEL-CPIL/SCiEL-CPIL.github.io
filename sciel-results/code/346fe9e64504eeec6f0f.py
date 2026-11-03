"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Try to find rank-16 decomposition via combinatorial search
    # Strategy: Use 7 Strassen-like terms for the 2x2x2 core (indices i,j,k in {0,1})
    # and solve for the remaining 9 terms (involving index 2) as rank-1 terms
    # Total: 7 + 9 = 16
    
    # The 2x2x2 core tensor has rank 7 (Strassen).
    # We need to verify that the "boundary" terms (those involving index 2)
    # can be expressed with rank 9, and that the total rank is 16.
    
    # Actually, let me think more carefully. The tensor M_{2,3,3} has entries
    # T[i,j,k] = delta[i,j] * delta[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # Wait, no. For matrix multiplication A(2x3) * B(3x3) = C(2x3):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 if this is a valid (i,j,k) triple, else 0.
    # Actually T[i,j,k] = delta_{i,j_valid} * delta_{j,k_valid}
    # More precisely: T[i,j,k] = 1 for all valid (i,j,k) since each entry 
    # contributes exactly once.
    
    # The tensor has 2*3*3 = 18 non-zero entries, each equal to 1.
    # We want to write T = sum_{r=1}^{rank} u_r * v_r * w_r (outer products)
    
    # Let's try rank 16. We'll use a constructive approach:
    # - 7 terms for the "core" 2x2x2 sub-tensor (j=0,1 and k=0,1, i=0,1)
    # - 9 terms for the remaining entries
    
    # Actually, let me just try to directly construct rank 16.
    
    # The 18 entries are:
    # (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # Entry value is 1 for all 18 triples.
    
    # Let me try a different approach: directly construct a rank-16 solution
    # by combining Strassen's 7 terms for the 2x2 block with 9 additional terms.
    
    # Strassen's 7 products for computing C = A*B where A is 2x2, B is 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6
    
    # For our 2x3 * 3x3 problem, the "core" uses j in {0,1}, k in {0,1}
    # The "boundary" uses j=2 or k=2.
    
    # Let's define the 16 rank-1 terms:
    # Terms 0-6: Strassen core (involving only A[i,0], A[i,1], B[0,k], B[1,k] for k in {0,1})
    # Terms 7-15: Boundary terms
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # (6, 16)
    V = [[0] * rank for _ in range(m * p)]  # (9, 16)
    W = [[0] * rank for _ in range(p * n)]  # (6, 16)
    
    def set_u(i, j, col, val):
        U[i * m + j][col] = val
        
    def set_v(j, k, col, val):
        V[j * p + k][col] = val
        
    def set_w(k, i, col, val):
        W[k * n + i][col] = val
    
    # Core: Strassen terms for the 2x2x2 sub-problem (j in {0,1}, k in {0,1})
    # These handle C[0,0], C[0,1], C[1,0], C[1,1] contributions from j=0,1
    
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    col = 0
    set_u(0, 0, col, 1)
    set_u(1, 1, col, 1)
    set_v(0, 0, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, 1)
    set_w(1, 1, col, 1)
    
    # M2 = (A10+A11)B00 -> contributes to C10 (+1) and C11 (-1)
    col = 1
    set_u(1, 0, col, 1)
    set_u(1, 1, col, 1)
    set_v(0, 0, col, 1)
    set_w(0, 1, col, 1)
    set_w(1, 1, col, -1)
    
    # M3 = A00(B01-B11) -> contributes to C01 (+1) and C11 (+1)
    col = 2
    set_u(0, 0, col, 1)
    set_v(0, 1, col, 1)
    set_v(1, 1, col, -1)
    set_w(1, 0, col, 1)
    set_w(1, 1, col, 1)
    
    # M4 = A11(B10-B00) -> contributes to C00 (+1) and C10 (+1)
    col = 3
    set_u(1, 1, col, 1)
    set_v(1, 0, col, 1)
    set_v(0, 0, col, -1)
    set_w(0, 0, col, 1)
    set_w(0, 1, col, 1)
    
    # M5 = (A00+A01)B11 -> contributes to C00 (-1) and C01 (+1)
    col = 4
    set_u(0, 0, col, 1)
    set_u(0, 1, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, -1)
    set_w(1, 0, col, 1)
    
    # M6 = (A10-A00)(B00+B01) -> contributes to C11 (+1)
    col = 5
    set_u(1, 0, col, 1)
    set_u(0, 0, col, -1)
    set_v(0, 0, col, 1)
    set_v(0, 1, col, 1)
    set_w(1, 1, col, 1)
    
    # M7 = (A01-A11)(B10+B11) -> contributes to C00 (+1)
    col = 6
    set_u(0, 1, col, 1)
    set_u(1, 1, col, -1)
    set_v(1, 0, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, 1)
    
    # Now boundary terms. We need to handle:
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    # C[0,0] += A02*B20
    # C[0,1] += A02*B21
    # C[1,0] += A12*B20
    # C[1,1] += A12*B21
    
    # That's 6 additional contributions to existing C entries, plus
    # 2 new C entries (C[0,2] and C[1,2]).
    # Total boundary: 8 distinct "output slots" but some share inputs.
    
    # Let me list all 18 (i,j,k) entries and which are handled by core:
    # Core handles: (0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,1)
    # That's 8 entries.
    # Boundary: (0,0,2), (0,1,2), (1,0,2), (1,1,2), (0,2,0), (0,2,1), (0,2,2), 
    #           (1,2,0), (1,2,1), (1,2,2)
    # That's 10 entries.
    
    # So we need 10 more rank-1 terms for the boundary, giving total 7+10=17.
    # To get rank 16, we need to merge one boundary term with a core term.
    
    # Alternative: use a different decomposition of the core that shares structure
    # with boundary terms.
    
    # Let me try: can we merge (0,2,0) with a core term?
    # (0,2,0) means A[0,2]*B[2,0] -> C[0,0]
    # This is a rank-1 term: u has 1 at (0,2), v has 1 at (2,0), w has 1 at (0,0).
    
    # Similarly (1,2,0): u at (1,2), v at (2,0), w at (0,1)
    # (0,2,1): u at (0,2), v at (2,1), w at (1,0)
    # (1,2,1): u at (1,2), v at (2,1), w at (1,1)
    # (0,2,2): u at (0,2), v at (2,2), w at (2,0)
    # (1,2,2): u at (1,2), v at (2,2), w at (2,1)
    # (0,0,2): u at (0,0), v at (0,2), w at (2,0)
    # (1,0,2): u at (1,0), v at (0,2), w at (2,1)
    # (0,1,2): u at (0,1), v at (1,2), w at (2,0)
    # (1,1,2): u at (1,1), v at (1,2), w at (2,1)
    
    # All 10 are distinct rank-1 terms. Total would be 17.
    
    # To achieve rank 16, we need to find a way to express one of these as a
    # linear combination of the others plus modified core terms.
    
    # Let me try a different approach: use 6 terms for the core instead of 7,
    # if possible, and 10 for boundary = 16. But 2x2x2 tensor has rank 7, so
    # we can't do the core in fewer than 7 terms.
    
    # Alternative: overlap some boundary terms with core terms by modifying
    # the core decomposition.
    
    # Let me try: instead of pure Strassen, use a modified core that also
    # partially handles boundary terms.
    
    # Actually, let me just verify: is rank 16 even achievable?
    # The tensor rank of M_{2,3,3} is known. Let me check if it's 16 or 17.
    
    # For now, let me try to construct rank 16 by being clever.
    
    # Idea: Merge (0,0,2) and (1,0,2) into one term?
    # (0,0,2): A00*B02 -> C02
    # (1,0,2): A10*B02 -> C12
    # These share B02 but have different A and C. Can't merge directly.
    
    # Idea: Merge (0,0,2) with something in the core?
    # (0,0,2) contributes to C[0,2]. Core terms don't touch C[0,2].
    
    # Let me try a completely different approach: use rank 16 with a 
    # non-standard decomposition.
    
    # Actually, let me just return the rank-17 solution if we can't find rank 16.
    # But first, let me try one more idea: use half-integer coefficients.
    
    # Try: Can we express the 10 boundary terms with 9 rank-1 terms using
    # half-integer coefficients?
    
    # The 10 boundary (i,j,k) triples:
    # B1=(0,0,2), B2=(1,0,2), B3=(0,1,2), B4=(1,1,2),
    # B5=(0,2,0), B6=(0,2,1), B7=(0,2,2), B8=(1,2,0), B9=(1,2,1), B10=(1,2,2)
    
    # Each is a distinct rank-1 tensor. The question is whether the sum of these 10
    # can be written as a sum of 9 rank-1 tensors (possibly with different structure).
    
    # This is equivalent to asking if the 10-dimensional subspace they span has
    # a rank-16 representation when combined with the 7 core terms.
    
    # Let me just try to verify rank 17 is correct and return it.
    
    # Actually, let me try one more constructive approach for rank 16.
    
    # What if we use a term like (A00+A10+A01+A11)*(B02+B12) and split it?
    # That would be:
    # (A00+A10+A01+A11)*(B02+B12) = A00*B02 + A00*B12 + A10*B02 + A10*B12 + 
    #                                A01*B02 + A01*B12 + A11*B02 + A11*B12
    # But we only need A00*B02, A10*B02, A01*B12, A11*B12 (for C02 and C12).
    # The extra terms A00*B12, A10*B12, A01*B02, A11*B02 would be spurious.
    
    # This doesn't work cleanly.
    
    # Let me just return the known rank-17 solution.
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, col, val):
        U[i * m + j][col] = val
        
    def set_v(j, k, col, val):
        V[j * p + k][col] = val
        
    def set_w(k, i, col, val):
        W[k * n + i][col] = val

    # Term 1: (a00 + a11) * (b00 + b11)
    col = 0
    set_u(0, 0, col, 1)
    set_u(1, 1, col, 1)
    set_v(0, 0, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, 1)
    set_w(1, 1, col, 1)

    # Term 2: (a10 + a11) * b00
    col = 1
    set_u(1, 0, col, 1)
    set_u(1, 1, col, 1)
    set_v(0, 0, col, 1)
    set_w(0, 1, col, 1)
    set_w(1, 1, col, -1)

    # Term 3: a00 * (b01 - b11)
    col = 2
    set_u(0, 0, col, 1)
    set_v(0, 1, col, 1)
    set_v(1, 1, col, -1)
    set_w(1, 0, col, 1)
    set_w(1, 1, col, 1)

    # Term 4: a11 * (b10 - b00)
    col = 3
    set_u(1, 1, col, 1)
    set_v(1, 0, col, 1)
    set_v(0, 0, col, -1)
    set_w(0, 0, col, 1)
    set_w(0, 1, col, 1)

    # Term 5: (a00 + a01) * b11
    col = 4
    set_u(0, 0, col, 1)
    set_u(0, 1, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, -1)
    set_w(1, 0, col, 1)

    # Term 6: (a10 - a00) * (b00 + b01)
    col = 5
    set_u(1, 0, col, 1)
    set_u(0, 0, col, -1)
    set_v(0, 0, col, 1)
    set_v(0, 1, col, 1)
    set_w(1, 1, col, 1)

    # Term 7: (a01 - a11) * (b10 + b11)
    col = 6
    set_u(0, 1, col, 1)
    set_u(1, 1, col, -1)
    set_v(1, 0, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, 1)

    # Term 8: a00 * b02
    col = 7
    set_u(0, 0, col, 1)
    set_v(0, 2, col, 1)
    set_w(2, 0, col, 1)

    # Term 9: a01 * b12
    col = 8
    set_u(0, 1, col, 1)
    set_v(1, 2, col, 1)
    set_w(2, 0, col, 1)

    # Term 10: a10 * b02
    col = 9
    set_u(1, 0, col, 1)
    set_v(0, 2, col, 1)
    set_w(2, 1, col, 1)

    # Term 11: a11 * b12
    col = 10
    set_u(1, 1, col, 1)
    set_v(1, 2, col, 1)
    set_w(2, 1, col, 1)

    # Term 12: a02 * b20
    col = 11
    set_u(0, 2, col, 1)
    set_v(2, 0, col, 1)
    set_w(0, 0, col, 1)

    # Term 13: a02 * b21
    col = 12
    set_u(0, 2, col, 1)
    set_v(2, 1, col, 1)
    set_w(1, 0, col, 1)

    # Term 14: a02 * b22
    col = 13
    set_u(0, 2, col, 1)
    set_v(2, 2, col, 1)
    set_w(2, 0, col, 1)

    # Term 15: a12 * b20
    col = 14
    set_u(1, 2, col, 1)
    set_v(2, 0, col, 1)
    set_w(0, 1, col, 1)

    # Term 16: a12 * b21
    col = 15
    set_u(1, 2, col, 1)
    set_v(2, 1, col, 1)
    set_w(1, 1, col, 1)

    # Term 17: a12 * b22
    col = 16
    set_u(1, 2, col, 1)
    set_v(2, 2, col, 1)
    set_w(2, 1, col, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
