import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _rank23_3x3x3():
    """Return rank-23 decomposition of 3x3x3 matrix multiplication.
    Returns list of (u_idx, v_idx, w_idx) triples where:
    u_idx is index into A (row*3+col), v_idx into B (row*3+col), w_idx into C (row*3+col)
    """
    # Strassen for 2x2x2 embedded in top-left, plus boundary terms
    # A indices: (i,j) -> i*3+j, i,j in {0,1,2}
    # B indices: (j,k) -> j*3+k
    # C indices: (i,k) -> i*3+k
    
    terms = []
    
    # Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # We need to express each Strassen multiplier as a linear combination
    # This requires 7 additional "virtual" multipliers, but in the tensor
    # decomposition framework, each term is a rank-1 term.
    # 
    # Actually, for the tensor decomposition, we need to express the 3x3x3
    # tensor as a sum of rank-1 tensors. The Strassen algorithm gives us
    # 7 multiplications for the 2x2 part, but each "multiplication" in
    # Strassen is itself a sum of products, so we need to expand.
    #
    # Let me use a different approach: directly enumerate a known rank-23.
    
    # Standard rank-23 decomposition for 3x3x3:
    # Based on the construction in "On the complexity of multiplying 3x3 matrices"
    
    # I'll use the following explicit decomposition with 23 terms.
    # Each term (a, b, c) means: add a_i * b_j * c_k to C[i,k] for all i,j,k
    # where a_i is the coefficient for A[i,j], b_j for B[j,k], c_k for C[i,k]
    
    # Actually, let me use a verified construction.
    # The 23 terms are:
    
    # Terms 1-7: Strassen-like for the 2x2 submatrix
    # Term 1: (A00+A11)*(B00+B11) contributes to C00,C01,C10,C11
    # This is not a rank-1 term in the tensor sense.
    
    # Let me use a different, explicit rank-23 decomposition.
    # From the literature, one such decomposition uses these 23 rank-1 terms:
    
    # I'll construct it by hand using the "block" approach.
    
    # Approach: Use 7 terms for 2x2x2 Strassen (top-left), 
    # then handle the remaining 12 "cross" terms and 4 corner terms.
    
    # Actually, the simplest correct approach: 
    # 3x3x3 can be done with 23 mults using the following scheme:
    
    # Let me just use schoolbook for now (27) and see if I can do better.
    # For 3x3x6: 2 * 27 = 54 (same as schoolbook)
    # With Strassen 2x2: we can save some.
    
    # Let me implement a proper rank-23 for 3x3x3.
    # The key insight: partition 3x3 into 2x2 + 1x1 blocks.
    
    # C = A*B
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2]
    # C[0:2,2] = A[0:2,0:2]*B[0:2,2] + A[0:2,2]*B[2,2]
    # C[2,0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2]*B[2,0:2]
    # C[2,2] = A[2,0:2]*B[0:2,2] + A[2,2]*B[2,2]
    
    # C[0:2,0:2]: 2x2*2x2 (7 Strassen) + 2x1*1x2 (2 schoolbook) = 9
    # C[0:2,2]: 2x2*2x1 (4) + 2x1*1x1 (2) = 6
    # C[2,0:2]: 1x2*2x2 (4) + 1x1*1x2 (2) = 6
    # C[2,2]: 1x2*2x1 (2) + 1x1*1x1 (1) = 3
    # Total: 9+6+6+3 = 24. Close to 23 but not quite.
    
    # With some sharing of Strassen intermediates, we can get to 23.
    # But implementing this correctly is complex.
    
    # For now, let me use a safe rank-24 decomposition and aim for 48 total.
    # Actually, let me just verify: can I get below 54?
    
    # Simple approach: use Strassen 2x2 for the top-left 2x2 of each 3x3:
    # For each 3x3x3: 
    #   C[0:2,0:2] via Strassen: 7 terms
    #   C[0:2,2]: A[0:2,:] * B[:,2] = 2x3 * 3x1 = 6 terms (schoolbook)
    #   C[2,0:2]: A[2,:] * B[:,0:2] = 1x3 * 3x2 = 6 terms
    #   C[2,2]: A[2,2]*B[2,2] = 1 term
    #   But C[0:2,0:2] also gets contributions from A[0:2,2]*B[2,0:2]
    #   C[0:2,0:2] += A[0:2,2]*B[2,0:2] = 2x1 * 1x2 = 2 terms
    #   C[0:2,2] += A[0:2,2]*B[2,2] = 2x1*1x1 = 2 terms (already counted)
    #   C[2,0:2] += A[2,2]*B[2,0:2] = 1x1*1x2 = 2 terms (already counted)
    #   C[2,2] += A[2,0:2]*B[0:2,2] = 1x2*2x1 = 2 terms
    #   
    #   Total: 7 + 6 + 6 + 1 + 2 + 2 = 24
    
    # So rank 24 per 3x3x3, total 48 for 3x3x6.
    
    # Let me implement rank-24 for 3x3x3.
    
    terms = []
    
    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Using 7 rank-1 terms (this is the standard Strassen decomposition)
    # But Strassen's 7 mults are not rank-1 in the tensor sense!
    # Each Strassen mult is a product of sums, which expands to multiple rank-1 terms.
    #
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This is 4 rank-1 terms, not 1!
    #
    # So Strassen doesn't directly give a tensor decomposition with rank 7.
    # The rank-7 is for the 2x2x2 tensor, meaning it CAN be decomposed into 7
    # rank-1 tensors, but the explicit form is non-trivial.
    #
    # The 7 rank-1 terms for 2x2x2 Strassen:
    # (using the standard form)
    # t1: (1,0; 0,1) x (1,0; 0,1) x (1,0; 0,1) -- no, this isn't right either.
    
    # Let me look up the actual rank-1 decomposition for 2x2x2.
    # The 7 terms are:
    # (U1, V1, W1) where each is a 2x2 matrix
    # U1 = [[1,0],[0,1]], V1 = [[1,0],[0,1]], W1 = [[1,0],[0,1]] -- no
    
    # The correct Strassen rank-1 decomposition of the 2x2x2 matmul tensor:
    # The tensor T has T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # For 2x2x2: T is a 4x4x4 tensor (or 2x2x2x2x2x2)
    
    # The 7 rank-1 terms (as triples of 2x2 matrices):
    # T1 = [[1,0],[0,0]] x [[1,0],[0,0]] x [[1,0],[0,0]]  (A00*B00 -> C00)
    # T2 = [[0,0],[0,1]] x [[0,0],[0,1]] x [[0,0],[0,1]]  (A11*B11 -> C11)
    # T3 = [[0,1],[0,0]] x [[1,0],[0,0]] x [[1,0],[0,0]]  (A01*B00 -> C00)
    # T4 = [[1,0],[0,0]] x [[0,1],[0,0]] x [[0,1],[0,0]]  (A00*B01 -> C01)
    # T5 = [[0,0],[1,0]] x [[1,0],[0,0]] x [[1,0],[0,0]]  (A10*B00 -> C10)
    # T6 = [[0,1],[0,0]] x [[0,0],[0,1]] x [[0,1],[0,0]]  (A01*B11 -> C11)
    # T7 = [[0,0],[1,0]] x [[0,1],[0,0]] x [[0,0],[1,0]]  (A10*B01 -> C10)
    # That's only 7 for the diagonal/cross terms but misses C01 and C10 fully.
    
    # I'm overcomplicating this. Let me just use schoolbook and accept rank 54,
    # or find a simpler improvement.
    
    # Actually, the simplest improvement: use the fact that we're multiplying
    # 3x3 by 3x6. We can split into two 3x3x3 problems.
    # Each 3x3x3 has rank 27 (schoolbook) or 23 (best known).
    # With rank 23: total 46.
    
    # Let me implement a verified rank-23 decomposition.
    # After research, here's a valid one:
    
    # The 23 terms for 3x3x3 matmul (indices 0-2 for rows/cols):
    # I'll use the construction from "A 23-multiplication algorithm for 3x3 matrices"
    
    # For now, let me just return schoolbook (rank 54) to ensure correctness,
    # and note that a rank-46 solution exists but requires careful implementation.
    
    # Actually, let me try a simple optimization: use 2x2 Strassen for the
    # top-left 2x2 block of each 3x3x3, and schoolbook for the rest.
    # This gives rank 7 + 20 = 27 per 3x3x3, no improvement.
    
    # The issue is that Strassen's 7 mults don't directly translate to rank-1
    # terms in the tensor decomposition.
    
    # Let me just return the schoolbook solution.
    return None  # placeholder

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition
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
# EVOLVE-BLOCK-END
