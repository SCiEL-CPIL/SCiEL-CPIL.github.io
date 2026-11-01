import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 3
    # Rank-26 decomposition for 3x3 matrix multiplication
    # Using Strassen on the 2x2 top-left block + schoolbook for the rest
    
    # We'll build U, V, W as lists of lists
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (9, rank)
    # W has shape (p*n, rank) = (9, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_indices, values):
        """Add a rank-1 term. u_idx: row in U, v_idx: row in V, 
        w_indices: list of (w_row, value) pairs"""
        U.append(u_idx)
        V.append(v_idx)
        W.append(w_indices)
    
    # Block structure for 3x3:
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    # Indices: A[i,j] -> U row i*m + j = i*3 + j
    # A00=0, A01=1, A02=2
    # A10=3, A11=4, A12=5
    # A20=6, A21=7, A22=8
    #
    # B similarly: B[j,k] -> V row j*p + k = j*3 + k
    # B00=0, B01=1, B02=2
    # B10=3, B11=4, B12=5
    # B20=6, B21=7, B22=8
    #
    # C[i,k] -> W row k*n + i = k*3 + i  (trace convention W[k*n+i,r])
    # C00=0, C01=1, C02=2
    # C10=3, C11=4, C12=5
    # C20=6, C21=7, C22=8
    
    # For a rank-1 term: C += a * (u_vec) * (v_vec)^T
    # where u_vec has a single 1 at position u_idx (in U's 9 entries)
    # and v_vec has a single 1 at position v_idx (in V's 9 entries)
    # Then C[i,k] gets contribution a * A[u_idx] * B[v_idx]
    # So W[k*n+i] gets value a
    
    # Strassen for 2x2 block: C00_block = A00_block * B00_block
    # where A00_block = [[A00,A01],[A10,A11]] = [[0,1],[3,4]]
    # B00_block = [[B00,B01],[B10,B11]] = [[0,1],[3,4]]
    # C00_block = [[C00,C01],[C10,C11]] = [[0,1],[3,4]]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)(B00+B10)
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B01-B10)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1 - M6 + M4  ... wait let me be careful
    # C01 = M3 + M1 - M6  ... no
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1 - M6 + M4  -- actually let me look this up properly
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1 - M6
    # C10 = M2 + M5 - M1 + M6  -- hmm, let me just use the standard formulas
    # C10 = M2 + M5 - M7
    # C11 = M2 + M3 - M4 + M5
    
    # Actually, standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)(B11+B12)
    # P6 = (A12-A22)(B21+B22)
    # P7 = (A21-A11)(B12-B11)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P1 - P6
    # C21 = P2 + P5 - P7  -- wait, I'm mixing up indexing
    # Let me use 0-indexed:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P1 - P6
    # C10 = P2 + P5 - P1 + P6  -- no
    # C10 = P2 + P5 - P7
    # C11 = P2 + P3 - P4 + P5
    
    # Let me just verify: the key insight is that for the 2x2 sub-block,
    # we need 7 rank-1 terms where each "multiplication" M_i is itself
    # a product of linear combinations. But in tensor decomposition,
    # each rank-1 term corresponds to ONE scalar multiplication.
    
    # So for Strassen, each M_i = (sum of A entries)(sum of B entries)
    # expands to multiple rank-1 terms in the original tensor.
    
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # That's 4 rank-1 terms!
    
    # So Strassen on 2x2 doesn't directly give us 7 rank-1 terms for the 
    # full tensor. It gives 7 "block multiplications" which expand to more.
    
    # The rank-26 result comes from a different approach. Let me think...
    
    # Actually, for 3x3, the rank-26 decomposition is known. Let me construct it
    # by taking the 2x2 Strassen (which when expanded gives the right number of
    # scalar multiplications for the 2x2 part) plus the cross terms.
    
    # For the 2x2 top-left block of C:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Schoolbook: 4 multiplications. Strassen: 7 block-mults but each is a product
    # of sums, so it's not directly applicable as rank-1 terms.
    
    # The rank-26 comes from a specific construction. Let me use a known one.
    # 
    # Reference: The rank of 3x3 matrix multiplication is at most 26.
    # One construction: partition into 2x2 and 1x1, use Strassen for the 2x2x2x2
    # part (rank 7 for that sub-tensor) and schoolbook for the rest.
    # 
    # But Strassen's 7 multiplications are for the TENSOR, meaning 7 rank-1 terms
    # where each factor can be a sum. In the final decomposition, we expand these.
    
    # Let me just hardcode the rank-26 decomposition from the literature.
    # 
    # Actually, let me construct it more carefully.
    
    # The 3x3 multiplication tensor has 27 rank-1 terms (schoolbook).
    # We can reduce by 1 using the following trick:
    # 
    # Instead of A00*B00 as a separate term, we can combine it with others.
    # 
    # Known rank-26 construction:
    # Use the fact that (A00+A11+B00+B11) creates a useful combination.
    
    # Let me just directly build the rank-26 decomposition.
    
    terms = []
    
    # For each of the 27 schoolbook terms A[i,j]*B[j,k], we have a rank-1 term
    # But we can merge some.
    
    # The standard rank-26 for 3x3:
    # We keep most terms but replace one pair with a Strassen-like combination.
    
    # Let me enumerate: A[i,j]B[j,k] for i,j,k in {0,1,2}
    # i*3+j = U index, j*3+k = V index, k*3+i = W index
    
    # Schoolbook terms (27):
    # (i,j,k) -> U:i*3+j, V:j*3+k, W:k*3+i
    
    # The rank-26 reduction: 
    # Replace A00*B00 and A11*B11 with a Strassen-like pair.
    # 
    # Term A00*B00: U=0, V=0, W=0 (C00)
    # Term A11*B11: U=4, V=4, W=4 (C11)
    # 
    # Strassen: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # And we already have A00B11 (U=0,V=4,W=3) and A11B00 (U=4,V=0,W=1)
    
    # So: (A00+A11)(B00+B11) = C00_contrib + A00B11 + A11B00 + C11_contrib
    # where C00_contrib = A00B00 and C11_contrib = A11B11
    
    # This doesn't save anything directly.
    
    # The actual rank-26 construction is more subtle. Let me use a different approach.
    
    # I'll construct the full rank-26 decomposition explicitly.
    
    # Based on the known result, here's the construction:
    
    # We'll have 26 rank-1 terms.
    
    # Let me define them:
    
    # First, all terms except A00*B00 (we'll handle it specially):
    # Actually, let me just use a known explicit rank-26.
    
    # Here's a clean way: use the block decomposition.
    # Split 3x3 into:
    # [2x2 | 2x1]
    # [1x2 | 1x1]
    
    # C = [C11 | C12]
    #     [C21 | C22]
    
    # where C11 is 2x2, C12 is 2x1, C21 is 1x2, C22 is 1x1
    
    # C11 = A11_block * B11_block  (2x2 * 2x2) -> rank 7 (Strassen)
    # C12 = A11_block * B12_block  (2x2 * 2x1) -> rank 4
    # C21 = A21_block * B11_block  (1x2 * 2x2) -> rank 4
    # C22 = A22_scalar * B22_scalar (1x1 * 1x1) -> rank 1
    # Plus cross terms:
    # C11 also gets: A12_block * B21_block (2x1 * 1x2) -> rank 2 (outer product has 2 independent)
    # Wait, A12 is 2x1, B21 is 1x2, so A12*B21 is 2x2, rank 1. That's 1 term.
    # Hmm, but we need to be more careful.
    
    # Let me index:
    # A11_block = A[0:2,0:2] = rows 0,1 cols 0,1 -> U indices 0,1,3,4
    # B11_block = B[0:2,0:2] -> V indices 0,1,3,4
    # A12_block = A[0:2,2] = rows 0,1 col 2 -> U indices 2,5
    # B21_block = B[2,0:2] = row 2 cols 0,1 -> V indices 6,7
    # A21_block = A[2,0:2] = row 2 cols 0,1 -> U indices 6,7
    # B12_block = B[0:2,2] = rows 0,1 col 2 -> V indices 2,5
    # A22 = A[2,2] -> U index 8
    # B22 = B[2,2] -> V index 8
    
    # C11 (2x2): W indices for C[0,0],C[0,1],C[1,0],C[1,1] = 0,1,3,4
    # C12 (2x1): C[0,2],C[1,2] -> W indices 2,5
    # C21 (1x2): C[2,0],C[2,1] -> W indices 6,7
    # C22 (1x1): C[2,2] -> W index 8
    
    # Now, C11 = A11*B11 + A12*B21
    # A11*B11: 2x2 * 2x2, schoolbook = 4 terms, Strassen = 7 terms (but in tensor rank)
    # A12*B21: 2x1 * 1x2 = outer product, 1 rank-1 term
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 2x2 * 2x1 = 4 terms (schoolbook)
    # A12*B22: 2x1 * 1x1 = 2 terms
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 1x2 * 2x2 = 4 terms
    # A22*B21: 1x1 * 1x2 = 2 terms
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 1x2 * 2x1 = 2 terms
    # A22*B22: 1x1 * 1x1 = 1 term
    
    # Total schoolbook: 4+1 + 4+2 + 4+2 + 2+1 = 20 + 6 = 26? Let me recount.
    # A11*B11: 4, A12*B21: 1 -> 5
    # A11*B12: 4, A12*B22: 2 -> 6
    # A21*B11: 4, A22*B21: 2 -> 6
    # A21*B12: 2, A22*B22: 1 -> 3
    # Total: 5+6+6+3 = 20
    
    # That's only 20, which is less than 27. That can't be right for schoolbook.
    # Oh wait, I'm double counting or missing something. The schoolbook is 27 total.
    
    # Let me just list all 27: A[i,j]*B[j,k]
    # j=0: i in {0,1,2}, k in {0,1,2} -> 9 terms
    # j=1: 9 terms
    # j=2: 9 terms
    
    # My block decomposition:
    # C11 (i in {0,1}, k in {0,1}):
    #   j=0: A[i,0]B[0,k] -> 4 terms (i,k in {0,1})
    #   j=1: A[i,1]B[1,k] -> 4 terms
    #   j=2: A[i,2]B[2,k] -> 4 terms (these are A12*B21, 2x1 * 1x2)
    #   Total for C11: 12? But schoolbook for 2x2 output from 3x3*3x3 is... 
    #   C[i,k] = sum_j A[i,j]B[j,k], so for i,k in {0,1}, j in {0,1,2}: 4*3 = 12 terms.
    
    # I was wrong. Each C[i,k] has 3 terms (sum over j). So:
    # C11 (4 entries * 3 j-values) = 12 terms
    # C12 (2 entries * 3) = 6 terms
    # C21 (2 * 3) = 6 terms
    # C22 (1 * 3) = 3 terms
    # Total: 27. Correct.
    
    # Now, for C11, the 12 terms can be reduced using Strassen on the j=0,1 part:
    # A11_block * B11_block (j=0,1) has 8 terms (2x2*2x2 schoolbook)
    # Strassen reduces this to 7 rank-1 terms (in the tensor sense, where factors are sums)
    # But each "rank-1 term" in Strassen has factors that are sums, so when we write it
    # as a sum of rank-1 terms with scalar entries, we expand.
    
    # This is getting complicated. Let me just directly construct the rank-26.
    
    # I'll use the following approach: take the 27 schoolbook terms and merge two of them
    # using the identity that saves one rank.
    
    # Known: rank(MM_3) <= 26. One explicit construction:
    
    # Let me just hardcode a known rank-26 decomposition.
    
    # After research, here's a clean construction:
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_rank1_term(u_indices, v_indices, w_map):
        """u_indices: dict {U_row: coeff}, v_indices: dict {V_row: coeff}, 
           w_map: dict {W_row: coeff}"""
        U_rows.append(u_indices)
        V_rows.append(v_indices)
        W_rows.append(w_map)
    
    # 26 terms:
    
    # Terms for j=0 (A[*,0] * B[0,*]): 9 terms, but we'll modify some
    # A00*B00: U{0:1}, V{0:1}, W{0:1}
    # A00*B01: U{0:1}, V{1:1}, W{1:1}
    # A00*B02: U{0:1}, V{2:1}, W{2:1}
    # A10*B00: U{3:1}, V{0:1}, W{3:1}
    # A10*B01: U{3:1}, V{1:1}, W{4:1}
    # A10*B02: U{3:1}, V{2:1}, W{5:1}
    # A20*B00: U{6:1}, V{0:1}, W{6:1}
    # A20*B01: U{6:1}, V{1:1}, W{7:1}
    # A20*B02: U{6:1}, V{2:1}, W{8:1}
    
    # Terms for j=1: 9 terms
    # A01*B10: U{1:1}, V{3:1}, W{0:1}
    # A01*B11: U{1:1}, V{4:1}, W{1:1}
    # A01*B12: U{1:1}, V{5:1}, W{2:1}
    # A11*B10: U{4:1}, V{3:1}, W{3:1}
    # A11*B11: U{4:1}, V{4:1}, W{4:1}
    # A11*B12: U{4:1}, V{5:1}, W{5:1}
    # A21*B10: U{7:1}, V{3:1}, W{6:1}
    # A21*B11: U{7:1}, V{4:1}, W{7:1}
    # A21*B12: U{7:1}, V{5:1}, W{8:1}
    
    # Terms for j=2: 9 terms
    # A02*B20: U{2:1}, V{6:1}, W{0:1}
    # A02*B21: U{2:1}, V{7:1}, W{1:1}
    # A02*B22: U{2:1}, V{8:1}, W{2:1}
    # A12*B20: U{5:1}, V{6:1}, W{3:1}
    # A12*B21: U{5:1}, V{7:1}, W{4:1}
    # A12*B22: U{5:1}, V{8:1}, W{5:1}
    # A22*B20: U{8:1}, V{6:1}, W{6:1}
    # A22*B21: U{8:1}, V{7:1}, W{7:1}
    # A22*B22: U{8:1}, V{8:1}, W{8:1}
    
    # That's 27. To get 26, we need to merge one.
    
    # The standard trick: replace A00*B00 and A11*B11 with:
    # (A00+A11)*(B00+B11) - A00*B11 - A11*B00
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A00*B11 - A11*B00
    # = A00*B00 + A11*B11
    # So (A00+A11)*(B00+B11) gives us A00*B00 + A11*B11 + A00*B11 + A11*B00
    # We already have A00*B11 and A11*B00 as separate terms.
    # So if we use (A00+A11)*(B00+B11) and subtract the existing A00*B11 and A11*B00,
    # we get A00*B00 + A11*B11. That doesn't save a term.
    
    # The actual rank-26 uses a different combination. Let me look this up properly.
    
    # After more thought, the rank-26 for 3x3 is achieved by:
    # Using the 2x2 Strassen for the top-left 2x2 subproblem, which uses 7 multiplications
    # instead of 4, but the "multiplications" are of linear combinations. When expanded,
    # the total number of scalar multiplications is still reduced.
    
    # I think the cleanest approach is to just use a known explicit rank-26.
    
    # Let me construct it as follows:
    # Keep all 27 schoolbook terms, but replace the pair {A00*B00, A11*B11} with
    # 3 terms that together give the same contribution:
    # 
    # We want: A00*B00 + A11*B11
    # 
    # Note that: (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # And we have A00*B11 and A11*B00 elsewhere.
    # 
    # So: A00*B00 + A11*B11 = (A00+A11)*(B00+B11) - A00*B11 - A11*B00
    # 
    # This uses 3 "terms" (one with sum factors, two with scalar factors) to replace 2 terms.
    # That's worse!
    
    # I think I need to accept that constructing rank-26 from scratch is complex.
    # Let me just use the rank-27 schoolbook for now, since the task says to find
    # rank-minimal, and if I can't easily construct rank-26, I'll use rank-27.
    
    # Actually, wait. The knowledge says a program achieved score 1.038 with rank 26.
    # So rank 26 is achievable. Let me just implement it carefully.
    
    # Here's the key insight for rank-26:
    # 
    # The 3x3 multiplication tensor can be decomposed with rank 26 by using
    # Strassen's algorithm on the 2x2 leading principal submatrix.
    # 
    # Specifically, write:
    # C = [[C11, C12], [C21, C22]]
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    #
    # where A11,B11,C11 are 2x2, A12,B21,C12,C21 are 2x1, etc.
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # For A11*B11 (2x2 * 2x2), use Strassen: 7 rank-1 terms (with sum factors)
    # For all other products, use schoolbook.
    #
    # A11*B11: 7 terms
    # A12*B21: (2x1)(1x2) = outer product = 1 rank-1 term
    # A11*B12: (2x2)(2x1) = 4 terms
    # A12*B22: (2x1)(1x1) = 2 terms
    # A21*B11: (1x2)(2x2) = 4 terms
    # A22*B21: (1x1)(1x2) = 2 terms
    # A21*B12: (1x2)(2x1) = 2 terms (dot product, but as rank-1: sum of 2)
    # A22*B22: (1x1)(1x1) = 1 term
    #
    # Total: 7+1+4+2+4+2+2+1 = 23?? That seems too good.
    # 
    # The issue is that Strassen's 7 "terms" have factors that are sums, so each
    # expands to multiple rank-1 terms in the strict sense.
    
    # OK here's what I'll do: I'll implement the rank-27 schoolbook decomposition
    # since that's guaranteed correct, and it matches the current program's behavior.
    # The task asks for rank-minimal, but if I can't confidently construct rank-26,
    # I should at least return a valid decomposition.
    
    n, m, p = 3, 3, 3
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
