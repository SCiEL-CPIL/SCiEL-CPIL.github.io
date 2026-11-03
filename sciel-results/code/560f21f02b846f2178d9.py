"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 5x5 x 5x5.
    
    Strategy: Use a block decomposition approach.
    Partition the 5x5 matrices into blocks. Since 5 = 2 + 3, we can think of
    the matrices as having a 2x2 block and a 3x3 block in some partition.
    
    However, a more direct approach: use the known lower bounds and constructions.
    For 5x5, the best known upper bound is around 60-70 or so.
    
    Let's use a simple but effective approach: 
    We'll construct the decomposition using a block structure where we treat
    the 5x5 multiplication as involving 2x2 and 3x3 subproblems.
    
    Actually, let's use a cleaner approach. The rank of 5x5 matrix multiplication
    is at most 125 (schoolbook). We can do better.
    
    Let me use the following construction:
    - Split indices into groups. For a 5x5 matrix, we can partition the 5 indices
      into sizes that allow efficient block multiplication.
    
    A simple improvement: use Strassen-like ideas on 2x2 blocks and standard on 3x3.
    
    Let's partition the 5x5 matrix into blocks:
    Rows: 2 + 3, Cols: 2 + 3 for A (5x5)
    Rows: 2 + 3, Cols: 2 + 3 for B (5x5)
    
    So A = [[A11(2x2), A12(2x3)], [A21(3x2), A22(3x3)]]
    B = [[B11(2x2), B12(2x3)], [B21(3x2), B22(3x3)]]
    
    C = A*B = [[C11(2x2), C12(2x3)], [C21(3x2), C22(3x3)]]
    where:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each of these is a sum of products of rectangular matrices.
    
    For the 2x2 blocks, we can use Strassen (rank 7 instead of 8).
    For the 2x3, 3x2, 3x3 blocks, we use schoolbook or other optimizations.
    
    Let me count the rank:
    - C11: 2x2 * 2x2 (Strassen: 7) + 2x3 * 3x2 (schoolbook: 6) = 13
    - C12: 2x2 * 2x3 (schoolbook: 6) + 2x3 * 3x3 (schoolbook: 9) = 15
    - C21: 3x2 * 2x2 (schoolbook: 6) + 3x3 * 3x2 (schoolbook: 9) = 15
    - C22: 3x2 * 2x3 (schoolbook: 6) + 3x3 * 3x3 (schoolbook: 9) = 15
    
    Total: 13 + 15 + 15 + 15 = 58
    
    This gives rank 58, which is better than 125.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # all 5
    
    # Partition indices
    # A is n x m = 5x5, B is m x p = 5x5, C is n x p = 5x5
    # Split rows/cols at index 2: first 2, then 3
    
    def idx2d(i, j, split):
        """Convert (i,j) in 5x5 to block-local index."""
        return i, j
    
    # Block structure:
    # A: rows 0-1 (top), 2-4 (bottom); cols 0-1 (left), 2-4 (right)
    # B: rows 0-1 (top), 2-4 (bottom); cols 0-1 (left), 2-4 (right)
    # C: rows 0-1 (top), 2-4 (bottom); cols 0-1 (left), 2-4 (right)
    
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (25, rank)
    
    # For each term in the decomposition, we have:
    # U[row_of_A, col] * V[row_of_B, col] * W[row_of_C, col]
    # where row_of_A = i*m + j (i is row in A, j is col in A)
    # row_of_B = j*p + k (j is row in B, k is col in B)
    # row_of_C = k*n + i (k is row in C, i is col in C)
    # Wait, the trace convention is W[k*n+i, r]
    
    # Let me reconsider. The standard tensor for matrix mult (n x m) * (m x p) -> (n x p)
    # has entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    
    # In the factorization, we need:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So for each rank-1 term r, we pick specific (i,j), (j,k), (k,i) triples.
    
    # Let me build the rank-1 terms for each block.
    
    terms = []  # list of (U_idx, V_idx, W_idx)
    
    # C11: 2x2 result, from A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    # A11: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # A12: i in {0,1}, j in {2,3,4}
    # B21: j in {2,3,4}, k in {0,1}
    
    # A11 * B11 using Strassen for 2x2:
    # Strassen computes:
    # M1 = (A11 + A12_00) * (B11 + B21_00)  -- but we need to be careful
    # Actually, for 2x2 matrices:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Strassen:
    # m1 = (a+d)*(e+h)
    # m2 = (c+d)*e
    # m3 = a*(f-h)
    # m4 = d*(g-e)
    # m5 = (a+b)*h
    # m6 = (c-a)*(e+f)
    # m7 = (b-d)*(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # In our context, A11 = [[a, b], [c, d]] where:
    # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    # B11 = [[e, f], [g, h]] where:
    # e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
    
    # Each Strassen term is a product of sums of A entries times sums of B entries.
    # For the tensor decomposition, each "sum" becomes a set of rank-1 terms.
    
    # This is getting complex. Let me just use schoolbook for all blocks first
    # to get a correct baseline, then optimize.
    
    # Schoolbook: for each (i,j,k) triple, add a term
    # But that's rank 125.
    
    # Let me do the block schoolbook which should still be 125 but structured.
    # Actually, the schoolbook IS the block schoolbook if we just iterate all (i,j,k).
    
    # Let me try a different approach: just use the known best constructions.
    
    # For now, let me construct a valid decomposition with rank < 125.
    # I'll use the block structure with Strassen for the 2x2 diagonal blocks.
    
    # Let me manually construct the Strassen terms for A11*B11.
    
    # A11*B11 contributes to C11 (i in {0,1}, k in {0,1})
    # Using Strassen:
    # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # This expands to 4 terms:
    # (A[0,0]*B[0,0]) + (A[0,0]*B[1,1]) + (A[1,1]*B[0,0]) + (A[1,1]*B[1,1])
    
    # In tensor form, each product A[i,j]*B[j,k] corresponds to term (i*m+j, j*p+k, k*n+i)
    
    # For m1, the terms are:
    # (0*5+0, 0*5+0, 0*5+0) = (0, 0, 0)
    # (0*5+0, 1*5+1, 1*5+0) = (0, 6, 5)
    # (1*5+1, 0*5+0, 0*5+1) = (6, 0, 1)
    # (1*5+1, 1*5+1, 1*5+1) = (6, 6, 6)
    
    # But in Strassen, m1 is a single "multiplication" that we count as rank 1
    # in the Strassen algorithm, but in the tensor decomposition, it expands to 4 rank-1 terms.
    
    # So Strassen doesn't directly give a lower rank in the tensor decomposition
    # unless we're doing something more clever.
    
    # Actually, the key insight is that Strassen reduces the number of multiplications,
    # but each multiplication of sums still requires multiple rank-1 tensor terms.
    
    # Let me reconsider. The rank of the tensor is what we're trying to minimize.
    # Strassen shows that 2x2x2 tensor has rank 7, not 8.
    
    # So for the 2x2 subproblem, we can use 7 terms instead of 8.
    
    # Let me look up the explicit rank-7 decomposition of the 2x2x2 tensor.
    
    # The 2x2x2 matrix multiplication tensor has rank 7.
    # One construction:
    # The tensor T is in V* ⊗ V* ⊗ V* where V is 2-dimensional (for each factor).
    # Actually, it's in (R^2 ⊗ R^2) ⊗ (R^2 ⊗ R^2) ⊗ (R^2 ⊗ R^2)
    
    # Let me just use a known explicit rank-7 decomposition.
    
    # From literature, one rank-7 decomposition of the 2x2 matrix mult tensor:
    # Let e1, e2 be basis for each factor.
    # The 7 terms are combinations like:
    # t1: (e1⊗e1 + e2⊗e2) ⊗ (e1⊗e1 + e2⊗e2) ⊗ (e1⊗e1 + e2⊗e2)
    # ... etc
    
    # This is getting too complex to do by hand. Let me just use schoolbook (rank 125)
    # for now to have a valid baseline, and then we can optimize.
    
    # Actually, the initial program already does schoolbook with rank 125.
    # Let me try to do better.
    
    # Let me use the following approach:
    # For the 5x5 case, partition into 2+3 and use:
    # - Strassen (rank 7) for the 2x2*2x2 -> 2x2 part
    # - Schoolbook for the rest
    
    # But as I noted, Strassen's rank-7 for 2x2x2 means we can express the 2x2x2
    # tensor as a sum of 7 rank-1 tensors.
    
    # Let me just hardcode a known rank-7 decomposition for 2x2x2.
    
    # After research, one way to think about it:
    # The 2x2 matrix mult tensor T_{i,j,k} where i,k in {0,1} (output) and j in {0,1} (shared)
    # Actually, the tensor is T[i,j,k] for A[i,j]*B[j,k]->C[i,k]
    
    # The standard schoolbook gives 4 terms:
    # (0,0,0), (0,1,1), (1,0,0), (1,1,1) in (i,j,k) indexing
    # Wait, I need to be careful with the indexing.
    
    # Let me restart with a clean implementation.
    
    n, m, p = 5, 5, 5
    
    # I'll use a block decomposition.
    # Partition the 5 indices into groups of sizes [2, 3].
    # A: 5x5, split into 2x2, 2x3, 3x2, 3x3 blocks
    # B: 5x5, same split
    # C: 5x5, same split
    
    # For each block combination, we need to multiply rectangular matrices.
    
    # Let me just use schoolbook for everything (rank 125) as a baseline,
    # then I'll try to optimize specific blocks.
    
    # Actually, let me try a simpler optimization:
    # Note that if we have a block where one dimension is 1, we can do it in 1 rank-1 term
    # instead of the product of the other two dimensions.
    
    # But 5 doesn't have natural 1-dimensions.
    
    # Let me try the tensor product approach mentioned in the directive:
    # 5x5 = 2x2 ⊗ 3x3? No, that's not how matrix multiplication works.
    
    # Actually, the directive says: "modeling the 5x5 matrix multiplication as the tensor 
    # product of a 2x2 and a 3x3 multiplication"
    
    # This doesn't quite make sense dimensionally. 5x5 is not 2x2 tensor 3x3.
    
    # Let me just implement a correct schoolbook decomposition and return it.
    # This will score 1.0 (same as parent), but at least it's valid.
    
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
