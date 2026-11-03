"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Split 5 = 2 + 3 into two blocks.
    # A = [[A22(2x2), A23(2x3)],
    #      [A32(3x2), A33(3x3)]]
    # B = [[B22(2x2), B23(2x3)],
    #      [B32(3x2), B33(3x3)]]
    # C = [[C22(2x2), C23(2x3)],
    #      [C32(3x2), C33(3x3)]]
    #
    # C22 = A22*B22         -> Strassen 2x2*2x2, rank 7
    # C23 = A22*B23 + A23*B33 -> schoolbook 2x2*2x3 (rank 6) + 2x3*3x3 (rank 18) = 24
    # C32 = A32*B22 + A33*B32 -> schoolbook 3x2*2x2 (rank 6) + 3x3*3x2 (rank 18) = 24
    # C33 = A32*B32 + A33*B33 -> schoolbook 3x2*2x3 (rank 18) + 3x3*3x3 (rank 27) = 45
    #
    # Total: 7 + 24 + 24 + 45 = 100

    # We'll build factor lists as (u_idx, v_idx, w_idx, coeff) tuples
    # u_idx is row in U (size n*m = 25), v_idx is row in V (size m*p = 25), w_idx is row in W (size p*n = 25)
    # Using trace convention: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read: W[k*n+i,r]
    # The tensor for matrix mult: T[i,j,k] = delta (A[i,j]*B[j,k] -> C[i,k])
    # Factorization: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # But the task says W[k*n+i, r], so W is indexed as W[k*n+i, r] meaning the first index is k*n+i.
    # So W row index = k*n + i, which corresponds to (i,k) pair but stored as k*n+i.
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, r] is the entry for output position (i,k).
    
    # Let me define indices carefully:
    # A is n x m = 5 x 5, indexed A[i,j] for i in 0..4, j in 0..4
    # B is m x p = 5 x 5, indexed B[j,k] for j in 0..4, k in 0..4  
    # C is n x p = 5 x 5, indexed C[i,k] for i in 0..4, k in 0..4
    #
    # U has shape (n*m, rank) = (25, rank), U[i*m+j, r]
    # V has shape (m*p, rank) = (25, rank), V[j*p+k, r]
    # W has shape (p*n, rank) = (25, rank), W[k*n+i, r]
    #
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r and j
    # Actually: sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) ... no
    # The standard tensor factorization: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where T[i,j,k] = 1 if this is the (i,j,k) term in the bilinear form
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 (all ones in the relevant sense)
    # Factorization: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    # Wait no. The matrix multiplication tensor is:
    # M_{i,j,k} such that C[i,k] = sum_{j} M_{i,j,k} A[i,j] B[j,k]
    # Here M_{i,j,k} = 1 for all i,j,k (since each term A[i,j]*B[j,k] contributes exactly once)
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    factors = []  # list of (u_idx, v_idx, w_idx, coeff)
    
    def add_factor(u_idx, v_idx, w_idx, coeff=1):
        factors.append((u_idx, v_idx, w_idx, coeff))
    
    # Block indices:
    # Block 0: rows 0,1 (size 2)
    # Block 1: rows 2,3,4 (size 3)
    
    # C22 = A22 * B22 (2x2 * 2x2) -> Strassen
    # A22 = A[0:2, 0:2], B22 = B[0:2, 0:2], C22 = C[0:2, 0:2]
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For Strassen, each Mi is a product of two "sums" which requires multiple rank-1 terms.
    # Actually, Strassen gives rank 7 for 2x2*2x2. Let me implement it directly.
    #
    # The 7 Strassen products:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[0,0])
    # M4 = A[1,1]*(B[0,1]-B[1,1])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[1,0])
    # M7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])
    #
    # Each Mi = (linear combo of A entries) * (linear combo of B entries)
    # This is a rank-1 term in the tensor decomposition!
    # For Mi = (sum_a c_a * A[i_a, j_a]) * (sum_b d_b * B[j_b, k_b])
    # We need: sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For a rank-1 term: U[i*m+j, r] = c_a if (i,j) = (i_a, j_a) else 0
    # V[j*p+k, r] = d_b if (j,k) = (j_b, k_b) else 0
    # W[k*n+i, r] = 1
    #
    # But wait, a single rank-1 term only contributes to ONE (i,j,k) triple.
    # For Strassen, M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # This means: for the C[0,0] output (i=0,k=0):
    #   contribution from j=0: A[0,0]*B[0,0] -> need U[0*5+0, r]*V[0*5+0, r]*W[0*5+0, r] = 1
    #   contribution from j=1: A[1,1]*B[0,0]... wait no.
    #
    # I'm overcomplicating this. Let me think again.
    #
    # The tensor M_{i,j,k} = 1 for all i,j,k in {0,...,4}.
    # A factorization of rank R means: M_{i,j,k} = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For schoolbook: for each (i,j,k), one rank-1 term with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    #
    # For Strassen 2x2*2x2 (using indices 0,1 for the 2x2 block):
    # We need to cover all (i,j,k) with i,j,k in {0,1}.
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) contributes to:
    #   (i=0,j=0,k=0): A[0,0]*B[0,0] -> coefficient 1
    #   (i=0,j=0,k=1): A[0,0]*B[0,1]... no wait.
    #
    # Hmm, I think I need to be more careful. In the Strassen decomposition of 2x2*2x2:
    # The 7 products each compute a bilinear form. Let me just directly enumerate.
    #
    # Actually, the cleanest way: for each of the 7 Strassen products, it's a product of
    # (sum of A entries) * (sum of B entries). Each such product expands to multiple (i,j,k) terms.
    # But in the tensor factorization, each rank-1 term corresponds to ONE (i,j,k) triple.
    # So a Strassen product like (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) = A[0,0]B[0,0] + A[0,0]B[1,0] + A[1,1]B[0,0] + A[1,1]B[1,0]
    # This is 4 separate (i,j,k) terms, each needing its own rank-1 factor.
    #
    # Wait, that's not right either. Let me reconsider.
    #
    # In the tensor M_{i,j,k}, each entry is 1. A rank-1 term U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # with U, V, W being vectors. For a "simple" rank-1 term, U has a 1 at position (i0,j0) and 0 elsewhere,
    # V has a 1 at position (j0,k0) and 0 elsewhere, W has a 1 at position (k0,i0) and 0 elsewhere.
    # This contributes 1 to M_{i0,j0,k0} only.
    #
    # For Strassen, the key insight is that we can have U, V, W with multiple non-zero entries.
    # E.g., for M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
    # U[0*5+0, r] = 1, U[1*5+1, r] = 1, others 0
    # V[0*5+0, r] = 1, V[1*5+0, r] = 1, others 0  -- wait, V is indexed by (j,k)
    #
    # Let me redo: V[j*p+k, r] where p=5. So V[j*5+k, r].
    # For M1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # This bilinear form = A[0,0]*B[0,0] + A[0,0]*B[1,0] + A[1,1]*B[0,0] + A[1,1]*B[1,0]
    # In terms of M_{i,j,k}:
    #   A[0,0]*B[0,0] -> i=0,j=0,k=0
    #   A[0,0]*B[1,0] -> i=0,j=1,k=0
    #   A[1,1]*B[0,0] -> i=1,j=0,k=0
    #   A[1,1]*B[1,0] -> i=1,j=1,k=0
    #
    # So for this rank-1 term:
    # U[0*5+0, r] = 1, U[1*5+1, r] = 1
    # V[0*5+0, r] = 1, V[1*5+0, r] = 1
    # W[0*5+0, r] = 1, W[0*5+1, r] = 1
    #
    # Check: sum over r of U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r]
    # For (i=0,j=0,k=0): U[0, r]*V[0, r]*W[0, r] = 1*1*1 = 1 ✓
    # For (i=0,j=1,k=0): U[1, r]*V[5, r]*W[0, r] = 1*1*1 = 1 ✓
    # For (i=1,j=0,k=0): U[5, r]*V[0, r]*W[0, r] = 1*1*1 = 1 ✓
    # For (i=1,j=1,k=0): U[6, r]*V[5, r]*W[0, r] = 1*1*1 = 1 ✓
    # For (i=0,j=0,k=1): U[0, r]*V[1, r]*W[5, r] = 1*0*0 = 0 ✓ (V[0*5+1]=V[1]=0)
    # Good!
    
    # Now let me implement Strassen for the 2x2 block (indices 0,1)
    # The 7 products:
    
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # U: (0,0)=1, (1,1)=1 -> indices 0, 6
    # V: (0,0)=1, (1,0)=1 -> indices 0, 5
    # W: (0,0)=1, (1,0)=1 -> indices 0, 5  (W[k*5+i], so (k=0,i=0)=0, (k=0,i=1)=1... wait)
    # W[k*n+i, r] with n=5: W[k*5+i, r]
    # For output (i=0,k=0): W[0*5+0, r] = W[0, r]
    # For output (i=1,k=0): W[0*5+1, r] = W[1, r]
    # So W indices: 0 and 1
    add_factor_set = []
    
    # Let me just build the full factor matrices directly.
    # I'll create a list of (u_indices, v_indices, w_indices, coeff) for each rank-1 term.
    
    rank_terms = []
    
    def add_strassen_term(u_idxs, v_idxs, w_idxs, coeff=1):
        rank_terms.append((u_idxs, v_idxs, w_idxs, coeff))
    
    # Strassen 2x2*2x2 for block (0,1) x (0,1) -> output (0,1) x (0,1)
    # M1 = (A00+A11)(B00+B11)
    add_strassen_term([0, 6], [0, 5], [0, 1], 1)
    # M2 = (A10+A11)B00
    add_strassen_term([5, 6], [0], [0, 1], 1)
    # M3 = A00(B10-B00)
    add_strassen_term([0], [5, 0], [0, 1], 1)
    # M4 = A11(B01-B11)
    add_strassen_term([6], [1, 6], [0, 1], 1)
    # M5 = (A00+A01)B11
    add_strassen_term([0, 1], [6], [0, 1], 1)
    # M6 = (A10-A00)(B00+B10)
    add_strassen_term([5, 0], [0, 5], [0, 1], 1)
    # M7 = (A01-A11)(B01+B11)
    add_strassen_term([1, 6], [1, 6], [0, 1], 1)
    
    # Now I need to verify these give the correct coefficients.
    # The Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # But in our tensor, each M_i contributes to multiple (i,j,k) entries simultaneously.
    # The issue is that the coefficients in the Strassen formulas involve additions and subtractions.
    # In the tensor factorization, we can't have a single rank-1 term with coefficient -1 for one
    # (i,j,k) and +1 for another.
    #
    # This means the simple Strassen approach doesn't directly translate to a tensor factorization
    # with the same rank! The Strassen rank-7 result is for the number of scalar multiplications,
    # not the tensor rank in this specific factorization form.
    #
    # Actually wait - it DOES translate. The tensor rank of 2x2*2x2 is 7, and Strassen achieves it.
    # The key is that each of the 7 rank-1 terms has specific U, V, W vectors, and the sum of all 7
    # gives the correct tensor. Let me look up the explicit Strassen tensor factorization.
    #
    # For 2x2*2x2, the 7 rank-1 terms (using 1-based indexing for A[i,j], B[j,k], C[i,k]):
    # Term 1: U has 1s at (1,1) and (2,2), V has 1s at (1,1) and (2,1), W has 1s at (1,1) and (2,1)
    #   -> This is (A11+A22)(B11+B21) contributing to C11 and C21
    # Term 2: U has 1s at (2,1) and (2,2), V has 1 at (1,1), W has 1s at (1,2) and (2,2)
    #   -> (A21+A22)B11 contributing to C12 and C22... 
    #
    # I'm getting confused with the indexing. Let me just use the known result that the tensor rank
    # of 2x2*2x2 is 7 and implement it correctly by solving the system.
    #
    # Actually, for the purpose of this problem, let me just use schoolbook for everything.
    # The schoolbook decomposition is guaranteed correct and gives rank 125.
    # Given the complexity of implementing Strassen correctly in this tensor factorization format,
    # and the risk of bugs, let me stick with schoolbook for now.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
