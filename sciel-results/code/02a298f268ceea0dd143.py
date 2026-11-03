import numpy as np
import time
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products
    # M1 = (A11)(B12-B22)
    # M2 = (A11+A22)(B22)
    # M3 = (A22-A11)(B11)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M2-M4-M7
    # C12 = M5+M7
    # C21 = M3+M4
    # C22 = M1-M2+M3+M5
    
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    
    # Index mapping: A row i, col j -> U index i*2+j
    # B row j, col k -> V index j*2+k
    # C row i, col k -> W index k*2+i (trace convention W[k*n+i])
    
    # For 2x2x2: n=2, m=2, p=2
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # M1 = (A11)(B12-B22): A[0,0]*B[0,1] - A[0,0]*B[1,1]
    # contributes to C11 = C[0,0] = W[0*2+0] = W[0]
    U[0, 0] = 1  # A[0,0]
    V[1, 0] = 1  # B[0,1]
    W[0, 0] = 1  # C[0,0]
    
    U[0, 1] = -1  # A[0,0]
    V[3, 1] = 1   # B[1,1]
    W[0, 1] = -1  # C[0,0]
    
    # M2 = (A11+A22)(B22): (A[0,0]+A[1,1])*B[1,1]
    # contributes to C11 and C22
    U[0, 2] = 1   # A[0,0]
    U[3, 2] = 1   # A[1,1]
    V[3, 2] = 1   # B[1,1]
    W[0, 2] = 1   # C[0,0]
    W[3, 2] = -1  # C[1,1]
    
    # M3 = (A22-A11)(B11): (A[1,1]-A[0,0])*B[0,0]
    # contributes to C21 and C22
    U[3, 3] = 1   # A[1,1]
    U[0, 3] = -1  # A[0,0]
    V[0, 3] = 1   # B[0,0]
    W[1, 3] = 1   # C[1,0]
    W[3, 3] = 1   # C[1,1]
    
    # M4 = (A22)(B21-B11): A[1,1]*(B[1,0]-B[0,0])
    # contributes to C11 and C21
    U[3, 4] = 1   # A[1,1]
    V[2, 4] = 1   # B[1,0]
    W[0, 4] = -1  # C[0,0]
    W[1, 4] = 1   # C[1,0]
    
    U[3, 5] = -1  # A[1,1]
    V[0, 5] = 1   # B[0,0]
    W[0, 5] = 1   # C[0,0]
    W[1, 5] = -1  # C[1,0]
    
    # M5 = (A11+A12)(B22): (A[0,0]+A[0,1])*B[1,1]
    # contributes to C12 and C22
    U[0, 6] = 1   # A[0,0]
    U[1, 6] = 1   # A[0,1]
    V[3, 6] = 1   # B[1,1]
    W[1, 6] = 1   # C[0,1]
    W[3, 6] = 1   # C[1,1]
    
    # M6 = (A21-A11)(B11+B12): (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # contributes to C21
    # This needs more terms... Let me use a cleaner formulation
    
    return U, V, W


def _build_strassen_2x2x2_correct():
    """Correct Strassen 2x2x2 decomposition with proper index mapping.
    
    For matrices A(2x2), B(2x2), C(2x2):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Index conventions:
    U[i*m+j, r] where m=2, so U[i*2+j, r]
    V[j*p+k, r] where p=2, so V[j*2+k, r]  
    W[k*n+i, r] where n=2, so W[k*2+i, r]
    """
    n = m = p = 2
    rank = 7
    
    U = np.zeros((n*m, rank), dtype=object)  # (4, 7)
    V = np.zeros((m*p, rank), dtype=object)  # (4, 7)
    W = np.zeros((p*n, rank), dtype=object)  # (4, 7)
    
    # Strassen's algorithm:
    # M1 = (A11)(B12-B22)
    # M2 = (A11+A22)(B22)  
    # M3 = (A22-A11)(B11)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M2 - M4 - M7
    # C12 = M5 + M7
    # C21 = M3 + M4
    # C22 = M1 - M2 + M3 + M5
    
    # Each M is a product of linear combinations, so each M contributes
    # to multiple entries in U, V, W for each term in the linear combination.
    
    # Let me define each product term explicitly.
    # M1 = A[0,0] * (B[0,1] - B[1,1])
    # This is: A[0,0]*B[0,1] - A[0,0]*B[1,1]
    # In tensor form, this contributes to C[0,0] (C11)
    
    # For a single product A[i,j]*B[j,k] contributing to C[i,k]:
    # U[i*2+j, r] += coeff
    # V[j*2+k, r] += coeff  
    # W[k*2+i, r] += coeff
    
    # M1 = A[0,0]*B[0,1] - A[0,0]*B[1,1]
    # Term 1: A[0,0]*B[0,1] -> C[0,0], coeff +1
    U[0*2+0, 0] += 1  # A[0,0]
    V[0*2+1, 0] += 1  # B[0,1]
    W[0*2+0, 0] += 1  # C[0,0]
    
    # Term 2: A[0,0]*B[1,1] -> C[0,0], coeff -1
    U[0*2+0, 0] += 0  # already counted
    # Actually, M1 is ONE rank-1 term with a linear combination in V
    # Let me reconsider. Each M is a product of two linear forms.
    # M1 = (sum a_i A[i,j]) * (sum b_k B[j,k])
    # This is a rank-1 tensor: U[i,j] = a_i, V[j,k] = b_k, W[k,i] = 1
    # But only if the linear forms are over the same index j.
    
    # Actually for Strassen, each M_r is:
    # M_r = (sum_{i,j} u_{ij}^{(r)} A[i,j]) * (sum_{j,k} v_{jk}^{(r)} B[j,k])
    # And it contributes to C as: C[i,k] += w_{ki}^{(r)} * M_r
    # But this isn't quite right either.
    
    # The standard tensor decomposition for matrix mult:
    # T[i,j,k] = 1 = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U, V, W are the factor matrices.
    
    # For Strassen 2x2x2, the 7 terms are:
    # r=0: M1 = A[0,0] * (B[0,1]-B[1,1])
    #   U[0,0,0]=1, V[0,1,0]=1, V[1,1,0]=-1, W[0,0,0]=1
    #   But this means U has only one nonzero, V has two, W has one.
    #   The product sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] would give:
    #   For C[0,0]: U[0,0,0]*V[0,0,0]*W[0,0,0] + ... 
    #   Hmm, this doesn't work because V[0,0,0]=0 but we need B[0,1].
    
    # I think the issue is that each rank-1 term must have exactly one
    # nonzero in each of U, V, W for a single product A[i,j]*B[j,k].
    # Strassen's algorithm uses LINEAR COMBINATIONS, so each "term" is
    # actually a sum of rank-1 terms.
    
    # Let me just enumerate all 8 products and then apply Strassen's
    # linear combinations to reduce to 7.
    
    # Schoolbook 2x2x2 has 8 terms:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen replaces these 8 with 7 products of linear combinations.
    # The key insight: we can't just have 7 rank-1 terms with single
    # nonzeros. We need to allow U, V, W to have multiple nonzeros per column.
    
    # Let me define the 7 Strassen products as linear forms:
    # P1 = A[0,0] * (B[0,1] - B[1,1])
    # P2 = (A[0,0] + A[1,1]) * B[1,1]
    # P3 = (A[1,1] - A[0,0]) * B[0,0]
    # P4 = A[1,1] * (B[1,0] - B[0,0])
    # P5 = (A[0,0] + A[0,1]) * B[1,1]
    # P6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # P7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # Then:
    # C[0,0] = P1 + P2 - P4 - P7
    # C[0,1] = P5 + P7
    # C[1,0] = P3 + P4
    # C[1,1] = P1 - P2 + P3 + P5
    
    # Now, each P_r is a product of a linear form in A and a linear form in B.
    # P_r = (sum_{i,j} alpha_{ij}^{(r)} A[i,j]) * (sum_{j,k} beta_{jk}^{(r)} B[j,k])
    #     = sum_{i,j,j',k} alpha_{ij}^{(r)} beta_{j'k}^{(r)} A[i,j] B[j',k]
    # This is NOT a rank-1 tensor in general because the j and j' indices
    # are independent.
    
    # However, in the standard tensor decomposition, each rank-1 term is:
    # U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where the same j index appears in both U and V.
    
    # So for P_r to be a single rank-1 term, we need the linear forms to
    # share the same j index. Let me check:
    # P1 = A[0,0] * (B[0,1] - B[1,1]): j=0 in A, j=0,1 in B. Not shared.
    
    # I see the issue. Strassen's algorithm works by precomputing linear
    # combinations and then doing 7 scalar multiplications. In the tensor
    # decomposition sense, each "product" of linear combinations expands
    # into multiple rank-1 terms.
    
    # The correct approach: the tensor T[i,j,k] = 1 for 2x2x2 has rank 7.
    # A rank-7 decomposition exists. Let me find it explicitly.
    
    # Actually, I recall that the Strassen decomposition corresponds to:
    # For each of the 7 products, we have a rank-1 tensor where:
    # U[i,j,r] is 1 if A[i,j] appears in the first linear form of product r
    # V[j,k,r] is 1 if B[j,k] appears in the second linear form of product r
    # W[k,i,r] is the coefficient of C[k,i] from product r
    # But this only works if the j indices match up properly.
    
    # Let me try a different approach: directly construct a valid rank-7
    # decomposition by solving the system.
    
    # For now, let me use the schoolbook for 2x2x2 (rank 8) and rely on
    # the hybrid approach for the overall 4x5x6 problem.
    
    # Actually, let me look up the explicit Strassen tensor decomposition.
    # The 7 rank-1 tensors for 2x2x2 are:
    
    # Term 1: U[0,0]=1, V[0,1]=1, W[0,0]=1  (A00*B01 -> C00)
    # Term 2: U[0,0]=-1, V[1,1]=1, W[0,0]=-1 (A00*B11 -> C00, with sign)
    # But this gives two separate rank-1 terms, not one.
    
    # I think the confusion is that in the standard presentation, Strassen's
    # 7 "products" each involve multiplying two 2x2 submatrices that are
    # linear combinations. In the tensor decomposition, each such product
    # of linear combinations (alpha . A) * (B . beta) where alpha and beta
    # are vectors, expands to sum_{i,j,k} alpha_i beta_k A[i,j] B[j,k]
    # which is NOT rank 1 unless alpha and beta have special structure.
    
    # Let me just use rank 8 (schoolbook) for 2x2x2 and focus on the
    # hybrid decomposition for 4x5x6.
    
    return None


def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition for n x m * m x p.
    Returns U (n*m x n*m*p), V (m*p x n*m*p), W (p*n x n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def _combine_decompositions(decomps, offsets):
    """Combine multiple decompositions into one.
    decomps: list of (U, V, W) tuples
    offsets: list of (u_offset, v_offset, w_offset) for index shifting
    Returns combined (U, V, W)
    """
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    
    # Determine total sizes
    n = decomps[0][2].shape[0] // decomps[0][0].shape[0] * decomps[0][0].shape[0]
    # Actually, let's just concatenate
    U_list = []
    V_list = []
    W_list = []
    
    for (U, V, W), (uo, vo, wo) in zip(decomps, offsets):
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
    
    U_combined = np.hstack(U_list)
    V_combined = np.hstack(V_list)
    W_combined = np.hstack(W_list)
    
    return U_combined, V_combined, W_combined


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    U_3d = U.T.reshape(n, m, rank)
    V_3d = V.T.reshape(m, p, rank)
    W_3d = W.T.reshape(p, n, rank)
    
    tensor = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    target = np.ones((n, m, p))
    
    error = np.max(np.abs(tensor - target))
    return error < tol, error


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 140  # 10s safety margin
    
    # Strategy: Hybrid decomposition
    # Split 4x5x6 into blocks:
    # - 4x4x2 block (top-left 4x4 of A, 4x2 of B, producing 4x2 of C)
    # - 4x1x4 block (top-right 4x1 of A, 1x4 of B, producing 4x4 of C)
    # - 0x4x2 doesn't make sense...
    
    # Better split:
    # A is 4x5, B is 5x6
    # Split A into A1 (4x4) and A2 (4x1)
    # Split B into B1 (4x6) and B2 (1x6)
    # C = A1*B1 + A2*B2
    # A1*B1 is 4x4 * 4x6 = 4x6, rank at most 4*4*6=96, or use Strassen
    # A2*B2 is 4x1 * 1x6 = 4x6, rank at most 4*1*6=24
    
    # Total rank: rank(A1*B1) + rank(A2*B2)
    # For A1*B1 (4x4x6): can split further
    #   B1 is 4x6, split into B1a (4x4) and B1b (4x2)
    #   A1*B1a is 4x4x4, A1*B1b is 4x4x2
    #   4x4x4: schoolbook rank 64, or Strassen-based
    #   4x4x2: schoolbook rank 32
    
    # Let's try:
    # Block 1: 4x4x4 (A1, B1a) - use 4 Strassen 2x2x2 blocks (rank 4*7=28) or schoolbook (rank 64)
    # Block 2: 4x4x2 (A1, B1b) - schoolbook rank 32
    # Block 3: 4x1x6 (A2, B2) - schoolbook rank 24
    # Total: 28 + 32 + 24 = 84 if Strassen works, or 64+32+24=120 schoolbook
    
    # Let me implement the Strassen 2x2x2 properly first.
    
    def _strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7), dtype=object)
        V = np.zeros((4, 7), dtype=object)
        W = np.zeros((4, 7), dtype=object)
        
        # The 7 Strassen products, each as a rank-1 tensor:
        # For product P_r = (linear form in A) * (linear form in B):
        # We need U[i,j,r], V[j,k,r], W[k,i,r] such that
        # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{i,k} (for the C matrix)
        # Wait, no. The tensor is T[i,j,k] = 1 where:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k.
        
        # A rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # For 2x2x2, r=7. Let me find the explicit decomposition.
        # This is a known result. The 7 terms are:
        
        # I'll use the formulation where each term has the form:
        # U[i,j,r] = a_{ij}^{(r)}, V[j,k,r] = b_{jk}^{(r)}, W[k,i,r] = c_{ki}^{(r)}
        # with the constraint that for each r, the "j" index in U and V
        # must be consistent (same j).
        
        # Actually, in the standard tensor decomposition, U[i,j,r] and
        # V[j,k,r] share the j index. So for each r, U has nonzeros at
        # specific (i,j) positions, V has nonzeros at specific (j,k) positions,
        # and W has nonzeros at specific (k,i) positions.
        
        # For a single scalar product A[i0,j0]*B[j0,k0] contributing to C[i0,k0]:
        # U[i0,j0,r]=1, V[j0,k0,r]=1, W[k0,i0,r]=1
        
        # Strassen's trick: instead of 8 such terms, use 7 terms where
        # some U, V, W entries are sums.
        
        # Let me just hardcode the known rank-7 decomposition for 2x2x2.
        # After some research, the decomposition is:
        
        # r=0: U[0,0]=1, V[0,1]=1, W[0,0]=1
        #       U[0,0]=-1, V[1,1]=1, W[0,0]=-1  -- no, this is two terms
        
        # I think I need to be more careful. Let me derive it.
        
        # We want: for all i,k: delta_{ik} = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Wait, that's not right either. Let me re-examine.
        
        # The multiplication tensor: T[i,j,k] = 1 for all i,j,k in {0,1}
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # So: 1 = sum_r U[i,j,r] V[j,k,r] W[k,i,r] for all i,j,k
        
        # This is a system of 8 equations (for i,j,k in {0,1}) with 7*4*3=84 unknowns
        # (4 entries in U per r, 4 in V per r, 4 in W per r, times 7 r's).
        
        # Let me try a specific construction.
        
        # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1  (A00*B00 -> C00)
        # Term 1: U[0,1]=1, V[1,0]=1, W[0,0]=1  (A01*B10 -> C00)
        # Term 2: U[0,0]=1, V[0,1]=1, W[1,0]=1  (A00*B01 -> C01)
        # Term 3: U[0,1]=1, V[1,1]=1, W[1,0]=1  (A01*B11 -> C01)
        # Term 4: U[1,0]=1, V[0,0]=1, W[0,1]=1  (A10*B00 -> C10)
        # Term 5: U[1,1]=1, V[1,0]=1, W[0,1]=1  (A11*B10 -> C10)
        # Term 6: U[1,0]=1, V[0,1]=1, W[1,1]=1  (A10*B01 -> C11)
        # Term 7: U[1,1]=1, V[1,1]=1, W[1,1]=1  (A11*B11 -> C11)
        # That's 8 terms (schoolbook).
        
        # To get 7 terms, we need to merge some. The key is that the tensor
        # T[i,j,k]=1 has rank 7, so a rank-7 decomposition exists.
        
        # Let me try:
        # r=0: U[0,0]=1, V[0,1]=1, W[0,0]=1
        #       Contribution to T[0,0,0]: U[0,0,0]*V[0,0,0]*W[0,0,0] = 1*0*1 = 0
        #       Contribution to T[0,0,1]: U[0,0,0]*V[0,1,0]*W[1,0,0] = 1*1*0 = 0
        #       Hmm, this is getting complicated.
        
        # Let me just use rank 8 for now and see if the hybrid approach
        # still gives an improvement.
        
        U = np.zeros((4, 8), dtype=object)
        V = np.zeros((4, 8), dtype=object)
        W = np.zeros((4, 8), dtype=object)
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j, col] = 1
                    V[j*2+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        
        return U, V, W
    
    # Build the hybrid decomposition
    # Split: A = [A1 (4x4) | A2 (4x1)], B = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    
    # Block 1: A1 (4x4) * B1 (4x6)
    #   Split B1 = [B1a (4x4) | B1b (4x2)]
    #   Block 1a: A1 (4x4) * B1a (4x4) -> 4x4
    #   Block 1b: A1 (4x4) * B1b (4x2) -> 4x2
    
    # Block 2: A2 (4x1) * B2 (1x6) -> 4x6
    
    # For Block 1a (4x4x4): use 4 Strassen 2x2x2 sub-blocks
    #   Split A1 into 4 2x2 blocks, B1a into 4 2x2 blocks
    #   Each 2x2*2x2 = 2x2 uses rank 8 (or 7 if we can get Strassen)
    #   But the blocks overlap in the output, so we need to be careful.
    
    # Actually, for 4x4x4, we can use 4 independent 2x2x2 blocks:
    # A1 = [[A11, A12], [A21, A22]] where each is 2x2
    # B1a = [[B11, B12], [B21, B22]] where each is 2x2
    # C1a = A1*B1a = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #                 [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    # Each 2x2*2x2 product is independent, so we can decompose each separately.
    # Total rank: 4 * rank(2x2x2) = 4*8 = 32 (schoolbook) or 4*7=28 (Strassen)
    
    # For Block 1b (4x4x2): schoolbook rank = 4*4*2 = 32
    # For Block 2 (4x1x6): schoolbook rank = 4*1*6 = 24
    
    # Total with schoolbook 2x2x2: 32 + 32 + 24 = 88
    # Total with Strassen 2x2x2: 28 + 32 + 24 = 84
    
    # Let's build this.
    
    def _build_4x4x4_strassen():
        """Build rank-32 (or 28) decomposition for 4x4x4 using 4 2x2x2 blocks."""
        n = m = p = 4
        # Use 4 independent 2x2x2 blocks
        # A[i,j] for i,j in 0..3, B[j,k] for j,k in 0..3, C[i,k] for i,k in 0..3
        # Block (bi, bk): A[2*bi:2*bi+2, 2*bj:2*bj+2] * B[2*bj:2*bj+2, 2*bk:2*bk+2]
        #   -> C[2*bi:2*bi+2, 2*bk:2*bk+2]
        
        U_blocks = []
        V_blocks = []
        W_blocks = []
        
        for bi in range(2):
            for bk in range(2):
                for bj in range(2):
                    # 2x2x2 block
                    U_b, V_b, W_b = _strassen_2x2x2()
                    
                    # Shift indices
                    U_shifted = np.zeros((16, U_b.shape[1]), dtype=object)
                    V_shifted = np.zeros((16, V_b.shape[1]), dtype=object)
                    W_shifted = np.zeros((16, W_b.shape[1]), dtype=object)
                    
                    for r in range(U_b.shape[1]):
                        for i_local in range(2):
                            for j_local in range(2):
                                i_global = 2*bi + i_local
                                j_global = 2*bj + j_local
                                U_shifted[i_global*4 + j_global, r] = U_b[i_local*2 + j_local, r]
                        
                        for j_local in range(2):
                            for k_local in range(2):
                                j_global = 2*bj + j_local
                                k_global = 2*bk + k_local
                                V_shifted[j_global*4 + k_global, r] = V_b[j_local*2 + k_local, r]
                        
                        for k_local in range(2):
                            for i_local in range(2):
                                k_global = 2*bk + k_local
                                i_global = 2*bi + i_local
                                W_shifted[k_global*4 + i_global, r] = W_b[k_local*2 + i_local, r]
                    
                    U_blocks.append(U_shifted)
                    V_blocks.append(V_shifted)
                    W_blocks.append(W_shifted)
        
        U = np.hstack(U_blocks)
        V = np.hstack(V_blocks)
        W = np.hstack(W_blocks)
        
        return U, V, W
    
    def _build_4x4x2_schoolbook():
        """Build schoolbook decomposition for 4x4x2."""
        n, m, p = 4, 4, 2
        rank = n * m * p  # 32
        U = np.zeros((n*m, rank), dtype=object)
        V = np.zeros((m*p, rank), dtype=object)
        W = np.zeros((p*n, rank), dtype=object)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return U, V, W
    
    def _build_4x1x6_schoolbook():
        """Build schoolbook decomposition for 4x1x6."""
        n, m, p = 4, 1, 6
        rank = n * m * p  # 24
        U = np.zeros((n*m, rank), dtype=object)
        V = np.zeros((m*p, rank), dtype=object)
        W = np.zeros((p*n, rank), dtype=object)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return U, V, W
    
    # Build each block and combine
    U1, V1, W1 = _build_4x4x4_strassen()  # 4x4x4, rank 32
    U2, V2, W2 = _build_4x4x2_schoolbook()  # 4x4x2, rank 32
    U3, V3, W3 = _build_4x1x6_schoolbook()  # 4x1x6, rank 24
    
    # Now we need to place these blocks in the full 4x5x6 tensor.
    # Full tensor: A (4x5), B (5x6), C (4x6)
    # A = [A1 (4x4) | A2 (4x1)]
    # B = [B1 (4x6); B2 (1x6)]
    # B1 = [B1a (4x4) | B1b (4x2)]
    
    # Block 1: A1 (cols 0-3 of A) * B1 (rows 0-3 of B) -> C (all 4x6)
    #   Block 1a: A1 (cols 0-3) * B1a (cols 0-3 of B1, i.e., cols 0-3 of B) -> C cols 0-3
    #   Block 1b: A1 (cols 0-3) * B1b (cols 4-5 of B1, i.e., cols 4-5 of B) -> C cols 4-5
    # Block 2: A2 (col 4 of A) * B2 (row 4 of B) -> C (all 4x6)
    
    # Full U: (4*5, total_rank) = (20, total_rank)
    # Full V: (5*6, total_rank) = (30, total_rank)
    # Full W: (6*4, total_rank) = (24, total_rank)
    
    total_rank = U1.shape[1] + U2.shape[1] + U3.shape[1]
    
    U_full = np.zeros((20, total_rank), dtype=object)
    V_full = np.zeros((30, total_rank), dtype=object)
    W_full = np.zeros((24, total_rank), dtype=object)
    
    # Block 1a: A1 (4x4, cols 0-3) * B1a (4x4, rows 0-3, cols 0-3) -> C (4x4, cols 0-3)
    # U1[i*4+j, r] -> U_full[i*5+j, r] for i,j in 0..3
    # V1[j*4+k, r] -> V_full[j*6+k, r] for j in 0..3, k in 0..3
    # W1[k*4+i, r] -> W_full[k*4+i, r] for k in 0..3, i in 0..3
    
    for r in range(U1.shape[1]):
        for i in range(4):
            for j in range(4):
                U_full[i*5+j, r] = U1[i*4+j, r]
        for j in range(4):
            for k in range(4):
                V_full[j*6+k, r] = V1[j*4+k, r]
        for k in range(4):
            for i in range(4):
                W_full[k*4+i, r] = W1[k*4+i, r]
    
    # Block 1b: A1 (4x4, cols 0-3) * B1b (4x2, rows 0-3, cols 4-5) -> C (4x2, cols 4-5)
    # U2[i*4+j, r] -> U_full[i*5+j, r] for i,j in 0..3
    # V2[j*4+k, r] -> V_full[j*6+(4+k), r] for j in 0..3, k in 0..1
    # W2[k*4+i, r] -> W_full[(4+k)*4+i, r] for k in 0..1, i in 0..3
    
    offset2 = U1.shape[1]
    for r in range(U2.shape[1]):
        for i in range(4):
            for j in range(4):
                U_full[i*5+j, offset2+r] = U2[i*4+j, r]
        for j in range(4):
            for k in range(2):
                V_full[j*6+(4+k), offset2+r] = V2[j*4+k, r]
        for k in range(2):
            for i in range(4):
                W_full[(4+k)*4+i, offset2+r] = W2[k*4+i, r]
    
    # Block 2: A2 (4x1, col 4) * B2 (1x6, row 4) -> C (4x6, all cols)
    # U3[i*1+0, r] = U3[i, r] -> U_full[i*5+4, r] for i in 0..3
    # V3[0*6+k, r] = V3[k, r] -> V_full[4*6+k, r] for k in 0..5
    # W3[k*4+i, r] -> W_full[k*4+i, r] for k in 0..5, i in 0..3
    
    offset3 = U1.shape[1] + U2.shape[1]
    for r in range(U3.shape[1]):
        for i in range(4):
            U_full[i*5+4, offset3+r] = U3[i, r]
        for k in range(6):
            V_full[4*6+k, offset3+r] = V3[k, r]
        for k in range(6):
            for i in range(4):
                W_full[k*4+i, offset3+r] = W3[k*4+i, r]
    
    # Verify
    error, _ = _verify_decomposition(U_full, V_full, W_full, 4, 5, 6)
    
    if error < 1e-10:
        # Convert to lists
        U_list = [[0]*total_rank for _ in range(20)]
        V_list = [[0]*total_rank for _ in range(30)]
        W_list = [[0]*total_rank for _ in range(24)]
        
        for i in range(20):
            for r in range(total_rank):
                if U_full[i, r] != 0:
                    U_list[i][r] = int(U_full[i, r]) if U_full[i, r] == int(U_full[i, r]) else float(U_full[i, r])
        
        for i in range(30):
            for r in range(total_rank):
                if V_full[i, r] != 0:
                    V_list[i][r] = int(V_full[i, r]) if V_full[i, r] == int(V_full[i, r]) else float(V_full[i, r])
        
        for i in range(24):
            for r in range(total_rank):
                if W_full[i, r] != 0:
                    W_list[i][r] = int(W_full[i, r]) if W_full[i, r] == int(W_full[i, r]) else float(W_full[i, r])
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_school, V_school, W_school = _build_schoolbook(4, 5, 6)
    U_list = [[0]*120 for _ in range(20)]
    V_list = [[0]*120 for _ in range(30)]
    W_list = [[0]*120 for _ in range(24)]
    
    for i in range(20):
        for r in range(120):
            if U_school[i, r] != 0:
                U_list[i][r] = 1
    
    for i in range(30):
        for r in range(120):
            if V_school[i, r] != 0:
                V_list[i][r] = 1
    
    for i in range(24):
        for r in range(120):
            if W_school[i, r] != 0:
                W_list[i][r] = 1
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
