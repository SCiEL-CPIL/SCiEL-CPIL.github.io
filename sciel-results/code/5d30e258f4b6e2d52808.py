"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # DFT-based decomposition using cyclic group structure
    # We use the fact that for dimensions related to cyclic groups,
    # we can exploit Fourier basis to reduce rank
    
    # Strategy: Use DFT to create structured rank-1 terms
    # For each "frequency" in the cyclic group, we create a rank-1 term
    # that captures the convolution-like structure
    
    # The matrix multiplication tensor T_{ijk} = delta_{i,k} (for appropriate indexing)
    # can be decomposed using DFT basis vectors
    
    # For dimensions n=4, m=4, p=5, we'll use a combination of:
    # 1. DFT-based terms for the cyclic structure
    # 2. Fallback to schoolbook for remaining terms
    
    rank = n * m * p  # Start with schoolbook, will try to reduce
    
    # Try DFT-based approach
    # The idea: for each "mode" in the cyclic group Z_5 (since p=5),
    # we can create rank-1 terms that factor the multiplication
    
    # DFT matrix for p=5
    omega = np.exp(2j * np.pi / p)
    
    # Build DFT basis vectors
    # u_k[i] = omega^(i*k) for i in range(n)
    # v_k[j] = omega^(j*k) for j in range(m)  
    # w_k[k'] = omega^(k'*k) for k' in range(p)
    
    # Actually, let's use a more direct approach:
    # For each pair (a, b) where a in range(n), b in range(m),
    # we can create a rank-1 term using DFT in the k dimension
    
    # Better approach: Use the fact that matrix multiplication is a bilinear map
    # and decompose it using tensor rank algorithms
    
    # For now, let's try a structured decomposition:
    # Split into blocks and use Strassen-like ideas
    
    # Block decomposition:
    # A is 4x4, B is 4x5, C is 4x5
    # Split A into 2x2 blocks of 2x2 matrices
    # Split B into 2x2 blocks (last column block is 2x1)
    # This doesn't work cleanly for rectangular
    
    # Let's try a different approach: use the cyclic group Z_5 structure
    # Since p=5 is prime, we can use the multiplicative group
    
    # For each element g in Z_5^* = {1,2,3,4}, we can create rank-1 terms
    # that exploit the cyclic symmetry
    
    # Actually, the cleanest approach for this specific case:
    # Use the fact that 4 = 2^2 and 5 is prime
    # Decompose using 2x2 Strassen blocks where possible
    
    # Let's build a hybrid decomposition:
    # - Use Strassen for 2x2x2 subproblems
    # - Use schoolbook for the remaining 2x2x3 subproblem
    
    # A (4x4) = [A11 A12; A21 A22] where each Ai is 2x2
    # B (4x5) = [B11 B12; B21 B22] where B11,B21 are 2x2, B12,B22 are 2x3
    
    # C (4x5) = [C11 C12; C21 C22] where C11,C21 are 2x2, C12,C22 are 2x3
    
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x3)
    # C21 = A21*B11 + A22*B21 (2x2)
    # C22 = A21*B12 + A22*B22 (2x3)
    
    # For 2x2x2 blocks, use Strassen (rank 7)
    # For 2x2x3 blocks, use a modified Strassen or schoolbook (rank 9 or 12)
    
    # Let's use rank 7 for 2x2x2 and rank 9 for 2x2x3
    # Total rank = 4 * 7 + 4 * 9 = 28 + 36 = 64
    
    rank = 64
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen formulas for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2x3, we'll use a direct extension:
    # Split B into B_left (2x2) and B_right (2x1)
    # Use Strassen-like formulas
    
    # Actually, let's just use schoolbook for 2x2x3 to be safe
    # Rank for 2x2x3 = 2*2*3 = 12
    
    # Recalculate: 4 * 7 + 4 * 12 = 28 + 48 = 76
    # That's worse than schoolbook (80) only slightly better
    
    # Let's try a different split:
    # Split A into 1x4 and 3x4, B into 4x1 and 4x4
    # Not helpful
    
    # Better: Use the DFT approach directly
    # For the k-dimension (size 5), use DFT to diagonalize the cyclic structure
    
    # The multiplication tensor for 4x4x5 can be viewed as:
    # For each i in range(4), j in range(4), k in range(5):
    # T_{ijk} = 1 if the indices are compatible, else 0
    
    # Using DFT in the k-dimension:
    # w_k^{(freq)} = omega^(k*freq) for freq in range(5)
    
    # This gives us 5 frequency components, each contributing to the rank
    
    # Let's build the decomposition:
    # For each frequency f in range(5), and for each i, j:
    # Create a rank-1 term with:
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = omega^(k*f) for k in range(p)
    # W[k*n + i, col] = omega^(-k*f) for k in range(p)
    
    # But this doesn't directly give us the right structure
    
    # Let me think differently. The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki} (with appropriate indexing)
    
    # For our indexing:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} (if that's the right structure)
    
    # Actually, for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T_{i,j,k} = 1 (all ones in the appropriate subspace)
    
    # Wait, the tensor for matrix multiplication is:
    # T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # So T_{i,j,k} = 1 for all i,j,k in the appropriate ranges
    
    # The decomposition requires:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a constant tensor! We need to decompose the all-ones tensor of size 4*4*5
    
    # For an all-ones tensor of size a*b*c, the rank is max(a,b,c)
    # because we can use: U[i,j,r] = 1/r, V[j,k,r] = 1, W[k,i,r] = r
    # or similar constructions
    
    # Actually, for all-ones tensor of size a*b*c:
    # Rank = max(a,b,c)
    
    # So for 4*4*5, rank = 5
    
    # Let's verify: 
    # U[i*m+j, r] = 1 (for all i,j,r)
    # V[j*p+k, r] = 1 (for all j,k,r)  
    # W[k*n+i, r] = 1 (for all k,i,r)
    # Then sum_r U*V*W = sum_r 1*1*1 = rank
    
    # We need this to equal 1, so we need rank = 1? No, that doesn't work.
    
    # Let me reconsider. The tensor T_{ijk} for matrix multiplication is not all ones.
    # It's the structure tensor of the algebra.
    
    # For matrix multiplication of n×m by m×p matrices:
    # The output is n×p
    # T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # T_{i,j,k} = 1 (the tensor has 1s in all positions)
    
    # Yes! The matrix multiplication tensor for multiplying an n×m matrix by an m×p matrix
    # has T_{ijk} = 1 for all i,j,k
    
    # So we need to decompose the all-ones tensor of size 4×4×5
    
    # The rank of the all-ones tensor of size a×b×c is max(a,b,c)
    
    # Proof: 
    # Lower bound: The tensor has a 2D slice of rank max(a,b), so tensor rank >= max(a,b)
    # Upper bound: Use construction with max(a,b,c) terms
    
    # For 4×4×5, rank = 5
    
    # Construction:
    # Let r = 5 (max dimension)
    # U[i*m+j, s] = 1 for all i,j,s
    # V[j*p+k, s] = 1 for all j,k,s
    # W[k*n+i, s] = 1 for all k,i,s
    # Then sum_s U*V*W = 5, not 1
    
    # We need sum_s U*V*W = 1
    # So set U[i*m+j, s] = 1/5, V[j*p+k, s] = 1, W[k*n+i, s] = 1
    # Then sum_s (1/5)*1*1 = 5*(1/5) = 1 ✓
    
    # But 1/5 is not an integer or half-integer!
    
    # We need integer or half-integer entries.
    
    # Alternative: Use rank 10
    # U[i*m+j, s] = 1 for s in [0,10)
    # V[j*p+k, s] = 1 for s in [0,10)
    # W[k*n+i, s] = 1 for s in [0,5), 0 for s in [5,10)
    # Then sum_s U*V*W = 5, not 1
    
    # Hmm, this is tricky with the integer constraint.
    
    # Let's go back to the block decomposition approach which is known to work.
    
    # Use Strassen for 2x2x2 blocks (rank 7) and schoolbook for 2x2x3 (rank 12)
    # Total: 4*7 + 4*12 = 28 + 48 = 76
    
    # Or use a better decomposition for 2x2x3
    
    # For 2x2x3, we can potentially do better than 12
    # The rank of 2x2x3 matrix multiplication is 9 (known result)
    
    # So total rank = 4*7 + 4*9 = 28 + 36 = 64
    
    # Let's implement this.
    
    # First, let's define the Strassen decomposition for 2x2x2
    
    # Strassen M1-M7 for 2x2x2:
    # Input: A (2x2), B (2x2)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2x3, we need a rank-9 decomposition
    # This is more complex. Let's use a known construction or fall back to schoolbook.
    
    # For simplicity and correctness, let's use:
    # - Strassen (rank 7) for the four 2x2x2 blocks
    # - Schoolbook (rank 12) for the four 2x2x3 blocks
    # Total: 4*7 + 4*12 = 76
    
    # This is better than 80.
    
    rank = 76
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Define helper to add a rank-1 term
    def add_term(u_val, v_val, w_val, col_idx):
        nonlocal col
        U[:, col_idx] = 0
        V[:, col_idx] = 0
        W[:, col_idx] = 0
        # u_val, v_val, w_val are dictionaries mapping index to value
        for idx, val in u_val.items():
            U[idx, col_idx] = val
        for idx, val in v_val.items():
            V[idx, col_idx] = val
        for idx, val in w_val.items():
            W[idx, col_idx] = val
    
    # Strassen decomposition for 2x2x2 block
    # Block (bi, bj) where bi in [0,2), bj in [0,2)
    # A block: rows [bi*2, bi*2+2), cols [bj*2, bj*2+2)
    # B block: rows [bj*2, bj*2+2), cols [bk*2, bk*2+2)
    # C block: rows [bi*2, bi*2+2), cols [bk*2, bk*2+2)
    
    # For each 2x2x2 subproblem (bi, bj, bk) where bk in [0,2):
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # A indices: (bi*2 + i, bj*2 + j) for i,j in [0,2)
                # B indices: (bj*2 + j, bk*2 + k) for j,k in [0,2)
                # C indices: (bi*2 + i, bk*2 + k) for i,k in [0,2)
                
                # Strassen M1: (A00+A11)(B00+B11)
                # A00 = A[bi*2, bj*2], A11 = A[bi*2+1, bj*2+1]
                # B00 = B[bj*2, bk*2], B11 = B[bj*2+1, bk*2+1]
                
                # M1 term
                u1 = {bi*2*m + bj*2: 1, bi*2*m + bj*2 + 1: 1}
                v1 = {bj*2*p + bk*2: 1, bj*2*p + bk*2 + 1: 1}
                w1 = {bk*2*n + bi*2: 1, bk*2*n + bi*2 + 1: 1}
                add_term(u1, v1, w1, col)
                col += 1
                
                # M2: (A10+A11)B00
                # A10 = A[bi*2+1, bj*2], A11 = A[bi*2+1, bj*2+1]
                # B00 = B[bj*2, bk*2]
                u2 = {bi*2*m + bj*2 + 1: 1}  # A10 and A11 both have row bi*2+1
                # Actually A10 = A[bi*2+1, bj*2], A11 = A[bi*2+1, bj*2+1]
                # So u2 should have entries for both
                u2 = {bi*2*m + bj*2: 0, bi*2*m + bj*2 + 1: 1}  # This is wrong
                # Let me redo this properly
                
    # This is getting complex. Let me use a simpler approach.
    
    # Given the complexity and time constraints, let's just return the schoolbook decomposition
    # which is known to be correct.
    
    schoolbook_rank = n * m * p  # 80
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    U_list = [[int(U[i, j]) for j in range(schoolbook_rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) for j in range(schoolbook_rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) for j in range(schoolbook_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
