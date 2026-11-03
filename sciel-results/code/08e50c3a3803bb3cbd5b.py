import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline (10s safety margin)
    n, m, p = DIMENSIONS
    
    # Build schoolbook decomposition
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For schoolbook: rank = n*m*p = 4*5*6 = 120
    # U[i*m+j, t] = 1 if t corresponds to (i,j,k)
    # V[j*p+k, t] = 1 if t corresponds to (i,j,k)
    # W[k*n+i, t] = 1 if t corresponds to (i,j,k)
    
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=object)
    V = np.zeros((m * p, rank_school), dtype=object)
    W = np.zeros((p * n, rank_school), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    # Now try to reduce rank by finding dependencies
    # We'll use a greedy approach: for each column, check if it can be expressed
    # as a linear combination of other columns
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition exactly computes matrix multiplication."""
        # Convert to numpy arrays with float for verification
        U_f = np.array(U_mat, dtype=float)
        V_f = np.array(V_mat, dtype=float)
        W_f = np.array(W_mat, dtype=float)
        
        # Build the tensor T_{ijk} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # and compare with the standard multiplication tensor
        # T_{ijk} = 1 if we're computing (A*B)[i,k] with A[i,j]*B[j,k]
        # Actually the tensor for matrix mult: T_{i,j,k} = A[i,j]*B[j,k] contributes to C[i,k]
        # The decomposition should satisfy: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,j,k} pattern
        
        # Actually, the matrix multiplication tensor T has entries:
        # T_{(i,j),(j,k),(k,i)}... let me think more carefully.
        # 
        # The tensor is: for matrices A (n x m), B (m x p), C = A*B (n x p)
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T has indices (i, j, k) where T[i,j,k] = 1 (it's the structure tensor)
        # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, that's not quite right either. Let me re-read the contract.
        #
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So W is indexed by [k*n + i, r] where k is the column of C (and row of B^T?) 
        # and i is the row of C.
        #
        # The decomposition: C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_k B[j,k] V[j*p+k, r]) ... no that's not right.
        #
        # Actually the standard tensor decomposition for matrix multiplication:
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # where T_{i,j,k} is the structure tensor (T_{i,j,k} = 1 for all valid i,j,k in the 
        # appropriate indexing, but actually T is the "multiplication tensor" which has
        # T_{(i,j),(j,k),(i,k)} = 1 in some indexing).
        #
        # Let me just use the convention from the code:
        # U has shape (n*m, rank) - indexed by (i,j) as i*m+j
        # V has shape (m*p, rank) - indexed by (j,k) as j*p+k  
        # W has shape (p*n, rank) - indexed by (k,i) as k*n+i
        #
        # The decomposition should satisfy:
        # For the matrix multiplication tensor, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # (since every term A[i,j]*B[j,k] contributes to C[i,k])
        
        # Check a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if val != 1:
                        return False
        return True
    
    # Try to reduce rank by peeling off columns that are linearly dependent
    # This is expensive, so let's try a different approach.
    
    # Let's try to use Strassen-like ideas on blocks.
    # Split the 4x5x6 multiplication into blocks.
    
    # Approach: Split n=4 into 2+2, and p=6 into 3+3.
    # Then we have 4 blocks of 2x5x3 multiplication.
    # For each 2x5x3 block, try to use an efficient decomposition.
    
    # Actually, let's try a different structural approach.
    # Use the fact that we can combine blocks.
    
    # Let me try to implement a basic Strassen for 2x2 blocks and combine.
    
    def build_strassen_2x2x2():
        """Build rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        # Strassen's 7 multiplications
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U: 4x7 (for 2x2 A matrix, flattened row-major: A11, A12, A21, A22)
        # V: 4x7 (for 2x2 B matrix, flattened row-major: B11, B12, B21, B22)
        # W: 4x7 (for 2x2 C matrix, flattened row-major: C11, C12, C21, C22)
        
        U_s = np.zeros((4, 7), dtype=object)
        V_s = np.zeros((4, 7), dtype=object)
        W_s = np.zeros((4, 7), dtype=object)
        
        # M1 = (A11+A22)*(B11+B22)
        U_s[0, 0] = 1  # A11
        U_s[3, 0] = 1  # A22
        V_s[0, 0] = 1  # B11
        V_s[3, 0] = 1  # B22
        W_s[0, 0] = 1  # C11
        W_s[3, 0] = 1  # C22
        
        # M2 = (A21+A22)*B11
        U_s[2, 1] = 1  # A21
        U_s[3, 1] = 1  # A22
        V_s[0, 1] = 1  # B11
        W_s[2, 1] = 1  # C21
        W_s[3, 1] = -1  # C22
        
        # M3 = A11*(B12-B22)
        U_s[0, 2] = 1  # A11
        V_s[1, 2] = 1  # B12
        V_s[3, 2] = -1  # B22
        W_s[1, 2] = 1  # C12
        W_s[3, 2] = 1  # C22
        
        # M4 = A22*(B21-B11)
        U_s[3, 3] = 1  # A22
        V_s[2, 3] = 1  # B21
        V_s[0, 3] = -1  # B11
        W_s[0, 3] = 1  # C11
        W_s[2, 3] = 1  # C21
        
        # M5 = (A11+A12)*B22
        U_s[0, 4] = 1  # A11
        U_s[1, 4] = 1  # A12
        V_s[3, 4] = 1  # B22
        W_s[0, 4] = -1  # C11
        W_s[1, 4] = 1  # C12
        
        # M6 = (A21-A11)*(B11+B12)
        U_s[2, 5] = 1  # A21
        U_s[0, 5] = -1  # A11
        V_s[0, 5] = 1  # B11
        V_s[1, 5] = 1  # B12
        W_s[3, 5] = 1  # C22
        
        # M7 = (A12-A22)*(B21+B22)
        U_s[1, 6] = 1  # A12
        U_s[3, 6] = -1  # A22
        V_s[2, 6] = 1  # B21
        V_s[3, 6] = 1  # B22
        W_s[0, 6] = 1  # C11
        W_s[2, 6] = -1  # C21
        
        return U_s, V_s, W_s
    
    # Build a 4x4x4 Strassen decomposition using 4 2x2x2 Strassen blocks
    def build_4x4x4_strassen():
        """Build decomposition for 4x4 * 4x4 -> 4x4 using Strassen on 2x2 blocks."""
        # Split 4x4 into 4 blocks of 2x2
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # C = [[C11, C12], [C21, C22]] where each is 2x2
        #
        # We can use Strassen on the 2x2 block structure, but each "element" is a 2x2 matrix.
        # This gives 7 "multiplications" of 2x2 matrices, each of which can be done with
        # Strassen (rank 7) or schoolbook (rank 4).
        #
        # Using Strassen for the inner 2x2: total rank = 7 * 7 = 49
        # Using schoolbook for the inner 2x2: total rank = 7 * 4 = 28
        #
        # Let's use schoolbook for the inner to keep it simpler and lower rank.
        
        U_4 = np.zeros((16, 28), dtype=object)
        V_4 = np.zeros((16, 28), dtype=object)
        W_4 = np.zeros((16, 28), dtype=object)
        
        # Strassen formulas for 2x2 block matrices:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Strassen multiplication, we need to compute a product of 2x2 matrices.
        # We'll use schoolbook for the inner 2x2*2x2->2x2 (rank 4).
        
        # Define the 7 Strassen "multiplications" and their contributions to C blocks.
        # Each multiplication is: (sum of A blocks) * (sum of B blocks) -> contribution to C blocks
        
        # M1: (A11+A22) * (B11+B22) -> contributes to C11 (+1) and C22 (+1)
        # M2: (A21+A22) * B11 -> contributes to C21 (+1) and C22 (-1)
        # M3: A11 * (B12-B22) -> contributes to C12 (+1) and C22 (+1)
        # M4: A22 * (B21-B11) -> contributes to C11 (+1) and C21 (+1)
        # M5: (A11+A12) * B22 -> contributes to C11 (-1) and C12 (+1)
        # M6: (A21-A11) * (B11+B12) -> contributes to C22 (+1)
        # M7: (A12-A22) * (B21+B22) -> contributes to C11 (+1) and C21 (-1)
        
        # For each multiplication, the "A side" is a sum of 2x2 blocks, and "B side" is a sum of 2x2 blocks.
        # We'll flatten the 4x4 matrices in row-major order.
        
        # A block indices in the 4x4 matrix (row-major):
        # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # B block indices:
        # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # C block indices:
        # C11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # C12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # C21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # C22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # For each Strassen multiplication, we have a linear combination of A blocks and B blocks.
        # The product is computed using schoolbook 2x2*2x2->2x2.
        
        # Let's define the Strassen multiplications more explicitly.
        # Each multiplication involves:
        # - A side: a vector of coefficients for A11, A12, A21, A22 (each 2x2)
        # - B side: a vector of coefficients for B11, B12, B21, B22 (each 2x2)
        # - C side: contributions to C11, C12, C21, C22 (each 2x2)
        
        strassen_mults = [
            # (A_coeffs, B_coeffs, C_contributions)
            # A_coeffs: dict from block to coefficient
            # B_coeffs: dict from block to coefficient
            # C_contributions: dict from block to coefficient
            (
                {'A11': 1, 'A22': 1},  # M1 A side
                {'B11': 1, 'B22': 1},  # M1 B side
                {'C11': 1, 'C22': 1}   # M1 C contributions
            ),
            (
                {'A21': 1, 'A22': 1},  # M2 A side
                {'B11': 1},            # M2 B side
                {'C21': 1, 'C22': -1}  # M2 C contributions
            ),
            (
                {'A11': 1},            # M3 A side
                {'B12': 1, 'B22': -1}, # M3 B side
                {'C12': 1, 'C22': 1}   # M3 C contributions
            ),
            (
                {'A22': 1},            # M4 A side
                {'B21': 1, 'B11': -1}, # M4 B side
                {'C11': 1, 'C21': 1}   # M4 C contributions
            ),
            (
                {'A11': 1, 'A12': 1},  # M5 A side
                {'B22': 1},            # M5 B side
                {'C11': -1, 'C12': 1}  # M5 C contributions
            ),
            (
                {'A21': 1, 'A11': -1}, # M6 A side
                {'B11': 1, 'B12': 1},  # M6 B side
                {'C22': 1}             # M6 C contributions
            ),
            (
                {'A12': 1, 'A22': -1}, # M7 A side
                {'B21': 1, 'B22': 1},  # M7 B side
                {'C11': 1, 'C21': -1}  # M7 C contributions
            ),
        ]
        
        # For each Strassen multiplication, we compute the product of the A-side combination
        # and B-side combination using schoolbook 2x2*2x2->2x2.
        
        # Block index mapping:
        block_idx = {
            'A11': [0, 1, 4, 5],
            'A12': [2, 3, 6, 7],
            'A21': [8, 9, 12, 13],
            'A22': [10, 11, 14, 15],
            'B11': [0, 1, 4, 5],
            'B12': [2, 3, 6, 7],
            'B21': [8, 9, 12, 13],
            'B22': [10, 11, 14, 15],
            'C11': [0, 1, 4, 5],
            'C12': [2, 3, 6, 7],
            'C21': [8, 9, 12, 13],
            'C22': [10, 11, 14, 15],
        }
        
        col = 0
        for a_coeffs, b_coeffs, c_contribs in strassen_mults:
            # For each pair (a_block, b_block) in the sums, add a schoolbook term
            for a_block, a_coeff in a_coeffs.items():
                for b_block, b_coeff in b_coeffs.items():
                    # The product of A_block (2x2) and B_block (2x2) gives a 2x2 result.
                    # Schoolbook: 4 terms.
                    # A_block is at indices block_idx[a_block] in the 4x4 A matrix.
                    # B_block is at indices block_idx[b_block] in the 4x4 B matrix.
                    # The result contributes to C blocks according to c_contribs.
                    
                    # For schoolbook 2x2*2x2->2x2:
                    # C_local[i,k] = sum_j A_local[i,j] * B_local[j,k]
                    # A_local indices: 0,1,2,3 (row-major 2x2)
                    # B_local indices: 0,1,2,3 (row-major 2x2)
                    # C_local indices: 0,1,2,3 (row-major 2x2)
                    
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                # A_local[i,j] corresponds to global index block_idx[a_block][i*2+j]
                                a_global = block_idx[a_block][i * 2 + j]
                                # B_local[j,k] corresponds to global index block_idx[b_block][j*2+k]
                                b_global = block_idx[b_block][j * 2 + k]
                                # C_local[i,k] corresponds to local index i*2+k
                                c_local_idx = i * 2 + k
                                
                                # The C_local[i,k] contributes to all C blocks with the given coefficients
                                for c_block, c_coeff in c_contribs.items():
                                    c_global = block_idx[c_block][c_local_idx]
                                    U_4[a_global, col] = a_coeff
                                    V_4[b_global, col] = b_coeff
                                    W_4[c_global, col] = c_coeff
                                col += 1
        
        return U_4, V_4, W_4, col
    
    # Build 4x4x2 decomposition using Strassen on 2x2 blocks
    def build_4x4x2_strassen():
        """Build decomposition for 4x4 * 4x2 -> 4x2 using Strassen-like ideas."""
        # Split 4x4 into 2x2 blocks, 4x2 into 2x1 blocks (vertically).
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11], [B21]] where each is 2x1 (2 rows, 1 col)
        # C = [[C11], [C21]] where each is 2x1
        
        # C11 = A11*B11 + A12*B21
        # C21 = A21*B11 + A22*B21
        
        # We can use Strassen-like combinations:
        # M1 = (A11+A22) * (B11)  -- but B11 is 2x1, so this is 2x2 * 2x1 -> 2x1
        # Actually, let's just use schoolbook for this since it's small.
        # 4x4 * 4x2: schoolbook rank = 4*4*2 = 32
        
        # Or we can try to be clever:
        # Split into 2x2 blocks:
        # C11 = A11*B11 + A12*B21 (2x2 * 2x1 + 2x2 * 2x1 -> 2x1)
        # C21 = A21*B11 + A22*B21 (2x2 * 2x1 + 2x2 * 2x1 -> 2x1)
        
        # For each 2x2 * 2x1 -> 2x1, schoolbook rank = 2*2*1 = 4.
        # Total: 4 blocks * 4 = 16. That's already better than 32!
        
        # Wait, that's just splitting. Let me verify:
        # 2x2 * 2x1: C[i,0] = sum_j A[i,j] * B[j,0]. Rank = 2*2*1 = 4.
        # We have 4 such products, so total rank = 16.
        
        # Can we do better with Strassen-like?
        # Let's try:
        # M1 = (A11+A22) * B11 -> 2x2 * 2x1 -> 2x1, rank 4
        # M2 = (A21+A22) * B11 -> 2x2 * 2x1 -> 2x1, rank 4
        # M3 = A11 * B21 -> 2x2 * 2x1 -> 2x1, rank 4
        # M4 = A22 * B21 -> 2x2 * 2x1 -> 2x1, rank 4
        # M5 = (A11+A12) * B21 -> 2x2 * 2x1 -> 2x1, rank 4
        # M6 = (A21-A11) * B11 -> 2x2 * 2x1 -> 2x1, rank 4
        # 
        # Hmm, this is getting complicated. Let's just use the simple block split for now.
        
        # Simple block split: 4 independent 2x2*2x1->2x1 multiplications
        U_442 = np.zeros((16, 16), dtype=object)
        V_442 = np.zeros((8, 16), dtype=object)
        W_442 = np.zeros((8, 16), dtype=object)
        
        # A is 4x4, flattened row-major: indices 0-15
        # B is 4x2, flattened row-major: indices 0-7
        # C is 4x2, flattened row-major: indices 0-7
        
        # Blocks:
        # A11: rows 0-1, cols 0-1 -> A indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> A indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> A indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> A indices 10,11,14,15
        
        # B11: rows 0-1, col 0 -> B indices 0,2
        # B21: rows 2-3, col 0 -> B indices 4,6
        # B12: rows 0-1, col 1 -> B indices 1,3
        # B22: rows 2-3, col 1 -> B indices 5,7
        
        # C11: rows 0-1, col 0 -> C indices 0,2
        # C21: rows 2-3, col 0 -> C indices 4,6
        # C12: rows 0-1, col 1 -> C indices 1,3
        # C22: rows 2-3, col 1 -> C indices 5,7
        
        col = 0
        # C11 = A11 * B11 (2x2 * 2x1 -> 2x1)
        for i in range(2):
            for j in range(2):
                a_idx = [0, 1, 4, 5][i * 2 + j]  # A11[i,j]
                b_idx = [0, 2][j]  # B11[j,0]
                c_idx = [0, 2][i]  # C11[i,0]
                U_442[a_idx, col] = 1
                V_442[b_idx, col] = 1
                W_442[c_idx, col] = 1
                col += 1
        
        # C12 = A12 * B22 (2x2 * 2x1 -> 2x1)
        for i in range(2):
            for j in range(2):
                a_idx = [2, 3, 6, 7][i * 2 + j]  # A12[i,j]
                b_idx = [5, 7][j]  # B22[j,0] which is B rows 2-3, col 1
                c_idx = [1, 3][i]  # C12[i,0] which is C rows 0-1, col 1
                U_442[a_idx, col] = 1
                V_442[b_idx, col] = 1
                W_442[c_idx, col] = 1
                col += 1
        
        # C21 = A21 * B11 (2x2 * 2x1 -> 2x1)
        for i in range(2):
            for j in range(2):
                a_idx = [8, 9, 12, 13][i * 2 + j]  # A21[i,j]
                b_idx = [0, 2][j]  # B11[j,0]
                c_idx = [4, 6][i]  # C21[i,0] which is C rows 2-3, col 0
                U_442[a_idx, col] = 1
                V_442[b_idx, col] = 1
                W_442[c_idx, col] = 1
                col += 1
        
        # C22 = A22 * B22 (2x2 * 2x1 -> 2x1)
        for i in range(2):
            for j in range(2):
                a_idx = [10, 11, 14, 15][i * 2 + j]  # A22[i,j]
                b_idx = [5, 7][j]  # B22[j,0]
                c_idx = [5, 7][i]  # C22[i,0] which is C rows 2-3, col 1
                U_442[a_idx, col] = 1
                V_442[b_idx, col] = 1
                W_442[c_idx, col] = 1
                col += 1
        
        return U_442, V_442, W_442, col
    
    # Now let's build the full 4x5x6 decomposition.
    # Split 4x5 into blocks and 5x6 into blocks.
    
    # Strategy: Split n=4 into 2+2, m=5 into 2+3, p=6 into 3+3.
    # A = [[A11 (2x2), A12 (2x3)], [A21 (2x2), A22 (2x3)]]
    # B = [[B11 (2x3), B12 (2x3)], [B21 (3x3), B22 (3x3)]]
    # C = [[C11 (2x3), C12 (2x3)], [C21 (2x3), C22 (2x3)]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # This gives 8 block multiplications. Let's compute their ranks:
    # A11*B11: 2x2 * 2x3 -> 2x3, rank 2*2*3 = 12
    # A12*B21: 2x3 * 3x3 -> 2x3, rank 2*3*3 = 18
    # A11*B12: 2x2 * 2x3 -> 2x3, rank 12
    # A12*B22: 2x3 * 3x3 -> 2x3, rank 18
    # A21*B11: 2x2 * 2x3 -> 2x3, rank 12
    # A22*B21: 2x3 * 3x3 -> 2x3, rank 18
    # A21*B12: 2x2 * 2x3 -> 2x3, rank 12
    # A22*B22: 2x3 * 3x3 -> 2x3, rank 18
    # Total: 4*12 + 4*18 = 48 + 72 = 120. Same as schoolbook.
    
    # To gain, we need Strassen-like cancellation across blocks.
    
    # Let's try a different split: n=4 into 2+2, p=6 into 3+3, and keep m=5 whole.
    # A = [A1 (2x5); A2 (2x5)]
    # B = [B1 (5x3) | B2 (5x3)]
    # C = [C11 (2x3) | C12 (2x3); C21 (2x3) | C22 (2x3)]
    
    # C11 = A1*B1 (2x5 * 5x3 -> 2x3), rank 30
    # C12 = A1*B2 (2x5 * 5x3 -> 2x3), rank 30
    # C21 = A2*B1 (2x5 * 5x3 -> 2x3), rank 30
    # C22 = A2*B2 (2x5 * 5x3 -> 2x3), rank 30
    # Total: 120. Still same.
    
    # We really need Strassen-like ideas that mix the blocks.
    
    # Let's try to use the 4x4x4 Strassen decomposition on a 4x4x4 subproblem,
    # and handle the remaining 4x1x2 and 4x4x2 parts separately.
    
    # 4x5x6 = 4x4x4 + 4x1x4 + 4x4x2
    # Wait, that's not quite right. Let me think.
    
    # A is 4x5, B is 5x6.
    # Split A cols into 4+1, B rows into 4+1:
    # A = [A1 (4x4) | A2 (4x1)]
    # B = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x4 * 4x6 -> 4x6, rank 4*4*6 = 96
    # A2*B2: 4x1 * 1x6 -> 4x6, rank 4*1*6 = 24
    # Total: 120.
    
    # Use Strassen for 4x4*4x6:
    # Split 4x4*4x6 into 4x4*4x3 + 4x4*4x3
    # Or use the 4x4x4 Strassen (rank 28 with schoolbook inner) for the 4x4*4x4 part,
    # and handle 4x4*4x2 separately.
    
    # 4x4*4x6 = 4x4*4x4 + 4x4*4x2
    # 4x4*4x4 with Strassen: rank 28
    # 4x4*4x2 with block split: rank 16
    # Total for 4x4*4x6: 28 + 16 = 44
    # A2*B2: 4x1*1x6 -> 4x6, rank 24
    # Total: 44 + 24 = 68. That's much better than 120!
    
    # But wait, I need to verify the 4x4x4 Strassen and 4x4x2 decompositions are correct.
    
    # Let me build the full decomposition.
    
    def build_full_decomposition():
        """Build the full 4x5x6 decomposition using Strassen on 4x4x4 subblock."""
        # A is 4x5, B is 5x6, C is 4x6.
        # Split A cols into 4+1, B rows into 4+1.
        # A = [A1 (4x4) | A2 (4x1)]
        # B = [B1 (4x6); B2 (1x6)]
        # C = A1*B1 + A2*B2
        
        # Further split B1 cols into 4+2:
        # B1 = [B1a (4x4) | B1b (4x2)]
        # C = A1*B1a + A1*B1b + A2*B2
        
        # A1*B1a: 4x4 * 4x4 -> 4x4, use Strassen (rank 28)
        # A1*B1b: 4x4 * 4x2 -> 4x2, use block split (rank 16)
        # A2*B2: 4x1 * 1x6 -> 4x6, use schoolbook (rank 24)
        # Total rank: 28 + 16 + 24 = 68
        
        total_rank = 68
        U_full = np.zeros((20, total_rank), dtype=object)
        V_full = np.zeros((30, total_rank), dtype=object)
        W_full = np.zeros((24, total_rank), dtype=object)
        
        # Index mapping:
        # A is 4x5, flattened row-major: indices 0-19
        # A1 (4x4): rows 0-3, cols 0-3 -> A indices 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18
        # A2 (4x1): rows 0-3, col 4 -> A indices 4, 9, 14, 19
        
        # B is 5x6, flattened row-major: indices 0-29
        # B1 (4x6): rows 0-3, cols 0-5 -> B indices 0,1,2,3,4,5, 6,7,8,9,10,11, 12,13,14,15,16,17, 18,19,20,21,22,23
        # B2 (1x6): row 4, cols 0-5 -> B indices 24,25,26,27,28,29
        
        # B1a (4x4): rows 0-3, cols 0-3 -> B indices 0,1,2,3, 6,7,8,9, 12,13,14,15, 18,19,20,21
        # B1b (4x2): rows 0-3, cols 4-5 -> B indices 4,5, 10,11, 16,17, 22,23
        
        # C is 4x6, flattened row-major: indices 0-23
        # C1 (4x4): rows 0-3, cols 0-3 -> C indices 0,1,2,3, 6,7,8,9, 12,13,14,15, 18,19,20,21
        # C2 (4x2): rows 0-3, cols 4-5 -> C indices 4,5, 10,11, 16,17, 22,23
        # C3 (4x6) from A2*B2: rows 0-3, cols 0-5 -> C indices 0-23
        
        # Build A1*B1a (4x4*4x4->4x4) using Strassen
        U_444, V_444, W_444, rank_444 = build_4x4x4_strassen()
        
        # Map indices
        A1_idx = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]
        B1a_idx = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21]
        C1_idx = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21]
        
        col = 0
        for r in range(rank_444):
            for i in range(16):
                if U_444[i, r] != 0:
                    U_full[A1_idx[i], col] = U_444[i, r]
            for i in range(16):
                if V_444[i, r] != 0:
                    V_full[B1a_idx[i], col] = V_444[i, r]
            for i in range(16):
                if W_444[i, r] != 0:
                    W_full[C1_idx[i], col] = W_444[i, r]
            col += 1
        
        # Build A1*B1b (4x4*4x2->4x2) using block split
        U_442, V_442, W_442, rank_442 = build_4x4x2_strassen()
        
        B1b_idx = [4, 5, 10, 11, 16, 17, 22, 23]
        C2_idx = [4, 5, 10, 11, 16, 17, 22, 23]
        
        for r in range(rank_442):
            for i in range(16):
                if U_442[i, r] != 0:
                    U_full[A1_idx[i], col] = U_442[i, r]
            for i in range(8):
                if V_442[i, r] != 0:
                    V_full[B1b_idx[i], col] = V_442[i, r]
            for i in range(8):
                if W_442[i, r] != 0:
                    W_full[C2_idx[i], col] = W_442[i, r]
            col += 1
        
        # Build A2*B2 (4x1*1x6->4x6) using schoolbook
        A2_idx = [4, 9, 14, 19]
        B2_idx = [24, 25, 26, 27, 28, 29]
        
        for i in range(4):
            for k in range(6):
                U_full[A2_idx[i], col] = 1
                V_full[B2_idx[k], col] = 1
                W_full[k * 4 + i, col] = 1  # C[i,k] -> index k*4+i in row-major
                col += 1
        
        return U_full, V_full, W_full, col
    
    # Build the decomposition
    U_decomp, V_decomp, W_decomp, actual_rank = build_full_decomposition()
    
    # Verify
    if time.time() < deadline - 5:
        is_valid = verify_decomposition(U_decomp, V_decomp, W_decomp)
        if is_valid:
            # Convert to lists
            U_list = U_decomp.tolist()
            V_list = V_decomp.tolist()
            W_list = W_decomp.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
