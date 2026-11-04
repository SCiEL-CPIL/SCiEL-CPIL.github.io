import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7 with Strassen blocks
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    def find_rank_by_dependency_search():
        """
        Find a lower-rank decomposition by exploiting linear dependencies.
        
        Strategy: The tensor T has structure T[i*m+j, j*p+k, k*n+i] = 1.
        We can view this as: for each output position (i,k), the coefficient of 
        A[i,j]*B[j,k] is 1.
        
        For n=2, we have outputs C[0,0..6] and C[1,0..6] (14 outputs total).
        Each output is a linear combination of the 4 A-entries and 4 B-entries.
        
        Key insight: We can try to find a set of "multiplication patterns" 
        (rank-1 tensors) whose linear span contains T.
        
        Approach: Enumerate possible rank-1 patterns and find a minimal set 
        that spans T.
        
        A rank-1 tensor is u ⊗ v ⊗ w where u has support on A indices,
        v on B indices, w on C indices.
        
        For our problem:
        - u is a vector in R^(n*m) = R^8
        - v is a vector in R^(m*p) = R^28  
        - w is a vector in R^(p*n) = R^14
        
        The rank-1 tensor u⊗v⊗w has entry [a,b,c] = u[a]*v[b]*w[c].
        
        We need sum_r u_r ⊗ v_r ⊗ w_r = T.
        
        Let's try to find a sparse set of such patterns.
        """
        # First, get the rank-50 as a starting point
        U50, V50, W50 = get_rank50()
        best_rank = 50
        best_U, best_V, best_W = U50, V50, W50
        
        # Try to find linear dependencies among the columns
        # For each subset of columns, check if they can be replaced by fewer columns
        
        # Alternative approach: Use the fact that T is sparse and structured.
        # Let's try a greedy approach to find a minimal set of rank-1 terms.
        
        # Define candidate rank-1 patterns
        # For 2x4x4x7, the natural patterns involve:
        # - Single A[i,j] entries
        # - Sums/differences of A entries
        # - Single B[j,k] entries  
        # - Sums/differences of B entries
        
        # Let's try to construct a decomposition using Winograd-like transforms
        
        # For 1x4 x 4x4 Winograd (6 mults):
        # A = [a0,a1,a2,a3], B = 4x4
        # Transforms:
        #   a' = [a0+a2, a1+a3, a0-a2, a1-a3, a0+a1, a2+a3]
        #   b'_k = [b0k+b2k, b1k+b3k, b0k-b2k, b1k-b3k, b0k+b1k, b2k+b3k]
        #   m_k = a'_j * b'_jk for j=0..5
        #   C_k = (m_0k + m_1k + m_2k + m_3k + m_4k + m_5k) / 2
        
        # For 1x4 x 4x7, use two overlapping blocks: cols 0-3 and 3-6
        # Block 1 (cols 0-3): 6 mults
        # Block 2 (cols 3-6): 6 mults
        # Total: 12 mults per row
        
        # For 2 rows: 24 mults total
        
        # Let's construct this explicitly
        rank = 24
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        # Winograd transforms for A (length 4)
        # a' = [a0+a2, a1+a3, a0-a2, a1-a3, a0+a1, a2+a3]
        A_transforms = [
            [1, 0, 1, 0],  # a0+a2
            [0, 1, 0, 1],  # a1+a3
            [1, 0, -1, 0], # a0-a2
            [0, 1, 0, -1], # a1-a3
            [1, 1, 0, 0],  # a0+a1
            [0, 0, 1, 1],  # a2+a3
        ]
        
        # Winograd transforms for B column (length 4)
        # b' = [b0+b2, b1+b3, b0-b2, b1-b3, b0+b1, b2+b3]
        B_transforms = [
            [1, 0, 1, 0],  # b0+b2
            [0, 1, 0, 1],  # b1+b3
            [1, 0, -1, 0], # b0-b2
            [0, 1, 0, -1], # b1-b3
            [1, 1, 0, 0],  # b0+b1
            [0, 0, 1, 1],  # b2+b3
        ]
        
        # For each row i (0,1) and each block (0: cols 0-3, 1: cols 3-6)
        for i in range(n):
            for block in range(2):
                # Block 0: columns 0,1,2,3
                # Block 1: columns 3,4,5,6
                col_offset = block * 3
                # Columns in this block: col_offset, col_offset+1, col_offset+2, col_offset+3
                # But for block 1, col_offset=3, so cols 3,4,5,6
                
                for t in range(6):
                    # A transform t
                    for j in range(4):
                        U[i * m + j, col] += A_transforms[t][j]
                    
                    # B transform t, for each column in the block
                    for b_col in range(4):
                        actual_col = col_offset + b_col
                        if actual_col < p:  # Only if column exists
                            for j in range(4):
                                V[j * p + actual_col, col] += B_transforms[t][j]
                    
                    # Output: for each column in the block
                    for b_col in range(4):
                        actual_col = col_offset + b_col
                        if actual_col < p:
                            W[actual_col * n + i, col] += 0.5
                        else:
                            # Column doesn't exist, skip
                            pass
                    
                    col += 1
        
        # Verify and trim if needed
        actual_rank = col
        U = U[:, :actual_rank]
        V = V[:, :actual_rank]
        W = W[:, :actual_rank]
        
        if verify_decomp(U, V, W):
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_U, best_V, best_W = U, V, W
        
        return best_U, best_V, best_W, best_rank

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a better decomposition
    if time.time() < deadline - 30:
        U_best, V_best, W_best, best_rank = find_rank_by_dependency_search()
        
        if verify_decomp(U_best, V_best, W_best) and best_rank < 50:
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}

    # Return the rank-50 decomposition
    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
# EVOLVE-BLOCK-END
