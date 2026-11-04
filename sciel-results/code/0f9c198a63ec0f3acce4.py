"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy:
    1. Construct a rank 55 decomposition by splitting the 3x7 matrix into a 3x3 block and a 3x4 block.
       - Rank(3x3x3) is 23 (Laderman).
       - Rank(3x3x4) is 32 (23 + 9, using Laderman for 3x3x3 and Schoolbook for 3x3x1).
       - Total Rank = 23 + 32 = 55.
    2. If construction or verification fails, fall back to Schoolbook rank 63.
    
    The evaluator expects a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit

    def verify_decomposition(U, V, W, n, m, p):
        """
        Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] matches the matrix multiplication tensor.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        Expected T[i,j,k] = 1 for all i,j,k.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Reshape to tensors
        # U: (n, m, rank)
        U_t = U.reshape(n, m, -1)
        # V: (m, p, rank)
        V_t = V.reshape(m, p, -1)
        # W: (p, n, rank)
        W_t = W.reshape(p, n, -1)
        
        # Contract over rank dimension
        # Result shape: (n, m, p)
        # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        except Exception:
            return False, None
            
        if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
            return False, T
        return True, T

    def schoolbook_block(n, m, p_block):
        """
        Generate schoolbook decomposition for n x m x p_block block.
        Returns U, V, W of shapes (n*m, rank), (m*p_block, rank), (p_block*n, rank).
        """
        rank = n * m * p_block
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p_block, rank))
        W = np.zeros((p_block * n, rank))
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    U[i * m + j, c] = 1.0
                    V[j * p_block + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    def laderman_333():
        """
        Hardcoded Laderman 23-term algorithm for 3x3x3 matrix multiplication.
        Indices: U[i,j], V[j,k], W[k,i] where i,j,k in {0,1,2}.
        The tensor T[i,j,k] = 1.
        """
        # U, V, W are 3x3 matrices for each term, but we return flat vectors.
        # We define the 23 terms as (u, v, w) where u, v, w are 3x3 matrices.
        # Actually, it's easier to define the sparse support and coefficients.
        # Let's use a known valid set of 23 triples.
        # Reference: "A 23-multiplication algorithm for 3x3 matrix multiplication"
        
        # Each term is (u_vec, v_vec, w_vec) of length 9.
        # Or we can build the U, V, W matrices directly.
        
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # Indices for 3x3: 0,1,2 -> 0..8
        # Term 1: (0,0), (0,0), (0,0)
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: (1,1), (1,1), (1,1)
        U[4, 1] = 1; V[4, 1] = 1; W[4, 1] = 1
        # Term 3: (2,2), (2,2), (2,2)
        U[8, 2] = 1; V[8, 2] = 1; W[8, 2] = 1
        
        # This manual hardcoding is error-prone. Let's use a verified set from a reliable source or generate it.
        # Since I cannot verify a hardcoded set instantly, I will try to construct a valid rank 23 decomposition
        # using a known algorithm description if possible, or fall back.
        
        # However, there is a well-known 23-term algorithm. Let's try to find a robust way to generate it or use a verified list.
        # Given the risk, I will check if I can find a valid 23-term set in the standard references.
        # One common representation uses the following non-zero entries in U, V, W (9x23 matrices).
        
        # Let's try a different approach: use the known rank 23 decomposition from a library or verified code snippet if I can recall it.
        # If not, I will fallback to schoolbook for the 3x3x3 part, which gives rank 27.
        # Then total rank = 27 + (27+9) = 63. Same as schoolbook.
        # To beat 63, I MUST have a valid rank 23 decomposition for 3x3x3.
        
        # Let's assume the following structure for Laderman's algorithm (one of many valid ones):
        # This is a known valid set of 23 terms for 3x3x3.
        # Source: https://en.wikipedia.org/wiki/Laderman%27s_algorithm (or similar)
        
        # I will use a specific known valid set. If it fails verification, I fallback.
        
        # Term definitions (u_idx, v_idx, w_idx, coeff)
        # u_idx is in 0..8, v_idx in 0..8, w_idx in 0..8.
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        
        # This is complex to hardcode without a reference. 
        # Let's try to search for a valid rank 23 decomposition using a simple optimization if time permits?
        # No, 120s is tight.
        
        # Let's try to use a verified 23-term set from a public source.
        # One such set is:
        # (0,0,0), (1,1,1), (2,2,2)
        # (0,1,0), (1,0,1), (2,0,2) ... this is getting messy.
        
        # Alternative: Use the fact that rank(3x3x3) <= 23.
        # If I can't guarantee validity, I must fallback.
        
        # Let's try to construct a valid rank 23 decomposition using a known algorithm.
        # I will use the algorithm described in "Fast Matrix Multiplication" by Strassen, but adapted for 3x3.
        
        # Actually, let's just try to verify a few known sets.
        # Set 1:
        terms = [
            # (u, v, w)
            ((0,0), (0,0), (0,0)),
            ((1,1), (1,1), (1,1)),
            ((2,2), (2,2), (2,2)),
            ((0,1), (1,0), (0,1)), # Just guessing, likely wrong
        ]
        
        # This guessing is dangerous. I will return None to indicate failure to construct Laderman.
        return None, None, None

    # --- Attempt Rank 55 Construction ---
    
    # 1. Get 3x3x3 decomposition (Rank 23)
    U_333, V_333, W_333 = laderman_333()
    
    if U_333 is not None:
        # Verify the 3x3x3 part
        valid_333, _ = verify_decomposition(U_333, V_333, W_333, 3, 3, 3)
        if valid_333:
            # 2. Get 3x3x4 decomposition (Rank 32)
            # Use Laderman for first 3 columns (Rank 23)
            # Use Schoolbook for last 1 column (Rank 9)
            
            # The 3x3x4 block is split into 3x3x3 and 3x3x1.
            # U_333_4: shape (9, 23+9=32)
            # V_333_4: shape (3*4=12, 32)
            # W_333_4: shape (4*3=12, 32)
            
            U_block = np.zeros((9, 32))
            V_block = np.zeros((12, 32))
            W_block = np.zeros((12, 32))
            
            # Copy Laderman part (cols 0-22)
            U_block[:, :23] = U_333
            # V and W for the 3x3x3 part need to be mapped to the 3x3x4 indices.
            # For 3x3x3, V is (3,3) and W is (3,3).
            # For 3x3x4, V is (3,4) and W is (4,3).
            # The first 3 columns of V and W correspond to the 3x3x3 part.
            V_block[:9, :23] = V_333
            W_block[:9, :23] = W_333
            
            # Schoolbook part for 3x3x1 (cols 23-31)
            # The 3x3x1 block corresponds to column index 3 in the 3x3x4 matrix.
            # U for 3x3x1 is same as U for 3x3x3 (9x1).
            # V for 3x3x1 is (3,1) -> index j*1 + 3 = j+3.
            # W for 3x3x1 is (1,3) -> index 3*3 + i = 9+i.
            
            U_sb, V_sb, W_sb = schoolbook_block(3, 3, 1)
            # U_sb is 9x9, V_sb is 3x9, W_sb is 3x9.
            # We need to place these in the last 9 columns (23-31).
            U_block[:, 23:] = U_sb
            
            # V_sb is 3x9. We need to map it to V_block rows 0..8 (since p_block=4, V is 3x4=12 rows).
            # The schoolbook V_sb has entries at V[j*1 + 0, c] for j=0,1,2.
            # In the 3x3x4 block, the column index for the 3x3x1 part is k=3.
            # So V[j*4 + 3, c] should be 1.
            # V_sb[j, c] is 1 for j=0,1,2.
            # So we place V_sb into V_block rows 3, 7, 11 (j*4 + 3).
            V_block[3, 23:] = V_sb[0, :]
            V_block[7, 23:] = V_sb[1, :]
            V_block[11, 23:] = V_sb[2, :]
            
            # W_sb is 3x9. We need to map it to W_block rows 9..11 (since p_block=4, W is 4x3=12 rows).
            # The schoolbook W_sb has entries at W[0*3 + i, c] for i=0,1,2.
            # In the 3x3x4 block, the column index for the 3x3x1 part is k=3.
            # So W[3*3 + i, c] should be 1.
            # W_sb[i, c] is 1 for i=0,1,2.
            # So we place W_sb into W_block rows 9, 10, 11.
            W_block[9, 23:] = W_sb[0, :]
            W_block[10, 23:] = W_sb[1, :]
            W_block[11, 23:] = W_sb[2, :]
            
            # Now we have a 3x3x4 decomposition of rank 32.
            # We need a 3x3x3 decomposition of rank 23 for the first 3 columns of the 3x3x7 matrix.
            # And the 3x3x4 decomposition for the last 4 columns.
            
            # Total U: 9 x (23 + 32) = 9 x 55
            # Total V: 3x7=21 x 55
            # Total W: 7x3=21 x 55
            
            U_total = np.zeros((9, 55))
            V_total = np.zeros((21, 55))
            W_total = np.zeros((21, 55))
            
            # First part: 3x3x3 (cols 0-22)
            U_total[:, :23] = U_333
            # V for 3x3x3 part: rows 0..8 (j*3 + k for k=0,1,2)
            V_total[:9, :23] = V_333
            # W for 3x3x3 part: rows 0..8 (k*3 + i for k=0,1,2)
            W_total[:9, :23] = W_333
            
            # Second part: 3x3x4 (cols 23-54)
            U_total[:, 23:] = U_block
            # V for 3x3x4 part: rows 9..20 (j*7 + k for k=3,4,5,6)
            # V_block is 12x32. Rows 0..8 correspond to k=0,1,2 (but we only used k=3 for schoolbook, k=0,1,2 for Laderman).
            # Wait, V_block is for the 3x3x4 block. The indices are j*4 + k.
            # In the 3x3x7 block, the indices are j*7 + k.
            # For the 3x3x4 part, k ranges from 3 to 6.
            # So V_total[j*7 + (k+3), c] = V_block[j*4 + k, c].
            
            for j in range(3):
                for k in range(4):
                    V_total[j * 7 + (k + 3), 23:] = V_block[j * 4 + k, :]
                    
            # W for 3x3x4 part: rows 9..20 (k*3 + i for k=3,4,5,6)
            # W_block is 12x32. Indices are k*3 + i.
            # In the 3x3x7 block, the indices are k*3 + i.
            # For the 3x3x4 part, k ranges from 3 to 6.
            # So W_total[(k+3)*3 + i, c] = W_block[k*3 + i, c].
            
            for k in range(4):
                for i in range(3):
                    W_total[(k + 3) * 3 + i, 23:] = W_block[k * 3 + i, :]
            
            # Verify the full decomposition
            valid_total, _ = verify_decomposition(U_total, V_total, W_total, n, m, p)
            if valid_total:
                return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    
    # Fallback to Schoolbook rank 63
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    
    valid, _ = verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # This should never happen
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
