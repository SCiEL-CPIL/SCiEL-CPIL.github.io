"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import expm, logm, solve, inv
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    Uses Strassen's algorithm on 2x2x2 blocks to reduce the rank from 60 to 56.
    """
    n, m, p = DIMENSIONS
    
    # Start with the naive decomposition (rank = 60)
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Apply Strassen's algorithm to each 2x2x2 block
    # For a 2x2x2 block with local indices i' in {0,1}, j' in {0,1}, k' in {0,1}
    # We replace the 8 naive terms with 7 Strassen terms
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(g+h)
    # m7 = (a-c)(e+f)
    # where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    
    # C = [[m1+m4-m5+m7, m3+m5],
    #      [m2+m4,         m1-m2+m3+m6]]
    
    # For each 2x2x2 block (i0, j0, k0) where i0 in {0,1}, j0 in {0,1}, k0 in {0,1}
    # we can apply Strassen if we have at least 2x2x2 sub-blocks
    
    # Since n=3, m=4, p=5, we can fit:
    # - 1 full 2x2x2 block in the top-left corner (i=0,1; j=0,1; k=0,1)
    # - Additional partial blocks along the edges
    
    # Let's apply Strassen to all possible 2x2x2 blocks
    # We'll identify blocks that are fully contained in the matrix dimensions
    
    # For a 2x2x2 block starting at (i0, j0, k0):
    # i0 in range(n-1) = range(2)
    # j0 in range(m-1) = range(3)
    # k0 in range(p-1) = range(4)
    
    # But we can't apply Strassen to overlapping blocks independently
    # because they would interfere. We need to partition the matrix into
    # non-overlapping 2x2x2 blocks.
    
    # Let's partition:
    # - Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # - Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # - Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    # - Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    # - Block 5: i in {0,1}, j in {0,1}, k in {4}  -- only 1x2x1, can't use Strassen
    # - Block 6: i in {0,1}, j in {2,3}, k in {4}  -- only 1x2x1
    # - Block 7: i in {2}, j in {0,1}, k in {0,1}  -- only 1x2x2
    # - Block 8: i in {2}, j in {2,3}, k in {0,1}  -- only 1x2x2
    # - Block 9: i in {2}, j in {0,1}, k in {2,3}  -- only 1x2x2
    # - Block 10: i in {2}, j in {2,3}, k in {2,3} -- only 1x2x2
    # - Block 11: i in {2}, j in {0,1}, k in {4}   -- only 1x2x1
    # - Block 12: i in {2}, j in {2,3}, k in {4}   -- only 1x2x1
    
    # So we can apply Strassen to 4 blocks, each reducing rank from 8 to 7
    # Total reduction: 4 * 1 = 4
    # New rank: 60 - 4 = 56
    
    # Let's implement this by removing the naive terms for these blocks
    # and adding the Strassen terms
    
    # First, let's identify which naive terms are in the 4 Strassen blocks
    strassen_blocks = [
        (0, 0, 0),  # i0=0, j0=0, k0=0
        (0, 2, 0),  # i0=0, j0=2, k0=0
        (0, 0, 2),  # i0=0, j0=0, k0=2
        (0, 2, 2),  # i0=0, j0=2, k0=2
    ]
    
    # For each block, remove the 8 naive terms and add 7 Strassen terms
    # We'll do this by modifying the U, V, W matrices
    
    # Create a copy to work with
    U_new = np.zeros((n * m, n * m * p - 4))  # 60 - 4 = 56
    V_new = np.zeros((m * p, n * m * p - 4))
    W_new = np.zeros((p * n, n * m * p - 4))
    
    # Map old columns to new columns
    # We'll keep all naive terms except those in the Strassen blocks
    # and add the Strassen terms at the end
    
    col_new = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is in a Strassen block
                in_strassen = False
                for (i0, j0, k0) in strassen_blocks:
                    if (i in [i0, i0+1] and j in [j0, j0+1] and k in [k0, k0+1]):
                        in_strassen = True
                        break
                
                if not in_strassen:
                    # Keep this naive term
                    U_new[i * m + j, col_new] = 1
                    V_new[j * p + k, col_new] = 1
                    W_new[k * n + i, col_new] = 1
                    col_new += 1
    
    # Now add the Strassen terms
    for (i0, j0, k0) in strassen_blocks:
        # The 7 Strassen products
        # m1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # m2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        # m3 = A[i0,j0] * (B[j0+1,k0+1] - B[j0,k0+1])
        # m4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0+1])
        # m5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        # m6 = (A[i0+1,j0] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        # m7 = (A[i0,j0] - A[i0+1,j0]) * (B[j0,k0] + B[j0,k0+1])
        
        # C[i0,k0] += m1 + m4 - m5 + m7
        # C[i0,k0+1] += m3 + m5
        # C[i0+1,k0] += m2 + m4
        # C[i0+1,k0+1] += m1 - m2 + m3 + m6
        
        # m1
        U_new[i0 * m + j0, col_new] = 1
        U_new[(i0+1) * m + (j0+1), col_new] = 1
        V_new[j0 * p + k0, col_new] = 1
        V_new[(j0+1) * p + (k0+1), col_new] = 1
        W_new[k0 * n + i0, col_new] = 1
        W_new[(k0+1) * n + (i0+1), col_new] = 1
        col_new += 1
        
        # m2
        U_new[(i0+1) * m + j0, col_new] = 1
        U_new[(i0+1) * m + (j0+1), col_new] = 1
        V_new[j0 * p + k0, col_new] = 1
        W_new[k0 * n + (i0+1), col_new] = 1
        col_new += 1
        
        # m3
        U_new[i0 * m + j0, col_new] = 1
        V_new[(j0+1) * p + (k0+1), col_new] = 1
        V_new[j0 * p + (k0+1), col_new] = -1
        W_new[(k0+1) * n + i0, col_new] = 1
        col_new += 1
        
        # m4
        U_new[(i0+1) * m + (j0+1), col_new] = 1
        V_new[(j0+1) * p + k0, col_new] = 1
        V_new[(j0+1) * p + (k0+1), col_new] = -1
        W_new[k0 * n + (i0+1), col_new] = 1
        col_new += 1
        
        # m5
        U_new[i0 * m + j0, col_new] = 1
        U_new[i0 * m + (j0+1), col_new] = 1
        V_new[(j0+1) * p + (k0+1), col_new] = 1
        W_new[k0 * n + i0, col_new] = -1
        W_new[(k0+1) * n + i0, col_new] = 1
        col_new += 1
        
        # m6
        U_new[(i0+1) * m + j0, col_new] = 1
        U_new[(i0+1) * m + (j0+1), col_new] = -1
        V_new[(j0+1) * p + k0, col_new] = 1
        V_new[(j0+1) * p + (k0+1), col_new] = 1
        W_new[(k0+1) * n + (i0+1), col_new] = 1
        col_new += 1
        
        # m7
        U_new[i0 * m + j0, col_new] = 1
        U_new[(i0+1) * m + j0, col_new] = -1
        V_new[j0 * p + k0, col_new] = 1
        V_new[j0 * p + (k0+1), col_new] = 1
        W_new[k0 * n + i0, col_new] = 1
        col_new += 1
    
    # Trim the matrices to the actual rank used
    rank = col_new
    U_final = U_new[:, :rank]
    V_final = V_new[:, :rank]
    W_final = W_new[:, :rank]
    
    # Convert to lists for the required format
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
