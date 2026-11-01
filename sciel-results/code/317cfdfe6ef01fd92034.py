import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 34
    rank = 34
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff: coefficient for this term in the linear combination
    
    col = 0
    
    # Block 1: 2x4x2 (k=0,1)
    # Split j=0,1 and j=2,3
    # Sub-block 1.1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Sub-block 1.2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    
    # Strassen for 2x2x2
    def add_strassen(U, V, W, col, i_off, j_off, k_off):
        # i_off: offset for i in A and C
        # j_off: offset for j in A and B
        # k_off: offset for k in B and C
        # A indices: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C indices: (k_off, i_off), (k_off, i_off+1), (k_off+1, i_off), (k_off+1, i_off+1)
        
        # M1 = (a+d)(x+v)
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # M2 = (c+d)x
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M3 = a(y-v)
        U[i_off*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = -1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # M4 = d(u-x)
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[j_off*p + k_off][col] = -1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        W[k_off*n + i_off][col] = -1
        col += 1
        
        # M5 = (a+b)v
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M6 = (c-a)(x+u)
        U[(i_off+1)*m + j_off][col] = 1
        U[i_off*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M7 = (b-d)(y+v)
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = -1
        col += 1
        
        return col

    # Apply Strassen to Block 1.1 (j=0,1, k=0,1)
    col = add_strassen(U, V, W, col, 0, 0, 0)
    # Apply Strassen to Block 1.2 (j=2,3, k=0,1)
    col = add_strassen(U, V, W, col, 0, 2, 0)
    
    # Block 2: 2x4x3 (k=2,3,4)
    # Split j=0,1 and j=2,3
    # Sub-block 2.1: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # Sub-block 2.2: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    
    # Rank 10 decomposition for 2x2x3
    def add_223(U, V, W, col, i_off, j_off, k_off):
        # k_off is start of 3 columns
        # A: 2x2, B: 2x3, C: 2x3
        # Coefficients for rank 10
        # M1 = a x
        U[i_off*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1
        
        # M2 = b u
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1
        
        # M3 = c y
        U[(i_off+1)*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M4 = d v
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M5 = (a+b)(y-v)
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = -1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M6 = (c+d)(x-u)
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + k_off][col] = -1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M7 = (a-b) z
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = -1
        V[j_off*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + i_off][col] = 1
        W[(k_off+2)*n + (i_off+1)][col] = -1
        col += 1
        
        # M8 = (c-d) w
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + (i_off+1)][col] = 1
        W[(k_off+2)*n + i_off][col] = -1
        col += 1
        
        # M9 = (a+b) w
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + i_off][col] = 1
        W[(k_off+2)*n + (i_off+1)][col] = 1
        col += 1
        
        # M10 = (c+d) z
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + i_off][col] = 1
        W[(k_off+2)*n + (i_off+1)][col] = 1
        col += 1
        
        return col

    # Apply 2x2x3 to Block 2.1 (j=0,1, k=2,3,4)
    col = add_223(U, V, W, col, 0, 0, 2)
    # Apply 2x2x3 to Block 2.2 (j=2,3, k=2,3,4)
    col = add_223(U, V, W, col, 0, 2, 2)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
