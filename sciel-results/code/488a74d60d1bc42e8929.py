import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank 42 decomposition using 6 Strassen blocks
    
    rank = 42
    # U shape: (n*m, rank) = (8, 42)
    # V shape: (m*p, rank) = (24, 42)
    # W shape: (p*n, rank) = (12, 42)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices:
    # A: (i, j) with i in 0..1, j in 0..3
    # B: (j, k) with j in 0..3, k in 0..5
    # C: (l, m) with l in 0..1, m in 0..5
    
    # Strassen block function
    def add_strassen_block(offset, i_base, j_base, k_base, l_base, m_base):
        # i_base, j_base, k_base, l_base, m_base are the starting indices for the 2x2x2 block
        # A block is A[i_base:i_base+2, j_base:j_base+2]
        # B block is B[j_base:j_base+2, k_base:k_base+2]
        # C block is C[l_base:l_base+2, m_base:m_base+2]
        
        # M1 = (a+d)(e+h)
        # a=A[i0,j0], d=A[i1,j1], e=B[j0,k0], h=B[j1,k1]
        U[i_base*m + j_base][offset] += 1
        U[(i_base+1)*m + (j_base+1)][offset] += 1
        V[j_base*p + k_base][offset] += 1
        V[(j_base+1)*p + (k_base+1)][offset] += 1
        W[l_base*n + m_base][offset] += 1
        W[(l_base+1)*n + (m_base+1)][offset] += 1
        
        # M2 = (c+d)e
        # c=A[i1,j0], d=A[i1,j1], e=B[j0,k0]
        U[(i_base+1)*m + j_base][offset+1] += 1
        U[(i_base+1)*m + (j_base+1)][offset+1] += 1
        V[j_base*p + k_base][offset+1] += 1
        W[l_base*n + (m_base+1)][offset+1] += 1
        W[(l_base+1)*n + (m_base+1)][offset+1] += 1
        
        # M3 = a(f-h)
        # a=A[i0,j0], f=B[j0,k1], h=B[j1,k1]
        U[i_base*m + j_base][offset+2] += 1
        V[j_base*p + (k_base+1)][offset+2] += 1
        V[(j_base+1)*p + (k_base+1)][offset+2] -= 1
        W[l_base*n + (m_base+1)][offset+2] += 1
        
        # M4 = d(g-e)
        # d=A[i1,j1], g=B[j1,k0], e=B[j0,k0]
        U[(i_base+1)*m + (j_base+1)][offset+3] += 1
        V[(j_base+1)*p + k_base][offset+3] += 1
        V[j_base*p + k_base][offset+3] -= 1
        W[(l_base+1)*n + m_base][offset+3] += 1
        
        # M5 = (a+b)h
        # a=A[i0,j0], b=A[i0,j1], h=B[j1,k1]
        U[i_base*m + j_base][offset+4] += 1
        U[i_base*m + (j_base+1)][offset+4] += 1
        V[(j_base+1)*p + (k_base+1)][offset+4] += 1
        W[l_base*n + m_base][offset+4] += 1
        W[(l_base+1)*n + (m_base+1)][offset+4] -= 1
        
        # M6 = (c-a)(e+f)
        # c=A[i1,j0], a=A[i0,j0], e=B[j0,k0], f=B[j0,k1]
        U[(i_base+1)*m + j_base][offset+5] += 1
        U[i_base*m + j_base][offset+5] -= 1
        V[j_base*p + k_base][offset+5] += 1
        V[j_base*p + (k_base+1)][offset+5] += 1
        W[(l_base+1)*n + (m_base+1)][offset+5] += 1
        
        # M7 = (b-d)(g+h)
        # b=A[i0,j1], d=A[i1,j1], g=B[j1,k0], h=B[j1,k1]
        U[i_base*m + (j_base+1)][offset+6] += 1
        U[(i_base+1)*m + (j_base+1)][offset+6] -= 1
        V[(j_base+1)*p + k_base][offset+6] += 1
        V[(j_base+1)*p + (k_base+1)][offset+6] += 1
        W[l_base*n + m_base][offset+6] += 1
        W[(l_base+1)*n + m_base][offset+6] -= 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(7, 0, 0, 2, 0, 2)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_block(14, 0, 0, 4, 0, 4)
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(21, 0, 2, 0, 0, 0)
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(28, 0, 2, 2, 0, 2)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen_block(35, 0, 2, 4, 0, 4)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
