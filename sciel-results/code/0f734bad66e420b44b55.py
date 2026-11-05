"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a low-rank exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses block structure with Strassen for 2x2x2 sub-blocks where applicable.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Strategy: split into blocks and use Strassen where 2x2x2 sub-problems exist
    # 4x5x6: split rows into 2+2, cols into 2+2+1, inner into 2+2+1
    # For each block pair (I,J,K) where sizes are (2,2,2), use Strassen
    # For other blocks, use schoolbook
    
    # Strassen 2x2x2: 7 rank-1 terms
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+cg, af+dh],[be+dg, bf+ch]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-d)(e+g)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def add_strassen_222(i0, j0, k0, col_start):
        """Add 7 rank-1 terms for 2x2x2 block at (i0,j0,k0). Returns next col."""
        nonlocal col
        # The 2x2x2 tensor at positions (i0..i0+1, j0..j0+1, k0..k0+1)
        # U indices: i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
        # V indices: j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
        # W indices: k0*n+i0, k0*n+i0+1, (k0+1)*n+i0, (k0+1)*n+i0+1
        
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + j0 + 1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k0 + 1
        v10 = (j0 + 1) * p + k0
        v11 = (j0 + 1) * p + k0 + 1
        
        w00 = k0 * n + i0
        w01 = k0 * n + i0 + 1
        w10 = (k0 + 1) * n + i0
        w11 = (k0 + 1) * n + i0 + 1
        
        # M1 = (a+d)(e+h): U has 1 at u00,u11; V has 1 at v00,v11; W has 1 at w00,w11
        c = col_start
        U[u00][c] = 1; U[u11][c] = 1
        V[v00][c] = 1; V[v11][c] = 1
        W[w00][c] = 1; W[w11][c] = 1
        col = c + 1
        
        # M2 = (c+d)e: U has 1 at u10,u11; V has 1 at v00; W has 1 at w00,w10
        c = col
        U[u10][c] = 1; U[u11][c] = 1
        V[v00][c] = 1
        W[w00][c] = 1; W[w10][c] = 1
        col = c + 1
        
        # M3 = a(f-h): U has 1 at u00; V has 1 at v01,v11; W has 1 at w01,w11
        # V: 1 at v01, -1 at v11
        c = col
        U[u00][c] = 1
        V[v01][c] = 1; V[v11][c] = -1
        W[w01][c] = 1; W[w11][c] = 1
        col = c + 1
        
        # M4 = d(g-c): U has 1 at u11; V has 1 at v10,v00; W has 1 at w00,w10
        # V: 1 at v10, -1 at v00
        c = col
        U[u11][c] = 1
        V[v10][c] = 1; V[v00][c] = -1
        W[w00][c] = 1; W[w10][c] = 1
        col = c + 1
        
        # M5 = (a+b)h: U has 1 at u00,u01; V has 1 at v11; W has 1 at w11,w01
        c = col
        U[u00][c] = 1; U[u01][c] = 1
        V[v11][c] = 1
        W[w11][c] = 1; W[w01][c] = 1
        col = c + 1
        
        # M6 = (c-d)(f+h): U has 1 at u10,-1 at u11; V has 1 at v01,v11; W has 1 at w11
        c = col
        U[u10][c] = 1; U[u11][c] = -1
        V[v01][c] = 1; V[v11][c] = 1
        W[w11][c] = 1
        col = c + 1
        
        # M7 = (b-d)(e+g): U has 1 at u01,-1 at u11; V has 1 at v00,v10; W has 1 at w00,w10
        c = col
        U[u01][c] = 1; U[u11][c] = -1
        V[v00][c] = 1; V[v10][c] = 1
        W[w00][c] = 1; W[w10][c] = 1
        col = c + 1
        
        return col
    
    # Block decomposition of 4x5x6:
    # Split: rows [0,1,2,3] -> [0,1],[2,3]
    #        cols [0,1,2,3,4] -> [0,1],[2,3],[4]
    #        inner [0,1,2,3,4,5] -> [0,1],[2,3],[4,5]
    
    # For 2x2x2 blocks, use Strassen (7 terms instead of 8)
    # For other blocks, use schoolbook (product of sizes terms)
    
    row_blocks = [(0, 2), (2, 4)]
    col_blocks = [(0, 2), (2, 4), (4, 5)]
    inner_blocks = [(0, 2), (2, 4), (4, 6)]
    
    for (i_start, i_end) in row_blocks:
        for (j_start, j_end) in col_blocks:
            for (k_start, k_end) in inner_blocks:
                ri = i_end - i_start
                rj = j_end - j_start
                rk = k_end - k_start
                
                if ri == 2 and rj == 2 and rk == 2:
                    add_strassen_222(i_start, j_start, k_start, col)
                else:
                    # Schoolbook for this block
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for k in range(k_start, k_end):
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
