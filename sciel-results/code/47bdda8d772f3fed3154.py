import time
import sys

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=5
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_rows, v_rows, w_rows):
        """
        u_rows: list of (index, value)
        v_rows: list of (index, value)
        w_rows: list of (index, value)
        """
        col_U = [0.0] * (n * m)
        col_V = [0.0] * (m * p)
        col_W = [0.0] * (p * n)
        
        for idx, val in u_rows:
            col_U[idx] = val
        for idx, val in v_rows:
            col_V[idx] = val
        for idx, val in w_rows:
            col_W[idx] = val
            
        U.append(col_U)
        V.append(col_V)
        W.append(col_W)

    # Helper to add schoolbook term for specific i, j, k
    def add_schoolbook(i, j, k):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # 1. Handle i=2 (1x5x5 block) with Schoolbook
    for j in range(m):
        for k in range(p):
            add_schoolbook(2, j, k)
            
    # 2. Handle i in {0, 1} (2x5x5 block)
    # Partition j (inner) into J0={0,1}, J1={2,3}, J2={4}
    # Partition k (outer) into K0={0,1}, K1={2,3}, K2={4}
    
    J_blocks = [
        [0, 1],
        [2, 3],
        [4]
    ]
    K_blocks = [
        [0, 1],
        [2, 3],
        [4]
    ]
    
    for J in J_blocks:
        for K in K_blocks:
            if len(J) == 2 and len(K) == 2:
                # Strassen 2x2x2
                j0, j1 = J
                k0, k1 = K
                
                # M1: (A00+A11)(B00+B11) -> C00 + C11
                add_term(
                    [(0*m+j0, 1), (1*m+j1, 1)],
                    [(j0*p+k0, 1), (j1*p+k1, 1)],
                    [(k0*n+0, 1), (k1*n+1, 1)]
                )
                
                # M2: (A10+A11)B00 -> C10
                add_term(
                    [(1*m+j0, 1), (1*m+j1, 1)],
                    [(j0*p+k0, 1)],
                    [(k0*n+1, 1)]
                )
                
                # M3: A00(B10-B11) -> C00
                add_term(
                    [(0*m+j0, 1)],
                    [(j1*p+k0, 1), (j1*p+k1, -1)],
                    [(k0*n+0, 1)]
                )
                
                # M4: A11(B01-B00) -> C11
                add_term(
                    [(1*m+j1, 1)],
                    [(j0*p+k1, 1), (j0*p+k0, -1)],
                    [(k1*n+1, 1)]
                )
                
                # M5: (A00+A01)B11 -> C01
                add_term(
                    [(0*m+j0, 1), (0*m+j1, 1)],
                    [(j1*p+k1, 1)],
                    [(k1*n+0, 1)]
                )
                
                # M6: (A10-A00)(B00+B01) -> C11
                add_term(
                    [(1*m+j0, 1), (0*m+j0, -1)],
                    [(j0*p+k0, 1), (j0*p+k1, 1)],
                    [(k1*n+1, 1)]
                )
                
                # M7: (A01-A11)(B10+B11) -> C00
                add_term(
                    [(0*m+j1, 1), (1*m+j1, -1)],
                    [(j1*p+k0, 1), (j1*p+k1, 1)],
                    [(k0*n+0, 1)]
                )
            else:
                # Schoolbook for this block
                for j in J:
                    for k in K:
                        # i=0
                        add_schoolbook(0, j, k)
                        # i=1
                        add_schoolbook(1, j, k)
                        
    # Convert lists of lists to required format
    # The task expects {'U': U, 'V': V, 'W': W}
    # U, V, W are lists of columns.
    # We need to transpose them to rows?
    # "factor matrices of shapes (n*m, rank)"
    # My U is a list of rank vectors (columns).
    # So U is currently rank x (n*m).
    # I need to return it as (n*m) x rank.
    
    def transpose(mat):
        if not mat: return []
        return [list(row) for row in zip(*mat)]
        
    U_t = transpose(U)
    V_t = transpose(V)
    W_t = transpose(W)
    
    return {"U": U_t, "V": V_t, "W": W_t}
# EVOLVE-BLOCK-END
