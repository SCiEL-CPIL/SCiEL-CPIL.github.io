import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _add_term(U, V, W, col, u_mask, v_mask, w_mask):
    """
    Add a rank-1 term at column 'col'.
    u_mask: dict mapping A index (i*m+j) to coefficient
    v_mask: dict mapping B index (j*p+k) to coefficient
    w_mask: dict mapping C index (k*n+i) to coefficient
    """
    for idx, val in u_mask.items():
        U[idx][col] += val
    for idx, val in v_mask.items():
        V[idx][col] += val
    for idx, val in w_mask.items():
        W[idx][col] += val

def _strassen_block(A_rows, B_cols, C_cols, p, col_offset, U, V, W):
    """
    Add a 7-term Strassen decomposition for a 2x2 block of B.
    A_rows: list of 2 A indices (i*m + j) for the rows of A
    B_cols: list of 2 B column indices (j*p + k) for the cols of B
    C_cols: list of 2 C column indices (k*n + i) for the cols of C
    p: the p dimension (for index calculation)
    col_offset: starting column index in U, V, W
    """
    # A is 2x2, indexed as A[0], A[1] (rows), with 2 cols each
    # But we need to map to the global U indices.
    # U index is i*m + j.
    # The "rows" of A are fixed i in {0,1}, j in {0,1}.
    # The "cols" of B are k in B_cols.
    # The "cols" of C are k in C_cols.
    
    # Let's assume A_rows = [0, 1, 2, 3] -> no, A is 2x2.
    # A_rows should be the 4 indices of A.
    # Actually, let's just pass the mapping.
    
    # For 2x2 A and 2x2 B block:
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,k1], f=B[0,k2], g=B[1,k1], h=B[1,k2]
    # C[0,k1], C[0,k2], C[1,k1], C[1,k2]
    
    # We need to know which A indices correspond to a,b,c,d.
    # And which B indices correspond to e,f,g,h.
    # And which C indices correspond to the outputs.
    
    # This helper is getting complex. Let's just hardcode the 7 terms for each block.
    pass

def construct():
    n, m, p = 2, 2, 5
    
    # Strategy:
    # Block 1: B cols 0,1. A is 2x2. C cols 0,1.
    # Block 2: B cols 2,3. A is 2x2. C cols 2,3.
    # Block 3: B col 4. A is 2x2. C col 4.
    
    # Total rank: 7 + 7 + 4 = 18.
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to define A indices
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    a, b, c, d = 0, 1, 2, 3
    
    # B indices: j*p + k
    # B col 0: 0:(0,0), 1:(1,0)
    # B col 1: 2:(0,1), 3:(1,1)
    # B col 2: 5:(0,2), 6:(1,2)
    # B col 3: 7:(0,3), 8:(1,3)
    # B col 4: 10:(0,4), 11:(1,4)
    
    # C indices: k*n + i
    # C col 0: 0:(0,0), 1:(0,1)  -- wait, k*n+i. k=0, i=0 -> 0. k=0, i=1 -> 1.
    # C col 1: 2:(1,0), 3:(1,1)  -- k=1, i=0 -> 2. k=1, i=1 -> 3.
    # C col 2: 4:(2,0), 5:(2,1)
    # C col 3: 6:(3,0), 7:(3,1)
    # C col 4: 8:(4,0), 9:(4,1)
    
    # --- Block 1: B cols 0,1 ---
    # e=0, f=2, g=1, h=3
    # C outputs: 0, 1, 2, 3
    
    # r=0: P1 = (a+d)(e+h)
    _add_term(U, V, W, 0, {a:1, d:1}, {e:1, h:1}, {0:1, 3:1})
    
    # r=1: P2 = (c+d)e
    _add_term(U, V, W, 1, {c:1, d:1}, {e:1}, {2:1, 3:-1})
    
    # r=2: P3 = a(f-h)
    _add_term(U, V, W, 2, {a:1}, {f:1, h:-1}, {1:1, 3:1})
    
    # r=3: P4 = d(g-e)
    _add_term(U, V, W, 3, {d:1}, {g:1, e:-1}, {0:1, 2:1})
    
    # r=4: P5 = (a+b)h
    _add_term(U, V, W, 4, {a:1, b:1}, {h:1}, {0:-1, 1:1})
    
    # r=5: P6 = (c-a)(e+f)
    _add_term(U, V, W, 5, {c:1, a:-1}, {e:1, f:1}, {3:1})
    
    # r=6: P7 = (b-d)(g+h)
    _add_term(U, V, W, 6, {b:1, d:-1}, {g:1, h:1}, {0:1})
    
    # --- Block 2: B cols 2,3 ---
    # e=5, f=7, g=6, h=8
    # C outputs: 4, 5, 6, 7
    
    # r=7: P1 = (a+d)(e+h)
    _add_term(U, V, W, 7, {a:1, d:1}, {e:1, h:1}, {4:1, 7:1})
    
    # r=8: P2 = (c+d)e
    _add_term(U, V, W, 8, {c:1, d:1}, {e:1}, {6:1, 7:-1})
    
    # r=9: P3 = a(f-h)
    _add_term(U, V, W, 9, {a:1}, {f:1, h:-1}, {5:1, 7:1})
    
    # r=10: P4 = d(g-e)
    _add_term(U, V, W, 10, {d:1}, {g:1, e:-1}, {4:1, 6:1})
    
    # r=11: P5 = (a+b)h
    _add_term(U, V, W, 11, {a:1, b:1}, {h:1}, {4:-1, 5:1})
    
    # r=12: P6 = (c-a)(e+f)
    _add_term(U, V, W, 12, {c:1, a:-1}, {e:1, f:1}, {7:1})
    
    # r=13: P7 = (b-d)(g+h)
    _add_term(U, V, W, 13, {b:1, d:-1}, {g:1, h:1}, {4:1})
    
    # --- Block 3: B col 4 ---
    # e=10, g=11 (only one column, so f,h are 0 or irrelevant)
    # C outputs: 8, 9
    
    # Schoolbook for 2x2 * 1x1 (column vector)
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4] = a*e + b*g
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4] = c*e + d*g
    
    # r=14: a*e -> C[0,4]
    _add_term(U, V, W, 14, {a:1}, {e:1}, {8:1})
    
    # r=15: b*g -> C[0,4]
    _add_term(U, V, W, 15, {b:1}, {g:1}, {8:1})
    
    # r=16: c*e -> C[1,4]
    _add_term(U, V, W, 16, {c:1}, {e:1}, {9:1})
    
    # r=17: d*g -> C[1,4]
    _add_term(U, V, W, 17, {d:1}, {g:1}, {9:1})
    
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
