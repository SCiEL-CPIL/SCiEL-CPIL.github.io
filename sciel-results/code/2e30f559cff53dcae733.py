import time
import sys

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # We know the rank is 32
    rank = 32
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Blocks 1-4: 2x2 blocks
    for t in range(4):
        k0 = 2 * t
        k1 = k0 + 1
        
        # P1: (a11 + a22)(b11 + b22)
        # U: a11(0), a22(3)
        # V: b11(k0), b22(9+k1)
        # W: c11(2k0), c22(2k1+1)
        add_term(0, 1, k0, 1, 2*k0, 1)
        add_term(3, 1, 9+k1, 1, 2*k1+1, 1)
        # Wait, add_term increments col. I need to put all non-zeros in the SAME column.
        # My helper is wrong for multi-index terms.
        
    # Redo with correct logic
    col = 0
    def add_rank1_term(u_sparsity, v_sparsity, w_sparsity):
        nonlocal col
        for idx, val in u_sparsity:
            U[idx][col] += val
        for idx, val in v_sparsity:
            V[idx][col] += val
        for idx, val in w_sparsity:
            W[idx][col] += val
        col += 1

    for t in range(4):
        k0 = 2 * t
        k1 = k0 + 1
        
        # P1
        add_rank1_term(
            [(0, 1), (3, 1)],
            [(k0, 1), (9+k1, 1)],
            [(2*k0, 1), (2*k1+1, 1)]
        )
        # P2
        add_rank1_term(
            [(2, 1), (3, 1)],
            [(k0, 1)],
            [(2*k0+1, 1), (2*k1+1, -1)]
        )
        # P3
        add_rank1_term(
            [(0, 1)],
            [(k1, 1), (9+k1, -1)],
            [(2*k1, 1), (2*k1+1, 1)]
        )
        # P4
        add_rank1_term(
            [(3, 1)],
            [(9+k0, 1), (k0, -1)],
            [(2*k0, 1), (2*k0+1, 1)]
        )
        # P5
        add_rank1_term(
            [(0, 1), (1, 1)],
            [(9+k1, 1)],
            [(2*k0, -1), (2*k1, 1)]
        )
        # P6
        add_rank1_term(
            [(2, 1), (0, -1)],
            [(k0, 1), (k1, 1)],
            [(2*k1+1, 1)]
        )
        # P7
        add_rank1_term(
            [(1, 1), (3, -1)],
            [(9+k0, 1), (9+k1, 1)],
            [(2*k0, 1)]
        )
        
    # Block 5: 2x1 block, k0 = 8
    k0 = 8
    # S1: a11 b11
    add_rank1_term(
        [(0, 1)],
        [(k0, 1)],
        [(2*k0, 1)]
    )
    # S2: a12 b21
    add_rank1_term(
        [(1, 1)],
        [(9+k0, 1)],
        [(2*k0, 1)]
    )
    # S3: a21 b11
    add_rank1_term(
        [(2, 1)],
        [(k0, 1)],
        [(2*k0+1, 1)]
    )
    # S4: a22 b21
    add_rank1_term(
        [(3, 1)],
        [(9+k0, 1)],
        [(2*k0+1, 1)]
    )
    
    return {"U": U, "V": V, "W": W}
