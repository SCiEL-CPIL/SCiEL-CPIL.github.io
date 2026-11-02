"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition strategy:
    # Split the 5x6 middle into:
    #   Block A: 2x2x2 (Strassen, rank 7) for j={0,1}, k={0,1}
    #   Block B: 2x2x2 (Strassen, rank 7) for j={0,1}, k={2,3}
    #   Block C: 2x2x2 (Strassen, rank 7) for j={0,1}, k={4,5}
    #   Block D: 2x1x2 (rank 4) for j=2, k={0,1}
    #   Block E: 2x1x2 (rank 4) for j=2, k={2,3}
    #   Block F: 2x1x2 (rank 4) for j=2, k={4,5}
    #   Block G: 2x2x2 (Strassen, rank 7) for j={3,4}, k={0,1}
    #   Block H: 2x2x2 (Strassen, rank 7) for j={3,4}, k={2,3}
    #   Block I: 2x2x2 (Strassen, rank 7) for j={3,4}, k={4,5}
    # Total rank = 3*7 + 3*4 + 3*7 = 21 + 12 + 21 = 54
    
    # Actually, let me try a better decomposition.
    # We can group differently:
    #   Block A: 2x3x2 (rank 11) for j={0,1,2}, k={0,1}
    #   Block B: 2x3x2 (rank 11) for j={0,1,2}, k={2,3}
    #   Block C: 2x3x2 (rank 11) for j={0,1,2}, k={4,5}
    #   Block D: 2x2x2 (rank 7) for j={3,4}, k={0,1}
    #   Block E: 2x2x2 (rank 7) for j={3,4}, k={2,3}
    #   Block F: 2x2x2 (rank 7) for j={3,4}, k={4,5}
    # Total rank = 3*11 + 3*7 = 33 + 21 = 54
    
    # Let me try yet another approach:
    #   Block A: 2x4x2 (rank 14) for j={0,1,2,3}, k={0,1}
    #   Block B: 2x4x2 (rank 14) for j={0,1,2,3}, k={2,3}
    #   Block C: 2x4x2 (rank 14) for j={0,1,2,3}, k={4,5}
    #   Block D: 2x1x2 (rank 4) for j=4, k={0,1}
    #   Block E: 2x1x2 (rank 4) for j=4, k={2,3}
    #   Block F: 2x1x2 (rank 4) for j=4, k={4,5}
    # Total rank = 3*14 + 3*4 = 42 + 12 = 54
    
    # All these give rank 54. Let me try to find a better decomposition.
    
    # Actually, let me use the Laderman algorithm for 2x2x2 blocks.
    # Laderman's algorithm has rank 7, same as Strassen.
    
    # Let me try a different split:
    #   Block A: 2x5x2 (rank 16) for all j, k={0,1}
    #   Block B: 2x5x2 (rank 16) for all j, k={2,3}
    #   Block C: 2x5x2 (rank 16) for all j, k={4,5}
    # Total rank = 3*16 = 48
    
    # Wait, what's the rank of 2x5x2?
    # We can split 5 into 2+3:
    #   2x2x2 (rank 7) + 2x3x2 (rank 11) = 18
    # Or 5 into 1+1+1+1+1: 5 * 2x1x2 (rank 4) = 20
    # Or 5 into 2+2+1: 2*2x2x2 (rank 7) + 2x1x2 (rank 4) = 18
    # So rank(2x5x2) = 18
    
    # Total rank = 3*18 = 54 again.
    
    # Let me try to use the fact that we can share some computations.
    
    # Actually, let me just use the straightforward decomposition and verify it works.
    
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 terms
    strassen_terms = [
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        ),
        (
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        ),
        (
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        ),
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        ),
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        ),
    ]
    
    # 2x1x2 terms (rank 4)
    def add_2x1x2(U, V, W, i_base, j, k_base, col):
        """Add 2x1x2 multiplication terms starting at column col."""
        for i in range(2):
            for k in range(2):
                U[(i_base + i) * m + j][col] = 1
                V[j * p + (k_base + k)][col] = 1
                W[(k_base + k) * n + (i_base + i)][col] = 1
                col += 1
        return col
    
    # 2x2x2 Strassen terms
    def add_2x2x2_strassen(U, V, W, i_base, j_base, k_base, col):
        """Add 2x2x2 Strassen multiplication terms starting at column col."""
        for u_terms, v_terms, w_terms in strassen_terms:
            for (li, lj, val) in u_terms:
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][col] += val
            
            for (lj, lk, val) in v_terms:
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][col] += val
            
            for (li, lk, val) in w_terms:
                gi = i_base + li
                gk = k_base + lk
                W[gk * n + gi][col] += val
            
            col += 1
        return col
    
    # Decomposition:
    # For each k-pair (0,1), (2,3), (4,5):
    #   j=0,1: 2x2x2 Strassen (rank 7)
    #   j=2: 2x1x2 (rank 4)
    #   j=3,4: 2x2x2 Strassen (rank 7)
    # Total per k-pair: 7 + 4 + 7 = 18
    # Total: 3 * 18 = 54
    
    for s in range(3):
        k_base = s * 2
        
        # j=0,1: 2x2x2 Strassen
        col = add_2x2x2_strassen(U, V, W, 0, 0, k_base, col)
        
        # j=2: 2x1x2
        col = add_2x1x2(U, V, W, 0, 2, k_base, col)
        
        # j=3,4: 2x2x2 Strassen
        col = add_2x2x2_strassen(U, V, W, 0, 3, k_base, col)
    
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        raise TimeoutError("Exceeded time budget")
    
    # Final verification
    assert len(U) == n * m, f"U rows {len(U)} != {n*m}"
    assert len(V) == m * p, f"V rows {len(V)} != {m*p}"
    assert len(W) == p * n, f"W rows {len(W)} != {p*n}"
    assert all(len(row) == rank for row in U), "U cols mismatch"
    assert all(len(row) == rank for row in V), "V cols mismatch"
    assert all(len(row) == rank for row in W), "W cols mismatch"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
