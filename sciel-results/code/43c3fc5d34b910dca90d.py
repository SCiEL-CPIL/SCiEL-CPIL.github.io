"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify a decomposition
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    # Build Strassen + schoolbook decomposition
    # Use 6 disjoint 2x2x2 blocks for maximum savings
    # Block positions: (i0, j0, k0) where block covers [i0,i0+1] x [j0,j0+1] x [k0,k0+1]
    
    # Optimal placement to cover as much as possible:
    # We can use i-pairs (0,1) and (1,2) to cover all 3 i-values
    # j-pairs: (0,1), (2,3) - leaves j=4
    # k-pairs: (0,1), (2,3), (4,5) - leaves k=6
    # Using i=(0,1): covers j-pairs x k-pairs = 2*3 = 6 blocks
    # Using i=(1,2): covers j-pairs x k-pairs = 2*3 = 6 more blocks, but overlaps with i=1
    # Since Strassen blocks can overlap in indices (the tensor sum still works),
    # let's try using both i-pairs to cover more triples
    
    # Strategy: use i=(0,1) for all 6 j-k combinations, and i=(1,2) for remaining
    # Actually, each (i,j,k) triple only needs to be covered once.
    # With i=(0,1): covers i in {0,1} for all 6 (j,k) pairs -> 2*6*2 = wait
    # Each block covers 2*2*2 = 8 triples
    # 6 blocks with i=(0,1) cover: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # That's 2 * 4 * 6 = 48 triples
    # Remaining: i=2 (all j,k) = 35 triples, plus i in {0,1} with j=4 or k=6
    # j=4: 2*1*7 = 14, k=6: 2*5*1 = 10, but (i in {0,1}, j=4, k=6) counted twice = 2
    # So remaining = 35 + 14 + 10 - 2 = 57... same as before
    
    # Alternative: also use i=(1,2) blocks for j=4 and k=6 regions
    # Block at (1,4,6)? No, j=4 is single, can't make 2x2
    # The limitation is j=4 and k=6 are single indices
    
    # Let's try: use i=(1,2) for some blocks to cover i=2 more efficiently
    # Block (1,0,0): covers i={1,2}, j={0,1}, k={0,1} - but i=1 already covered
    # This doesn't reduce schoolbook count since we still need all 105 triples covered
    # The benefit of Strassen is 7 terms for 8 triples instead of 8 terms
    # So we want to maximize the number of 2x2x2 blocks, each saving 1 term
    
    # Max disjoint 2x2x2 blocks in 3x5x7:
    # i: can fit 1 pair (0,1) or (1,2), not both disjointly
    # j: can fit 2 pairs (0,1) and (2,3)
    # k: can fit 3 pairs (0,1), (2,3), (4,5)
    # So max disjoint = 1*2*3 = 6 blocks, saving 6 terms -> rank 105-6=99
    
    # To do better, we'd need non-disjoint blocks or different algorithms
    # For now, keep the 6 blocks but verify correctness
    blocks = [
        (0, 0, 0),  # i:{0,1}, j:{0,1}, k:{0,1}
        (0, 2, 0),  # i:{0,1}, j:{2,3}, k:{0,1}
        (0, 0, 2),  # i:{0,1}, j:{0,1}, k:{2,3}
        (0, 2, 2),  # i:{0,1}, j:{2,3}, k:{2,3}
        (0, 0, 4),  # i:{0,1}, j:{0,1}, k:{4,5}
        (0, 2, 4),  # i:{0,1}, j:{2,3}, k:{4,5}
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i0, j0, k0):
        """Add 7 Strassen terms for a 2x2x2 block at offset (i0, j0, k0).
        
        Standard Strassen (0-indexed within block):
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 - M4 + M5 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        """
        # M1: (A00+A11)(B00+B11) -> C00: +1, C11: +1
        U = {(i0, j0): 1, (i0+1, j0+1): 1}
        V = {(j0, k0): 1, (j0+1, k0+1): 1}
        W = {(k0, i0): 1, (k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M2: (A10+A11)B00 -> C10: +1, C11: -1
        U = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        V = {(j0, k0): 1}
        W = {(k0, i0+1): 1, (k0+1, i0+1): -1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M3: A00(B01-B11) -> C01: +1, C11: +1
        U = {(i0, j0): 1}
        V = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        W = {(k0+1, i0): 1, (k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M4: A11(B10-B00) -> C00: -1, C10: +1
        U = {(i0+1, j0+1): 1}
        V = {(j0+1, k0): 1, (j0, k0): -1}
        W = {(k0, i0): -1, (k0, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M5: (A00+A01)B11 -> C00: +1, C01: +1
        U = {(i0, j0): 1, (i0, j0+1): 1}
        V = {(j0+1, k0+1): 1}
        W = {(k0, i0): 1, (k0+1, i0): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M6: (A10-A00)(B00+B01) -> C11: +1
        U = {(i0+1, j0): 1, (i0, j0): -1}
        V = {(j0, k0): 1, (j0, k0+1): 1}
        W = {(k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M7: (A01-A11)(B10+B11) -> C00: -1, C11: +1
        U = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        V = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        W = {(k0, i0): -1, (k0+1, i0+1): 1}
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

    # Add all Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0)
    
    # Identify covered (i,j,k) triples
    covered = set()
    for i0, j0, k0 in blocks:
        for i in [i0, i0+1]:
            for j in [j0, j0+1]:
                for k in [k0, k0+1]:
                    covered.add((i, j, k))
    
    # Add schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U = {(i, j): 1}
                    V = {(j, k): 1}
                    W = {(k, i): 1}
                    U_cols.append(U); V_cols.append(V); W_cols.append(W)
    
    # Convert sparse columns to dense matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U_cols[c].items():
            U[i * m + j][c] = val
        for (j, k), val in V_cols[c].items():
            V[j * p + k][c] = val
        for (k, i), val in W_cols[c].items():
            W[k * n + i][c] = val
    
    # Verify the decomposition
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
