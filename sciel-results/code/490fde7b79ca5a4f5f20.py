"""Rewritten program for 2x4x5 matrix multiplication tensor decomposition.

Strategy: Split the 2x4 * 4x5 multiplication into six 2x2 * 2x2 blocks and two
2x2 * 2x1 blocks.  Each 2x2*2x2 block uses Strassen's 7-multiplication algorithm.
The two 2x2*2x1 blocks use the trivial rank-4 decomposition.
Total rank = 5*7 + 2*4 = 43.

Wait, that's worse than 36. Let me reconsider.

Actually, looking more carefully: the current approach splits A into left (2x2) and
right (2x2), and B into top (2x5) and bottom (2x5). Then:
- A_L * B_T: 2x2 * 2x5, split into 2x2*2x2 + 2x2*2x2 + 2x2*2x1 = 7+7+4 = 18
- A_R * B_B: 2x2 * 2x5, split into 2x2*2x2 + 2x2*2x2 + 2x2*2x1 = 7+7+4 = 18
Total = 36.

But we could also do:
- A_L * B_T and A_R * B_T both contribute to the same output columns.
- We could potentially combine some Strassen terms.

Actually, let me think about a different block structure. What if we split B into
three parts: 2x2, 2x2, 2x1 (columns 0-1, 2-3, 4)?

A is 2x4, B is 4x5.
C[i,k] = sum_j A[i,j] * B[j,k]

Let me try: split A into A_left (2x2, cols 0,1) and A_right (2x2, cols 2,3).
Split B into B_top (rows 0,1) and B_bottom (rows 2,3).

C = A_left * B_top + A_right * B_bottom

Each 2x2 * 2x5 multiplication. For 2x2 * 2x5:
Split the 2x5 into 2x2 + 2x2 + 2x1.
2x2*2x2: rank 7 (Strassen)
2x2*2x2: rank 7 (Strassen)  
2x2*2x1: rank 4 (trivial)
Total per 2x2*2x5: 18

Two such multiplications: 36. Same as before.

Alternatively, can we do better on 2x2*2x5?
The tensor for 2x2*2x5 has rank... let me think. It's the sum of 2x2*2x2 and 2x2*2x1.
Actually, 2x2*2x5 is equivalent to multiplying a 2x2 matrix by a 2x5 matrix.
This can be seen as: for each of the 5 columns, compute a 2x2*2x1 product.
But that's 5 * 4 = 20, which is worse.

The Strassen-based approach gives 7+7+4 = 18, which seems good.

Let me just implement this cleanly.
"""

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    # Count total rank: 5 Strassen blocks (7 each) + 2 trivial blocks (4 each) = 35 + 8 = 43
    # Wait, that's 43. But the current is 36. Let me recount.
    
    # Current approach:
    # A_L (2x2) * B_T (2x5): split B_T into cols 0-1 (2x2), 2-3 (2x2), 4 (2x1)
    #   -> 7 + 7 + 4 = 18
    # A_R (2x2) * B_B (2x5): split B_B into cols 0-1 (2x2), 2-3 (2x2), 4 (2x1)
    #   -> 7 + 7 + 4 = 18
    # Total: 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_2x2(y, z, w_offsets):
        """
        y: list of 4 (i,j) indices for the 2x2 Y block
        z: list of 4 (j,k) indices for the 2x2 Z block
        w_offsets: list of 4 (k,i) indices for the 2x2 result positions
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w_offsets[0])][col] += 1
        W[w_idx(*w_offsets[3])][col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*w_offsets[2])][col] += 1
        W[w_idx(*w_offsets[3])][col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z01)][col] = 1
        V[v_idx(*z11)][col] = -1
        W[w_idx(*w_offsets[1])][col] += 1
        W[w_idx(*w_offsets[3])][col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z00)][col] = -1
        W[w_idx(*w_offsets[0])][col] += 1
        W[w_idx(*w_offsets[2])][col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w_offsets[0])][col] -= 1
        W[w_idx(*w_offsets[1])][col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y00)][col] = -1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*w_offsets[3])][col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] = 1
        U[u_idx(*y11)][col] = -1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w_offsets[0])][col] += 1
        col += 1
    
    # A_L: top-left 2x2 of A (rows 0,1; cols 0,1)
    # A_R: top-right 2x2 of A (rows 0,1; cols 2,3)
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    
    # B_T: top 2x5 of B (rows 0,1; all cols)
    # B_B: bottom 2x5 of B (rows 2,3; all cols)
    
    # For A_L * B_T:
    # B_T cols 0-1: (0,0),(0,1),(1,0),(1,1)
    # B_T cols 2-3: (0,2),(0,3),(1,2),(1,3)
    # B_T col 4: (0,4),(1,4)
    
    # C output for A_L*B_T: C[0:2, 0:5]
    # W index: k*n + i where k is col of C, i is row of C
    
    # Block 1: A_L * B_T[:,0:2] -> C[:,0:2]
    B_T_01 = [(0,0), (0,1), (1,0), (1,1)]
    C_01 = [(0,0), (1,0), (0,1), (1,1)]  # (k,i): (0,0),(1,0),(0,1),(1,1)
    strassen_2x2(A_L, B_T_01, C_01)
    
    # Block 2: A_L * B_T[:,2:4] -> C[:,2:4]
    B_T_23 = [(0,2), (0,3), (1,2), (1,3)]
    C_23 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_2x2(A_L, B_T_23, C_23)
    
    # Block 3: A_L * B_T[:,4] -> C[:,4] (2x2 * 2x1, rank 4)
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # For A_R * B_B:
    # B_B cols 0-1: (2,0),(2,1),(3,0),(3,1)
    # B_B cols 2-3: (2,2),(2,3),(3,2),(3,3)
    # B_B col 4: (2,4),(3,4)
    
    # Block 4: A_R * B_B[:,0:2] -> C[:,0:2]
    B_B_01 = [(2,0), (2,1), (3,0), (3,1)]
    strassen_2x2(A_R, B_B_01, C_01)
    
    # Block 5: A_R * B_B[:,2:4] -> C[:,2:4]
    B_B_23 = [(2,2), (2,3), (3,2), (3,3)]
    strassen_2x2(A_R, B_B_23, C_23)
    
    # Block 6: A_R * B_B[:,4] -> C[:,4] (2x2 * 2x1, rank 4)
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
