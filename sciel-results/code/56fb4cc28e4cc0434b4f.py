"""Matrix multiplication 2,2,13: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 13
    deadline = time.time() + 140  # safety margin before 150s kill

    # Build the 2x2x13 multiplication tensor T of shape (4, 26, 26)
    # T[a_idx, b_idx, c_idx] = 1 if C[c_idx] = sum A[a_idx]*B[b_idx] for that term
    # A is 2x2: a_idx = i*m + j = i*2 + j, i in [0,1], j in [0,1]
    # B is 2x13: b_idx = j*p + k = j*13 + k, j in [0,1], k in [0,12]
    # C is 2x13: c_idx = k*n + i = k*2 + i, k in [0,12], i in [0,1]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1

    # We want to find rank-minimal decomposition:
    # T[a,b,c] = sum_{r} U[a,r] * V[b,r] * W[c,r]

    # Use Strassen for the 2x2 blocks and schoolbook for the remaining columns.
    # p=13: columns k=0..12.
    # Group k into blocks of 2: k=0,1; 2,3; 4,5; 6,7; 8,9; 10,11; then k=12 alone.
    # That's 6 Strassen blocks (rank 7 each) + 1 schoolbook block (rank 4) = 6*7+4 = 46.
    # Total rank: 46

    # Actually let's verify: for each pair of columns (k, k+1), we multiply A(2x2) by B_col(2x2).
    # Strassen gives rank 7 for each 2x2x2 block.
    # For the last single column k=12, schoolbook gives rank 4.
    # Total: 6*7 + 4 = 46.

    # But wait - can we do better? The tensor is 2x2x13.
    # The rank of 2x2xp multiplication: known results suggest rank is around 7*floor(p/2) + 4*ceil(p/2) for block approach.
    # For p=13: 6*7 + 1*4 = 46. This is the standard block decomposition.

    # Let's construct the 46-rank decomposition.
    # For each Strassen block (columns k, k+1 where k is even, k=0,2,4,6,8,10):
    #   Strassen 2x2x2: 7 terms
    # For the schoolbook block (column k=12):
    #   4 terms

    # Strassen 2x2x2 decomposition:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # In our indexing for a block at columns (k, k+1):
    # A indices: a_idx = i*2+j for i,j in [0,1] -> (0,1,2,3) for (0,0),(0,1),(1,0),(1,1)
    # B indices: b_idx = j*13+k for the block. For column pair (k,k+1):
    #   B[0, k] = b_idx 0*13+k = k
    #   B[0, k+1] = b_idx 0*13+k+1 = k+1
    #   B[1, k] = b_idx 1*13+k = 13+k
    #   B[1, k+1] = b_idx 1*13+k+1 = 13+k+1
    # C indices: c_idx = k'*2+i for k' in [k,k+1], i in [0,1]:
    #   C[0,k] = k*2+0 = 2k
    #   C[0,k+1] = (k+1)*2+0 = 2k+2
    #   C[1,k] = k*2+1 = 2k+1
    #   C[1,k+1] = (k+1)*2+1 = 2k+3

    # U has shape (4, rank), V has shape (26, rank), W has shape (26, rank)
    # U rows: A indices (4)
    # V rows: B indices (26)
    # W rows: C indices (26)

    U_cols = []
    V_cols = []
    W_cols = []

    # Strassen block helper
    def add_strassen_block(k_base):
        """Add Strassen decomposition for columns k_base and k_base+1."""
        # A: [[A00, A01], [A10, A11]] at U rows [0,1,2,3]
        # B: [[B0k, B0(k+1)], [B1k, B1(k+1)]] at V rows [k_base, k_base+1, 13+k_base, 13+k_base+1]
        # C: [[C0k, C0(k+1)], [C1k, C1(k+1)]] at W rows [2k_base, 2k_base+2, 2k_base+1, 2k_base+3]

        b0k = k_base          # B[0, k_base]
        b1kp1 = 13 + k_base + 1  # B[1, k_base+1]
        b0kp1 = k_base + 1   # B[0, k_base+1]
        b1k = 13 + k_base    # B[1, k_base]

        c0k = 2 * k_base          # C[0, k_base]
        c0kp1 = 2 * k_base + 2    # C[0, k_base+1]
        c1k = 2 * k_base + 1      # C[1, k_base]
        c1kp1 = 2 * k_base + 3    # C[1, k_base+1]

        # Strassen terms (M1 through M7)
        # M1 = (A00+A11)(B0k+B1kp1)
        U1 = [1, 0, 1, 0]
        V1 = [0]*26; V1[b0k] = 1; V1[b1kp1] = 1
        W1 = [0]*26; W1[c0k] = 1; W1[c1kp1] = 1
        U_cols.append(U1); V_cols.append(V1); W_cols.append(W1)

        # M2 = (A10+A11)B0k
        U2 = [0, 0, 1, 1]
        V2 = [0]*26; V2[b0k] = 1
        W2 = [0]*26; W2[c1k] = -1; W2[c1kp1] = 1
        U_cols.append(U2); V_cols.append(V2); W_cols.append(W2)

        # M3 = A00(B0kp1-B1k)
        U3 = [1, 0, 0, 0]
        V3 = [0]*26; V3[b0kp1] = 1; V3[b1k] = -1
        W3 = [0]*26; W3[c0kp1] = 1; W3[c1kp1] = 1
        U_cols.append(U3); V_cols.append(V3); W_cols.append(W3)

        # M4 = A11(B1k-B0k)
        U4 = [0, 0, 0, 1]
        V4 = [0]*26; V4[b1k] = 1; V4[b0k] = -1
        W4 = [0]*26; W4[c0k] = 1; W4[c1k] = 1
        U_cols.append(U4); V_cols.append(V4); W_cols.append(W4)

        # M5 = (A00+A01)B1kp1
        U5 = [1, 1, 0, 0]
        V5 = [0]*26; V5[b1kp1] = 1
        W5 = [0]*26; W5[c0k] = -1; W5[c0kp1] = 1
        U_cols.append(U5); V_cols.append(V5); W_cols.append(W5)

        # M6 = (A10-A00)(B0k+B0kp1)
        U6 = [1, 0, -1, 0]
        V6 = [0]*26; V6[b0k] = 1; V6[b0kp1] = 1
        W6 = [0]*26; W6[c1kp1] = 1
        U_cols.append(U6); V_cols.append(V6); W_cols.append(W6)

        # M7 = (A01-A11)(B1k+B1kp1)
        U7 = [0, 1, 0, -1]
        V7 = [0]*26; V7[b1k] = 1; V7[b1kp1] = 1
        W7 = [0]*26; W7[c0k] = 1
        U_cols.append(U7); V_cols.append(V7); W_cols.append(W7)

    # Schoolbook for single column k=12
    def add_schoolbook_column(k_base):
        """Add schoolbook decomposition for single column k_base."""
        b0k = k_base
        b1k = 13 + k_base
        c0k = 2 * k_base
        c1k = 2 * k_base + 1

        # C0k = A00*B0k
        U1 = [1, 0, 0, 0]
        V1 = [0]*26; V1[b0k] = 1
        W1 = [0]*26; W1[c0k] = 1
        U_cols.append(U1); V_cols.append(V1); W_cols.append(W1)

        # C0k += A01*B1k
        U2 = [0, 1, 0, 0]
        V2 = [0]*26; V2[b1k] = 1
        W2 = [0]*26; W2[c0k] = 1
        U_cols.append(U2); V_cols.append(V2); W_cols.append(W2)

        # C1k = A10*B0k
        U3 = [0, 0, 1, 0]
        V3 = [0]*26; V3[b0k] = 1
        W3 = [0]*26; W3[c1k] = 1
        U_cols.append(U3); V_cols.append(V3); W_cols.append(W3)

        # C1k += A11*B1k
        U4 = [0, 0, 0, 1]
        V4 = [0]*26; V4[b1k] = 1
        W4 = [0]*26; W4[c1k] = 1
        U_cols.append(U4); V_cols.append(V4); W_cols.append(W4)

    # Add 6 Strassen blocks for k=0,2,4,6,8,10
    for k in range(0, 12, 2):
        add_strassen_block(k)

    # Add 1 schoolbook column for k=12
    add_schoolbook_column(12)

    rank = len(U_cols)

    # Convert to lists
    U = [U_cols[r][row] for row in range(4) for r in range(rank)]  # wrong, need U[row][r]
    # Fix: U is (4, rank), so U[row][col]
    U = [[U_cols[r][row] for r in range(rank)] for row in range(4)]
    V = [[V_cols[r][row] for r in range(rank)] for row in range(26)]
    W = [[W_cols[r][row] for r in range(rank)] for row in range(26)]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
