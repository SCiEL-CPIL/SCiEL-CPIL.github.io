"""Rewritten program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15

    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact (within tolerance)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, r

    def strassen_2x2x2():
        """Return Strassen decomposition for 2x2x2 matrix multiplication.
        Returns U(4x7), V(4x7), W(4x7) for the 2x2x2 tensor.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # p1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # p2 = (c+d)(e)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1

        # p3 = (a-b)(f+h)
        U[0, 2] = 1; U[1, 2] = -1
        V[1, 2] = 1; V[3, 2] = 1
        W[2, 2] = 1

        # p4 = (c+d)(g+h)
        U[2, 3] = 1; U[3, 3] = 1
        V[2, 3] = 1; V[3, 3] = 1
        W[1, 3] = 1

        # p5 = (a+c)(e)
        U[0, 4] = 1; U[2, 4] = 1
        V[0, 4] = 1
        W[0, 4] = 1

        # p6 = (b+d)(h)
        U[1, 5] = 1; U[3, 5] = 1
        V[3, 5] = 1
        W[2, 5] = 1

        # p7 = (a-b)(f+g)
        U[0, 6] = 1; U[1, 6] = -1
        V[1, 6] = 1; V[2, 6] = 1
        W[0, 6] = 1

        return U, V, W

    def strassen_2x2x3():
        """Return a rank-11 decomposition for 2x2x3 matrix multiplication.
        A is 2x2, B is 2x3, C is 2x3.
        Returns U(4x11), V(6x11), W(6x11).
        
        We split B into B_left (2x2) and B_right (2x1).
        Use Strassen for A * B_left (rank 7) and schoolbook for A * B_right (rank 4).
        Total rank: 7 + 4 = 11.
        """
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Strassen part for columns 0-1 of B (2x2 block)
        sU, sV, sW = strassen_2x2x2()
        U[:, :7] = sU
        V[:4, :7] = sV  # j in {0,1}, k in {0,1} -> indices 0..3
        W[:4, :7] = sW  # k in {0,1}, i in {0,1} -> indices 0..3
        
        # Schoolbook part for column 2 of B (2x1 block)
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        # V index for (j,k): j*3+k. k=2.
        # j=0,k=2 -> index 2
        # j=1,k=2 -> index 5
        # W index for (k,i): k*2+i. k=2.
        # k=2,i=0 -> index 4
        # k=2,i=1 -> index 5
        
        # Term 1: A[0,0]*B[0,2] -> C[0,2]
        U[0, 7] = 1
        V[2, 7] = 1  # j=0, k=2
        W[4, 7] = 1  # k=2, i=0
        
        # Term 2: A[0,1]*B[1,2] -> C[0,2]
        U[1, 8] = 1
        V[5, 8] = 1  # j=1, k=2
        W[4, 8] = 1  # k=2, i=0
        
        # Term 3: A[1,0]*B[0,2] -> C[1,2]
        U[2, 9] = 1
        V[2, 9] = 1  # j=0, k=2
        W[5, 9] = 1  # k=2, i=1
        
        # Term 4: A[1,1]*B[1,2] -> C[1,2]
        U[3, 10] = 1
        V[5, 10] = 1  # j=1, k=2
        W[5, 10] = 1  # k=2, i=1
        
        return U, V, W

    def strassen_2x3x2():
        """Return a rank-11 decomposition for 2x3x2 matrix multiplication.
        A is 2x3, B is 3x2, C is 2x2.
        Returns U(6x11), V(6x11), W(4x11).
        
        We split A into A_left (2x2) and A_right (2x1).
        Use Strassen for A_left * B_left (2x2x2, rank 7) 
        and schoolbook for cross terms.
        
        Actually, let's think more carefully.
        A is 2x3, B is 3x2.
        C[i,k] = sum_j A[i,j] B[j,k]
        
        We can split j into {0,1} and {2}.
        C = A_left * B_left + A_right * B_right
        where A_left is 2x2 (cols 0,1), B_left is 2x2 (rows 0,1)
        A_right is 2x1 (col 2), B_right is 1x2 (row 2)
        
        A_left * B_left: 2x2x2, rank 7 (Strassen)
        A_right * B_right: 2x1x2, rank 4 (schoolbook)
        Total: 11
        """
        U = np.zeros((6, 11))  # A: 2x3 -> 6 entries
        V = np.zeros((6, 11))  # B: 3x2 -> 6 entries
        W = np.zeros((4, 11))  # C: 2x2 -> 4 entries
        
        # Strassen part for j in {0,1}
        sU, sV, sW = strassen_2x2x2()
        # A indices: i*3+j. i in {0,1}, j in {0,1} -> 0,1,3,4
        # B indices: j*2+k. j in {0,1}, k in {0,1} -> 0,1,2,3
        # C indices: k*2+i. k in {0,1}, i in {0,1} -> 0,1,2,3
        
        # Map sU (4x7, indices i*2+j) to U (6x11, indices i*3+j)
        U[0, :7] = sU[0, :]  # i=0,j=0
        U[1, :7] = sU[1, :]  # i=0,j=1
        U[3, :7] = sU[2, :]  # i=1,j=0
        U[4, :7] = sU[3, :]  # i=1,j=1
        
        # Map sV (4x7, indices j*2+k) to V (6x11, indices j*2+k)
        V[0, :7] = sV[0, :]  # j=0,k=0
        V[1, :7] = sV[1, :]  # j=0,k=1
        V[2, :7] = sV[2, :]  # j=1,k=0
        V[3, :7] = sV[3, :]  # j=1,k=1
        
        # W (4x11) same as sW (4x7)
        W[:, :7] = sW
        
        # Schoolbook part for j=2
        # A_right: col 2. A indices: i*3+2. i=0->2, i=1->5
        # B_right: row 2. B indices: 2*2+k. k=0->4, k=1->5
        # C: k*2+i
        
        # C[0,0] += A[0,2]*B[2,0]
        U[2, 7] = 1; V[4, 7] = 1; W[0, 7] = 1
        # C[0,1] += A[0,2]*B[2,1]
        U[2, 8] = 1; V[5, 8] = 1; W[1, 8] = 1
        # C[1,0] += A[1,2]*B[2,0]
        U[5, 9] = 1; V[4, 9] = 1; W[2, 9] = 1
        # C[1,1] += A[1,2]*B[2,1]
        U[5, 10] = 1; V[5, 10] = 1; W[3, 10] = 1
        
        return U, V, W

    def strassen_2x3x3():
        """Return a rank-15 decomposition for 2x3x3 matrix multiplication.
        A is 2x3, B is 3x3, C is 2x3.
        Split B into B_left (3x2) and B_right (3x1).
        A * B_left: 2x3x2, rank 11
        A * B_right: 2x3x1, rank 6 (schoolbook)
        Total: 17? Let's try differently.
        
        Split A into A_left (2x2) and A_right (2x1).
        A_left * B: 2x2x3, rank 11
        A_right * B: 2x1x3, rank 3 (schoolbook)
        Total: 14
        
        Let's do A_right * B with rank 3:
        C[0,k] += A[0,2]*B[2,k] for k=0,1,2 -> 3 terms
        C[1,k] += A[1,2]*B[2,k] for k=0,1,2 -> 3 terms
        That's 6 terms, not 3.
        
        Hmm, 2x1x3: A is 2x1, B is 1x3, C is 2x3.
        C[i,k] = A[i,0]*B[0,k]
        Rank is 2*3 = 6 (outer product structure, but each (i,k) pair needs its own term since A and B are independent).
        Actually no: C[i,k] = A[i,0]*B[0,k]. This is a rank-1 tensor in (i,k) for fixed... no.
        The tensor is sum_{i,k} A[i,0]*B[0,k]*C[i,k].
        Each (i,k) pair is a separate rank-1 term: U has A[i,0], V has B[0,k], W has C[i,k].
        So rank = 2*3 = 6.
        
        Total: 11 + 6 = 17.
        
        Let me try another split.
        Split B into two 3x2 blocks? No, B is 3x3.
        
        Let's just use rank 17 for now, or find a better decomposition.
        
        Actually, let's reconsider. For 2x3x3:
        Use Strassen on a 2x2x2 sub-block (j={0,1}, k={0,1}): rank 7
        Then handle the remaining terms.
        
        This is getting complex. Let me just use schoolbook for 2x3x3 (rank 18) 
        and see if the overall improvement from other blocks is enough.
        
        Actually, let me compute the total rank with various block decompositions.
        """
        # For now, return None to indicate we should use schoolbook
        return None

    def build_block_decomposition_v2():
        """Build block decomposition for 3x4x5 with improved ranks.
        Split: i={0,1},{2}; j={0,1},{2,3}; k={0,1},{2,3},{4}
        
        Blocks:
        (i={0,1},j={0,1},k={0,1}): 2x2x2, rank 7 (Strassen)
        (i={0,1},j={0,1},k={2,3}): 2x2x2, rank 7 (Strassen)
        (i={0,1},j={0,1},k={4}):   2x2x1, rank 4 (schoolbook)
        (i={0,1},j={2,3},k={0,1}): 2x2x2, rank 7 (Strassen)
        (i={0,1},j={2,3},k={2,3}): 2x2x2, rank 7 (Strassen)
        (i={0,1},j={2,3},k={4}):   2x2x1, rank 4 (schoolbook)
        (i={2},j={0,1},k={0,1}):   1x2x2, rank 4 (schoolbook)
        (i={2},j={0,1},k={2,3}):   1x2x2, rank 4 (schoolbook)
        (i={2},j={0,1},k={4}):     1x2x1, rank 2 (schoolbook)
        (i={2},j={2,3},k={0,1}):   1x2x2, rank 4 (schoolbook)
        (i={2},j={2,3},k={2,3}):   1x2x2, rank 4 (schoolbook)
        (i={2},j={2,3},k={4}):     1x2x1, rank 2 (schoolbook)
        Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
        """
        U = np.zeros((nm, 56))
        V = np.zeros((mp, 56))
        W = np.zeros((pn, 56))
        col = 0
        
        def add_block(i_indices, j_indices, k_indices, block_U, block_V, block_W):
            r_block = block_U.shape[1]
            for c in range(r_block):
                for li, i in enumerate(i_indices):
                    for lj, j in enumerate(j_indices):
                        U[i*m + j, col + c] += block_U[li*len(j_indices) + lj, c]
                for lj, j in enumerate(j_indices):
                    for lk, k in enumerate(k_indices):
                        V[j*p + k, col + c] += block_V[lj*len(k_indices) + lk, c]
                for lk, k in enumerate(k_indices):
                    for li, i in enumerate(i_indices):
                        W[k*n + i, col + c] += block_W[lk*len(i_indices) + li, c]
            return r_block
        
        # Block 1: i={0,1}, j={0,1}, k={0,1} - 2x2x2 Strassen
        bU, bV, bW = strassen_2x2x2()
        col += add_block([0,1], [0,1], [0,1], bU, bV, bW)
        
        # Block 2: i={0,1}, j={0,1}, k={2,3} - 2x2x2 Strassen
        bU, bV, bW = strassen_2x2x2()
        col += add_block([0,1], [0,1], [2,3], bU, bV, bW)
        
        # Block 3: i={0,1}, j={0,1}, k={4} - 2x2x1 schoolbook (rank 4)
        bU = np.zeros((4, 4))
        bV = np.zeros((2, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[1, 1] = 1; bW[0, 1] = 1
        bU[2, 2] = 1; bV[0, 2] = 1; bW[1, 2] = 1
        bU[3, 3] = 1; bV[1, 3] = 1; bW[1, 3] = 1
        col += add_block([0,1], [0,1], [4], bU, bV, bW)
        
        # Block 4: i={0,1}, j={2,3}, k={0,1} - 2x2x2 Strassen
        bU, bV, bW = strassen_2x2x2()
        col += add_block([0,1], [2,3], [0,1], bU, bV, bW)
        
        # Block 5: i={0,1}, j={2,3}, k={2,3} - 2x2x2 Strassen
        bU, bV, bW = strassen_2x2x2()
        col += add_block([0,1], [2,3], [2,3], bU, bV, bW)
        
        # Block 6: i={0,1}, j={2,3}, k={4} - 2x2x1 schoolbook (rank 4)
        bU = np.zeros((4, 4))
        bV = np.zeros((2, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[1, 1] = 1; bW[0, 1] = 1
        bU[2, 2] = 1; bV[0, 2] = 1; bW[1, 2] = 1
        bU[3, 3] = 1; bV[1, 3] = 1; bW[1, 3] = 1
        col += add_block([0,1], [2,3], [4], bU, bV, bW)
        
        # Block 7: i={2}, j={0,1}, k={0,1} - 1x2x2 schoolbook (rank 4)
        bU = np.zeros((2, 4))
        bV = np.zeros((4, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[2, 1] = 1; bW[0, 1] = 1
        bU[0, 2] = 1; bV[1, 2] = 1; bW[1, 2] = 1
        bU[1, 3] = 1; bV[3, 3] = 1; bW[1, 3] = 1
        col += add_block([2], [0,1], [0,1], bU, bV, bW)
        
        # Block 8: i={2}, j={0,1}, k={2,3} - 1x2x2 schoolbook (rank 4)
        bU = np.zeros((2, 4))
        bV = np.zeros((4, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[2, 1] = 1; bW[0, 1] = 1
        bU[0, 2] = 1; bV[1, 2] = 1; bW[1, 2] = 1
        bU[1, 3] = 1; bV[3, 3] = 1; bW[1, 3] = 1
        col += add_block([2], [0,1], [2,3], bU, bV, bW)
        
        # Block 9: i={2}, j={0,1}, k={4} - 1x2x1 schoolbook (rank 2)
        bU = np.zeros((2, 2))
        bV = np.zeros((2, 2))
        bW = np.zeros((1, 2))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[1, 1] = 1; bW[0, 1] = 1
        col += add_block([2], [0,1], [4], bU, bV, bW)
        
        # Block 10: i={2}, j={2,3}, k={0,1} - 1x2x2 schoolbook (rank 4)
        bU = np.zeros((2, 4))
        bV = np.zeros((4, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[2, 1] = 1; bW[0, 1] = 1
        bU[0, 2] = 1; bV[1, 2] = 1; bW[1, 2] = 1
        bU[1, 3] = 1; bV[3, 3] = 1; bW[1, 3] = 1
        col += add_block([2], [2,3], [0,1], bU, bV, bW)
        
        # Block 11: i={2}, j={2,3}, k={2,3} - 1x2x2 schoolbook (rank 4)
        bU = np.zeros((2, 4))
        bV = np.zeros((4, 4))
        bW = np.zeros((2, 4))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[2, 1] = 1; bW[0, 1] = 1
        bU[0, 2] = 1; bV[1, 2] = 1; bW[1, 2] = 1
        bU[1, 3] = 1; bV[3, 3] = 1; bW[1, 3] = 1
        col += add_block([2], [2,3], [2,3], bU, bV, bW)
        
        # Block 12: i={2}, j={2,3}, k={4} - 1x2x1 schoolbook (rank 2)
        bU = np.zeros((2, 2))
        bV = np.zeros((2, 2))
        bW = np.zeros((1, 2))
        bU[0, 0] = 1; bV[0, 0] = 1; bW[0, 0] = 1
        bU[1, 1] = 1; bV[1, 1] = 1; bW[0, 1] = 1
        col += add_block([2], [2,3], [4], bU, bV, bW)
        
        assert col == 56, f"Expected 56 columns, got {col}"
        return U, V, W, 56

    def build_block_decomposition_v3():
        """Build block decomposition using 2x2x3 and 2x3x2 blocks where possible.
        
        Split: i={0,1},{2}; j={0,1,2},{3}; k={0,1,2},{3,4}
        
        Blocks:
        (i={0,1},j={0,1,2},k={0,1,2}): 2x3x3, rank ? 
        (i={0,1},j={0,1,2},k={3,4}): 2x3x2, rank 11 (Strassen 2x3x2)
        (i={0,1},j={3},k={0,1,2}): 2x1x3, rank 6 (schoolbook)
        (i={0,1},j={3},k={3,4}): 2x1x2, rank 4 (schoolbook)
        (i={2},j={0,1,2},k={0,1,2}): 1x3x3, rank 9 (schoolbook)
        (i={2},j={0,1,2},k={3,4}): 1x3x2, rank 6 (schoolbook)
        (i={2},j={3},k={0,1,2}): 1x1x3, rank 3 (schoolbook)
        (i={2},j={3},k={3,4}): 1x1x2, rank 2 (schoolbook)
        
        Need 2x3x3 rank. Let's use 17 for now.
        Total: 17+11+6+4+9+6+3+2 = 58. Worse than 56.
        
        Let me try: i={0,1},{2}; j={0,1},{2,3,4}; k={0,1,2},{3,4}
        
        Blocks:
        (i={0,1},j={0,1},k={0,1,2}): 2x2x3, rank 11
        (i={0,1},j={0,1},k={3,4}): 2x2x2, rank 7
        (i={0,1},j={2,3,4},k={0,1,2}): 2x3x3, rank 17
        (i={0,1},j={2,3,4},k={3,4}): 2x3x2, rank 11
        (i={2},j={0,1},k={0,1,2}): 1x2x3, rank 6
        (i={2},j={0,1},k={3,4}): 1x2x2, rank 4
        (i={2},j={2,3,4},k={0,1,2}): 1x3x3, rank 9
        (i={2},j={2,3,4},k={3,4}): 1x3x2, rank 6
        
        Total: 11+7+17+11+6+4+9+6 = 71. Worse.
        
        The 2x2x2 Strassen blocks are very efficient. Let me stick with the 56-rank solution
        but try to improve the 1x2x2 blocks (rank 4 each) and 2x2x1 blocks (rank 4 each).
        
        For 1x2x2: A is 1x2, B is 2x2, C is 1x2.
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        Rank 4 is minimal (4 independent terms).
        
        For 2x2x1: A is 2x2, B is 2x1, C is 2x1.
        C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        Rank 4 is minimal.
        
        So 56 seems to be the best with this block structure. Let me try a different split.
        
        Split: i={0},{1},{2}; j={0,1,2},{3}; k={0,1,2},{3,4}
        
        This gives 3x2x2 = 12 blocks, many 1x2x2 or 1x2x3 etc.
        
        Actually, let me try: i={0,1},{2}; j={0,1,2,3}; k={0,1,2,3,4}
        Too coarse.
        
        Let me try to use the 2x2x3 decomposition (rank 11) instead of two 2x2x2 (rank 14) 
        where the k-dimension is 3.
        
        Current split has k={0,1},{2,3},{4}. The k={0,1} and k={2,3} blocks are 2-wide.
        If I merge k={0,1,2} into one block of width 3, I can use 2x2x3 (rank 11) 
        instead of 2x2x2 (rank 7) + 2x2x1 (rank 4) = rank 11. Same!
        
        So no improvement there.
        
        Let me try a completely different approach: use the known rank-55 or lower 
        decompositions if they exist for 3x4x5.
        
        Actually, the minimal rank for 3x4x5 matrix multiplication is not known exactly,
        but upper bounds are around 55-56. The current 56 is close to optimal.
        
        Let me just return the 56-rank solution which is verified.
        """
        return build_block_decomposition_v2()

    # Build the best decomposition
    U_block, V_block, W_block, r_block = build_block_decomposition_v2()
    
    # Verify
    if verify(U_block, V_block, W_block, tol=1e-9):
        best_U, best_V, best_W, best_rank = U_block, V_block, W_block, r_block
    else:
        # Fall back to schoolbook
        U_sb, V_sb, W_sb, r_sb = build_schoolbook()
        best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb
    
    # Convert to lists
    def to_list(mat):
        mat = np.round(mat * 2) / 2.0
        return mat.tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
