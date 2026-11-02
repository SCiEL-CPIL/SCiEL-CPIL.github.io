import numpy as np
import time
from fractions import Fraction
import sys
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline (10s safety margin)

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def check_half_integer(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    # Build rank-28 decomposition (known good)
    def build_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            # M1
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M2
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = -1
            col += 1
            # M3
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M4
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = -1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M5
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            col += 1
            # M6
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (0 + j_off)][col] = -1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M7
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = -1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            col += 1

        # Block 1: j=0 (schoolbook: 2*5 = 10 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        # Block 2: j=1,2, k=0,1 (Strassen: 7 terms)
        add_strassen(0, 1, 0)

        # Block 3: j=1,2, k=2 (schoolbook: 2*2 = 4 terms)
        for i in range(n):
            for j in [1, 2]:
                U[i * m + j][col] = 1
                V[j * p + 2][col] = 1
                W[2 * n + i][col] = 1
                col += 1

        # Block 4: j=1,2, k=3,4 (Strassen: 7 terms)
        add_strassen(0, 1, 3)

        assert col == rank, f"Column count mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    def try_rank_27_structured():
        """
        Structured rank-27 search:
        - Fix Strassen blocks for j∈{1,2}×k∈{0,1} (7 terms) and j∈{1,2}×k∈{3,4} (7 terms) = 14 terms
        - The remaining sub-tensor involves:
          * j=0, all k (10 terms in schoolbook)
          * j∈{1,2}, k=2 (4 terms in schoolbook)
          Total schoolbook for remainder = 14 terms
        - We need to cover the remainder in 27-14 = 13 terms
        - Strategy: set up the 14 fixed Strassen columns, then solve a linear system
          for 13 additional columns that handle the remaining tensor entries.
        """
        rank = 27
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        # Add Strassen block for j=1,2 and k=0,1
        def add_strassen_block(j_off, k_off):
            nonlocal col
            # 7 Strassen terms for 2x2x2 (i=0,1; j=j_off,j_off+1; k=k_off,k_off+1)
            # M1: (A00+A11)(B00+B11)
            U[0 * m + j_off][col] = 1; U[1 * m + j_off + 1][col] = 1
            V[j_off * p + k_off][col] = 1; V[j_off + 1 * p + k_off + 1][col] = 1
            W[k_off * n + 0][col] = 1; W[k_off + 1 * n + 1][col] = 1
            col += 1
            # M2: (A10+A11)(B00)
            U[1 * m + j_off][col] = 1; U[1 * m + j_off + 1][col] = 1
            V[j_off * p + k_off][col] = 1
            W[k_off * n + 1][col] = 1; W[k_off + 1 * n + 1][col] = -1
            col += 1
            # M3: (A00)(B11 - B01)
            U[0 * m + j_off][col] = 1
            V[j_off * p + k_off + 1][col] = 1; V[j_off + 1 * p + k_off + 1][col] = -1
            W[k_off + 1 * n + 0][col] = 1; W[k_off + 1 * n + 1][col] = 1
            col += 1
            # M4: (A11)(B00 - B10)
            U[1 * m + j_off + 1][col] = 1
            V[j_off + 1 * p + k_off][col] = 1; V[j_off * p + k_off][col] = -1
            W[k_off * n + 0][col] = 1; W[k_off * n + 1][col] = 1
            col += 1
            # M5: (A00+A01)(B11)
            U[0 * m + j_off][col] = 1; U[0 * m + j_off + 1][col] = 1
            V[j_off + 1 * p + k_off + 1][col] = 1
            W[k_off * n + 0][col] = -1; W[k_off + 1 * n + 0][col] = 1
            col += 1
            # M6: (A10-A00)(B00+B01)
            U[1 * m + j_off][col] = 1; U[0 * m + j_off][col] = -1
            V[j_off * p + k_off][col] = 1; V[j_off * p + k_off + 1][col] = 1
            W[k_off + 1 * n + 1][col] = 1
            col += 1
            # M7: (A01-A11)(B10+B11)
            U[0 * m + j_off + 1][col] = 1; U[1 * m + j_off + 1][col] = -1
            V[j_off + 1 * p + k_off][col] = 1; V[j_off + 1 * p + k_off + 1][col] = 1
            W[k_off * n + 0][col] = 1
            col += 1
        
        add_strassen_block(1, 0)  # j=1,2 k=0,1 -> 7 cols
        add_strassen_block(1, 3)  # j=1,2 k=3,4 -> 7 cols
        # col should be 14 now
        
        remaining_rank = rank - col  # 27 - 14 = 13
        
        # The remaining tensor entries to cover:
        # For each (i,j,k) not already covered by the Strassen blocks:
        # The Strassen blocks cover: j∈{1,2}, k∈{0,1,3,4} for all i∈{0,1}
        # Remaining: j=0 all k (10 entries) + j∈{1,2} k=2 (4 entries) = 14 entries
        # But we only have 13 columns. We need cross-term cancellations.
        
        # Build the linear system:
        # For each remaining (i,j,k) entry, we need:
        #   sum_{r=14}^{26} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # (since the Strassen blocks contribute 0 to these entries by construction)
        
        # Actually, let me verify: the Strassen blocks for j=1,2 k=0,1 only have
        # non-zero U entries at rows (0*m+1),(1*m+1),(0*m+2),(1*m+2) = rows 1,2,4,5
        # and V entries at rows (1*p+0),(1*p+1),(2*p+0),(2*p+1) = rows 5,6,10,11
        # and W entries at rows (0*n+0),(0*n+1),(1*n+0),(1*n+1) = rows 0,1,2,3
        # For the second block (j=1,2 k=3,4):
        # V rows: (1*p+3),(1*p+4),(2*p+3),(2*p+4) = rows 8,9,13,14
        # W rows: (3*n+0),(3*n+1),(4*n+0),(4*n+1) = rows 6,7,8,9
        
        # So for entry (i=0,j=0,k=0): U[0,·] is zero in cols 0-13, V[0,·] is zero, W[0,·] has entries
        # Actually U[0,·] for cols 0-13: row 0 is i=0,j=0 -> U[0*m+0]=U[0]. In Strassen block 1,
        # U entries are at rows 1,2,4,5. So U[0,0:14]=0. Good.
        # V[0*p+0]=V[0]. In Strassen block 1, V entries are at rows 5,6,10,11. So V[0,0:14]=0. Good.
        # So the contribution from Strassen to (0,0,0) is 0. 
        
        # Similarly for all j=0 entries and j∈{1,2},k=2 entries, the Strassen blocks contribute 0.
        # So we need: sum_{r=14}^{26} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for each of the 14 entries.
        
        # This is a bilinear system in the 13 unknown columns.
        # Let's parameterize: for each r in 14..26, let u_r = U[:,r], v_r = V[:,r], w_r = W[:,r]
        # The constraint for entry (i,j,k) is: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
        
        # This is still nonlinear. Let's try a different approach:
        # Fix the structure of the 13 remaining columns.
        
        # Approach: Use the fact that j=0 entries form a 2x5 matrix multiplication (rank 10 schoolbook)
        # and j=1,2 k=2 entries form a 2x2 matrix (rank 4 schoolbook).
        # Total = 14. We need 13. So we need 1 cross-term saving.
        
        # Idea: One of the 13 columns can serve double duty for a j=0 entry and a j∈{1,2},k=2 entry
        # if it has non-zero entries in both U[i*m+0] and U[i*m+1] or U[i*m+2],
        # AND in both V[0*p+k] and V[j'*p+2],
        # AND in both W[k*n+i] and W[2*n+i'].
        
        # Let's try: a column that covers (i=0,j=0,k=0) and (i=1,j=1,k=2) simultaneously.
        # U[0, r] = a, U[3, r] = b  (rows for (0,0) and (1,1))
        # V[0, r] = c, V[7, r] = d  (rows for (0,0) and (1,2))  
        # W[0, r] = e, W[2*n+1]=W[3, r] = f  (rows for (0,0) and (2,1))
        # Contribution to (0,0,0): a*c*e
        # Contribution to (1,1,2): b*d*f
        # We need both to be 1, so a*c*e=1 and b*d*f=1.
        # But this column also contributes to other entries!
        # (0,0,0) gets a*c*e, but also (0,1,2) gets U[1,r]*V[7,r]*W[2,r]... wait U[1,r] might be 0.
        
        # This is getting complex. Let me try a direct numerical approach:
        # Set up 13 columns with specific sparse patterns and solve.
        
        # Alternative: Try rank 27 by allowing the j=0 block to use Strassen-like tricks.
        # j=0 is a 2x1x5 problem (2 rows, 1 col of A, 5 cols of B).
        # This is just C[i,k] = A[i,0]*B[0,k], which is rank 10 (can't do better for 2x5).
        # Unless we combine with the k=2 column of j=1,2.
        
        # Let me try: 10 columns for j=0 (schoolbook) + 3 columns for j∈{1,2},k=2.
        # For j∈{1,2},k=2: C[i,2] = A[i,1]*B[1,2] + A[i,2]*B[2,2]
        # This is a 2x2x1 problem. Rank is at most 4 (schoolbook). Can we do it in 3?
        # C[0,2] = A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,2] = A[1,1]*B[1,2] + A[1,2]*B[2,2]
        # This is the dot product of two 2-vectors, which has tensor rank 2 (not 4!)
        # Wait no, it's 2x2x1 which is just a 2x2 matrix times a vector... 
        # Actually the tensor is T[i,j,k=2] = delta for the multiplication.
        # The 2x2x1 tensor: T[i,j,0] = 1 for all i,j. This is a rank-2 tensor (outer product of ones).
        # So C[0,2] = A[0,1]*B[1,2] + A[0,2]*B[2,2] needs rank 2 (not 4).
        # Similarly C[1,2] = A[1,1]*B[1,2] + A[1,2]*B[2,2] needs rank 2.
        # But they share the same B[1,2] and B[2,2]!
        # So we can do: 
        #   col_a: U[0*m+1]=1, U[1*m+1]=1, V[1*p+2]=1, W[2*n+0]=1, W[2*n+1]=1
        #   col_b: U[0*m+2]=1, U[1*m+2]=1, V[2*p+2]=1, W[2*n+0]=1, W[2*n+1]=1
        # That's only 2 columns for all 4 entries! 
        # Total: 7 + 7 + 10 + 2 = 26!
        
        # Wait, let me recheck. For the j=1,2 k=2 block:
        # We need C[i,2] = A[i,1]*B[1,2] + A[i,2]*B[2,2] for i=0,1.
        # Column a: U[1,·]=1 (i=0,j=1), U[4,·]=1 (i=1,j=1), V[7,·]=1 (j=1,k=2), W[2,·]=1 (k=2,i=0), W[3,·]=1 (k=2,i=1)
        #   Contribution to (0,1,2): U[1]*V[7]*W[2] = 1*1*1 = 1 ✓
        #   Contribution to (1,1,2): U[4]*V[7]*W[3] = 1*1*1 = 1 ✓
        #   Other contributions: U[1] is row i=0,j=1. V[7] is j=1,k=2. W[2] is k=2,i=0.
        #   Does this column contribute to any other (i,j,k)? 
        #   It has U non-zero at rows 1 and 4 (j=1), V non-zero at row 7 (j=1,k=2), W non-zero at rows 2,3 (k=2).
        #   For entry (i',j',k'): need U[i'*m+j']≠0 AND V[j'*p+k']≠0 AND W[k'*n+i']≠0.
        #   U non-zero: j'=1 (rows 1,4). V non-zero: j'=1,k'=2 (row 7). W non-zero: k'=2 (rows 2,3).
        #   So only entries with j'=1, k'=2: (0,1,2) and (1,1,2). ✓
        
        # Column b: U[2,·]=1 (i=0,j=2), U[5,·]=1 (i=1,j=2), V[12,·]=1 (j=2,k=2), W[2,·]=1 (k=2,i=0), W[3,·]=1 (k=2,i=1)
        #   Contribution to (0,2,2): 1 ✓, (1,2,2): 1 ✓
        #   Only contributes to j'=2, k'=2 entries. ✓
        
        # So total rank = 7 + 7 + 10 + 2 = 26!
        
        # Let me build this rank-26 decomposition:
        col = 0
        U = np.zeros((n * m, 26))
        V = np.zeros((m * p, 26))
        W = np.zeros((p * n, 26))
        
        # Strassen block 1: j=1,2 k=0,1
        add_strassen_block(1, 0)
        # Strassen block 2: j=1,2 k=3,4
        add_strassen_block(1, 3)
        # col = 14
        
        # Schoolbook for j=0: 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        # col = 24
        
        # 2 columns for j=1,2 k=2
        # Column 24: handles (i,1,2) for i=0,1
        U[0 * m + 1][col] = 1  # i=0, j=1
        U[1 * m + 1][col] = 1  # i=1, j=1
        V[1 * p + 2][col] = 1  # j=1, k=2
        W[2 * n + 0][col] = 1  # k=2, i=0
        W[2 * n + 1][col] = 1  # k=2, i=1
        col += 1
        # Column 25: handles (i,2,2) for i=0,1
        U[0 * m + 2][col] = 1  # i=0, j=2
        U[1 * m + 2][col] = 1  # i=1, j=2
        V[2 * p + 2][col] = 1  # j=2, k=2
        W[2 * n + 0][col] = 1  # k=2, i=0
        W[2 * n + 1][col] = 1  # k=2, i=1
        col += 1
        
        assert col == 26, f"Expected 26 cols, got {col}"
        
        # Verify
        if verify(U.tolist(), V.tolist(), W.tolist(), 26):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try structured rank-26
    if time_left() > 30:
        res_26 = try_rank_27_structured()
        if res_26:
            if verify(res_26["U"], res_26["V"], res_26["W"], 26):
                return res_26

    # Fallback to rank 28
    res_28 = build_rank_28()
    if verify(res_28["U"], res_28["V"], res_28["W"], 28):
        return res_28

    # Ultimate fallback: schoolbook rank 30
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
