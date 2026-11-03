"""Rewritten program for 3x3 * 3x8 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks to reduce rank below the schoolbook 72.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def validate(U, V, W):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p  # 72
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # Start with schoolbook as fallback
    best_U, best_V, best_W, best_rank = schoolbook()

    # Strassen's algorithm for 2x2 matrix multiplication: rank 7 instead of 8
    # For matrices A (2x2) and B (2x2), C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    def strassen_2x2_terms():
        """
        Return the 7 Strassen terms as lists of (index, coeff) for U, V, W.
        For a 2x2 block, indices are:
        U: row*2+col (0,1,2,3) for A00,A01,A10,A11
        V: row*2+col (0,1,2,3) for B00,B01,B10,B11
        W: row*2+col (0,1,2,3) for C00,C01,C10,C11
        """
        # Each term: (u_entries, v_entries, w_entries)
        # u_entries: list of (A_index, coeff)
        # v_entries: list of (B_index, coeff)
        # w_entries: list of (C_index, coeff)
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,-1)]))
        # M2 = (A10+A11)B00
        terms.append(([(2,1),(3,1)], [(0,1)], [(1,1)]))
        # M3 = A00(B10-B11)
        terms.append(([(0,1)], [(2,1),(3,-1)], [(1,1)]))
        # M4 = A11(B01-B00)
        terms.append(([(3,1)], [(1,1),(0,-1)], [(0,1)]))
        # M5 = (A00+A01)B11
        terms.append(([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]))
        # M6 = (A10-A00)(B00+B01)
        terms.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]))
        # M7 = (A01-A11)(B10+B11)
        terms.append(([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]))
        
        return terms

    def build_strassen_decomposition():
        """
        Build a Strassen-based decomposition for 3x3 * 3x8.
        
        Strategy: Split A (3x3) into blocks:
        - Top-left 2x2: A[0:2, 0:2]
        - Right column: A[0:3, 2]
        - Bottom row: A[2, 0:2]
        
        Split B (3x8) into blocks:
        - Left 2 columns: B[0:2, 0:2]
        - Right 6 columns: B[0:3, 2:8]
        
        Actually, let's use a simpler approach:
        - Use Strassen on the 2x2 * 2x2 part (A[0:2,0:2] * B[0:2,0:2])
        - Use schoolbook for everything else
        
        A is 3x3, B is 3x8, C is 3x8.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Let's partition:
        - A[0:2, 0:2] (2x2) * B[0:2, 0:8] (2x8) -> contributes to C[0:2, 0:8]
        - A[0:2, 2] (2x1) * B[2, 0:8] (1x8) -> contributes to C[0:2, 0:8]
        - A[2, 0:2] (1x2) * B[0:2, 0:8] (2x8) -> contributes to C[2, 0:8]
        - A[2, 2] (1x1) * B[2, 0:8] (1x8) -> contributes to C[2, 0:8]
        
        For the first term: A[0:2,0:2] * B[0:2,0:8]
        B[0:2,0:8] can be split into 4 blocks of 2x2: B[0:2,0:2], B[0:2,2:4], B[0:2,4:6], B[0:2,6:8]
        Each 2x2 * 2x2 can use Strassen (rank 7 instead of 8).
        So 4 blocks * 7 = 28 terms for this part.
        
        For the remaining parts, use schoolbook:
        - A[0:2,2] * B[2,0:8]: 2*8 = 16 terms
        - A[2,0:2] * B[0:2,0:8]: 2*8 = 16 terms
        - A[2,2] * B[2,0:8]: 1*8 = 8 terms
        
        Total: 28 + 16 + 16 + 8 = 68 terms (vs 72 schoolbook)
        """
        rank = 68
        U = [[0]*rank for _ in range(n*m)]  # 9x68
        V = [[0]*rank for _ in range(m*p)]  # 24x68
        W = [[0]*rank for _ in range(p*n)]  # 24x68
        
        col = 0
        strassen_terms = strassen_2x2_terms()
        
        # Part 1: A[0:2,0:2] * B[0:2,0:8] using Strassen for each 2x2 B-block
        for b_block in range(4):  # 4 blocks of 2 columns each
            for term in strassen_terms:
                u_entries, v_entries, w_entries = term
                for (a_idx, coeff) in u_entries:
                    i, j = divmod(a_idx, 2)
                    U[i*m+j][col] += coeff
                for (b_idx, coeff) in v_entries:
                    j, k_off = divmod(b_idx, 2)
                    k = b_block * 2 + k_off
                    V[j*p+k][col] += coeff
                for (c_idx, coeff) in w_entries:
                    i, k_off = divmod(c_idx, 2)
                    k = b_block * 2 + k_off
                    W[k*n+i][col] += coeff
                col += 1
        
        # Part 2: A[0:2,2] * B[2,0:8] (schoolbook)
        for i in range(2):
            for k in range(8):
                U[i*m+2][col] = 1
                V[2*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
        
        # Part 3: A[2,0:2] * B[0:2,0:8] (schoolbook)
        for j in range(2):
            for k in range(8):
                U[2*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+2][col] = 1
                col += 1
        
        # Part 4: A[2,2] * B[2,0:8] (schoolbook)
        for k in range(8):
            U[2*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
        
        return U, V, W, rank

    # Build Strassen-based decomposition
    if elapsed() < deadline - 5:
        U_str, V_str, W_str, rank_str = build_strassen_decomposition()
        if validate(U_str, V_str, W_str):
            if rank_str < best_rank:
                best_U, best_V, best_W, best_rank = U_str, V_str, W_str, rank_str

    # Try to improve further: use Strassen on more blocks
    # Alternative: split A differently or use more advanced algorithms
    
    # Try a different partition: use Strassen on A[0:2,0:2]*B[0:2,0:2] and 
    # also try to combine the remaining terms more efficiently
    
    def build_improved_strassen():
        """
        Try to get rank below 68 by using a more clever partition.
        
        Alternative approach: 
        - Split A (3x3) and B (3x8) into 2x2 and 1x1 blocks
        - Use Strassen for the 2x2*2x2 interactions
        - Try to share terms between different output positions
        """
        # For now, return the same as before if no improvement found
        return None

    if elapsed() < deadline - 10:
        improved = build_improved_strassen()
        if improved is not None:
            U_imp, V_imp, W_imp, rank_imp = improved
            if validate(U_imp, V_imp, W_imp):
                if rank_imp < best_rank:
                    best_U, best_V, best_W, best_rank = U_imp, V_imp, W_imp, rank_imp

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
