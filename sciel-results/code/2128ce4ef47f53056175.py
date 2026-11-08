"""Tensor decomposition for 3x3 by 3x7 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition.
    
    Returns dict with 'U', 'V', 'W' matrices.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Entries can be integers or half-integers (multiples of 1/2).
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Deadline: 145 seconds (5s safety margin for 150s limit)
    deadline = time.time() + 145
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p  # 63
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def schoolbook_rank():
        return n * m * p
    
    # Verify a decomposition using the trace convention W[k*n+i, r]
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        if not U or not V or not W:
            return False
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def verify_numpy(U, V, W):
        """Faster verification using numpy."""
        if not U or not V or not W:
            return False
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for k in range(p):
                w_val = W_arr[k*n+i]  # shape (rank,)
                for j in range(m):
                    u_val = U_arr[i*m+j]  # shape (rank,)
                    v_val = V_arr[j*p+k]  # shape (rank,)
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2_block(i1, i2, j1, j2, k1, k2, U, V, W, start_col):
        """
        Apply Strassen's 7-multiplication algorithm to a 2x2 * 2x2 sub-block.
        
        Parameters:
        - i1, i2: row indices of A (2 rows)
        - j1, j2: column indices of A / row indices of B (2 cols/rows)
        - k1, k2: column indices of B (2 cols)
        - U, V, W: the factor matrices (modified in place)
        - start_col: starting column index in the rank dimension
        
        Returns the number of columns used (should be 7).
        """
        # Strassen's 7 multiplications for 2x2 * 2x2:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # The 7 multiplications correspond to 7 rank-1 terms
        # Each term r has U[i,j], V[j,k], W[k,i] values
        
        # M1: (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # This contributes to C[i1,k1], C[i1,k2], C[i2,k1], C[i2,k2]
        # U values: U[i1*m+j1] += 1, U[i2*m+j2] += 1
        # V values: V[j1*p+k1] += 1, V[j2*p+k2] += 1
        # W values: W[k1*n+i1] += 1, W[k2*n+i1] += 1, W[k1*n+i2] += 1, W[k2*n+i2] += 1
        
        col = start_col
        
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # U: +1 at (i1,j1), +1 at (i2,j2)
        # V: +1 at (j1,k1), +1 at (j2,k2)
        # W: +1 at (k1,i1), +1 at (k2,i1), +1 at (k1,i2), +1 at (k2,i2)
        U[i1*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        W[k2*n+i1][col] += 1
        W[k1*n+i2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        # U: +1 at (i2,j1), +1 at (i2,j2)
        # V: +1 at (j1,k1)
        # W: +1 at (k1,i2)
        U[i2*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1
        col += 1
        
        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        # U: +1 at (i1,j1)
        # V: +1 at (j1,k2), -1 at (j2,k2)
        # W: +1 at (k2,i1)
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1
        V[j2*p+k2][col] -= 1
        W[k2*n+i1][col] += 1
        col += 1
        
        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        # U: +1 at (i2,j2)
        # V: +1 at (j2,k1), -1 at (j1,k1)
        # W: +1 at (k1,i2)
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1
        V[j1*p+k1][col] -= 1
        W[k1*n+i2][col] += 1
        col += 1
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        # U: +1 at (i1,j1), +1 at (i1,j2)
        # V: +1 at (j2,k2)
        # W: +1 at (k2,i1)
        U[i1*m+j1][col] += 1
        U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k2*n+i1][col] += 1
        col += 1
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        # U: +1 at (i2,j1), -1 at (i1,j1)
        # V: +1 at (j1,k1), +1 at (j1,k2)
        # W: +1 at (k1,i2), +1 at (k2,i2)
        U[i2*m+j1][col] += 1
        U[i1*m+j1][col] -= 1
        V[j1*p+k1][col] += 1
        V[j1*p+k2][col] += 1
        W[k1*n+i2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        # U: +1 at (i1,j2), -1 at (i2,j2)
        # V: +1 at (j2,k1), +1 at (j2,k2)
        # W: +1 at (k1,i1), +1 at (k2,i1)
        U[i1*m+j2][col] += 1
        U[i2*m+j2][col] -= 1
        V[j2*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        W[k2*n+i1][col] += 1
        col += 1
        
        return col - start_col  # Should be 7
    
    def build_strassen_blocks():
        """
        Build a decomposition using Strassen blocks where possible,
        and schoolbook for the rest.
        
        Strategy: Partition the problem into 2x2*2x2 blocks where Strassen
        can be applied, and use schoolbook for remaining terms.
        
        For 3x3 * 3x7:
        - i (rows of A): 0,1,2
        - j (cols of A / rows of B): 0,1,2
        - k (cols of B): 0,1,2,3,4,5,6
        
        We can form 2x2*2x2 Strassen blocks for:
        - i={0,1}, j={0,1}, k={0,1}
        - i={0,1}, j={0,1}, k={2,3}
        - i={0,1}, j={0,1}, k={4,5}
        
        Remaining terms (schoolbook):
        - All terms with i=2 (3*3*7 = 63, but 2*3*7=42 already covered by Strassen)
          Actually, let me count more carefully.
        
        Total schoolbook terms: 3*3*7 = 63
        Strassen blocks cover: 3 blocks * 8 terms each = 24 terms, replaced by 3*7=21 terms
        Remaining schoolbook terms: 63 - 24 = 39 terms
        Total rank: 21 + 39 = 60
        
        Wait, let me recount. The Strassen blocks cover specific (i,j,k) combinations.
        Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 8 terms
        Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> 8 terms
        Block 3: i in {0,1}, j in {0,1}, k in {4,5} -> 8 terms
        Total covered: 24 terms, replaced by 21 terms (savings of 3)
        
        Remaining 39 terms use schoolbook.
        Total rank: 21 + 39 = 60
        """
        rank = 60
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Apply Strassen blocks
        # Block 1: i={0,1}, j={0,1}, k={0,1}
        col += strassen_2x2_block(0, 1, 0, 1, 0, 1, U, V, W, col)
        # Block 2: i={0,1}, j={0,1}, k={2,3}
        col += strassen_2x2_block(0, 1, 0, 1, 2, 3, U, V, W, col)
        # Block 3: i={0,1}, j={0,1}, k={4,5}
        col += strassen_2x2_block(0, 1, 0, 1, 4, 5, U, V, W, col)
        
        # Now add schoolbook terms for the remaining (i,j,k) combinations
        # Strassen covered: i in {0,1}, j in {0,1}, k in {0,1,2,3,4,5}
        # Remaining: 
        #   i in {0,1}, j in {0,1}, k = 6
        #   i in {0,1}, j = 2, k in {0,...,6}
        #   i = 2, j in {0,1,2}, k in {0,...,6}
        
        remaining_ijk = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check if this was covered by Strassen
                    if i in [0, 1] and j in [0, 1] and k in [0, 1, 2, 3, 4, 5]:
                        continue
                    remaining_ijk.append((i, j, k))
        
        for (i, j, k) in remaining_ijk:
            U[i*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
        
        # Verify
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None
    
    def random_greedy_search(target_rank, n_trials=10):
        """
        Try to find a rank-target_rank decomposition using random initialization
        and greedy completion.
        """
        if time.time() > deadline - 5:
            return None
        
        for trial in range(n_trials):
            if time.time() > deadline - 3:
                return None
            
            # Initialize with random small values
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Try a few iterations of adjustment
            for _ in range(50):
                if time.time() > deadline - 3:
                    break
                # Check current error
                max_err = 0
                worst_ijk = (0, 0, 0)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(target_rank):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            err = abs(s - 1)
                            if err > max_err:
                                max_err = err
                                worst_ijk = (i, j, k)
                
                if max_err < 1e-6:
                    # Check if all are close to 1
                    if verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
                    break
                
                # Try to adjust the worst term
                i, j, k = worst_ijk
                # Find the term r that contributes most to this (i,j,k)
                contributions = []
                for r in range(target_rank):
                    contrib = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    contributions.append((abs(contrib), r))
                contributions.sort(reverse=True)
                
                # Try flipping the sign of the largest contributor
                if contributions:
                    r_best = contributions[0][1]
                    # Try flipping U
                    old_u = U[i*m+j][r_best]
                    U[i*m+j][r_best] = -old_u
                    if verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
                    U[i*m+j][r_best] = old_u
                    
                    # Try flipping V
                    old_v = V[j*p+k][r_best]
                    V[j*p+k][r_best] = -old_v
                    if verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
                    V[j*p+k][r_best] = old_v
                    
                    # Try flipping W
                    old_w = W[k*n+i][r_best]
                    W[k*n+i][r_best] = -old_w
                    if verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
                    W[k*n+i][r_best] = old_w
            
            # Reset for next trial
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)]
        
        return None
    
    def try_structured_reduction():
        """
        Try to construct a lower-rank decomposition using structured patterns.
        """
        # For now, try a few specific constructions
        return None
    
    # Main search strategy
    best_result = None
    best_rank = schoolbook_rank()  # 63
    
    # First, try the Strassen block construction (should give rank 60)
    if time.time() < deadline - 10:
        result = build_strassen_blocks()
        if result is not None:
            best_rank = 60
            best_result = result
    
    # Try random greedy search for ranks below current best
    if best_result is None or best_rank > 55:
        if time.time() < deadline - 10:
            for target_rank in [58, 56, 54, 52, 50, 48, 46, 44, 42]:
                if time.time() > deadline - 5:
                    break
                result = random_greedy_search(target_rank, n_trials=5)
                if result is not None:
                    best_rank = target_rank
                    best_result = result
                    break
    
    # If nothing found, try a few more ranks
    if best_result is None and time.time() < deadline - 10:
        for target_rank in [56, 54, 52, 50, 48, 46, 44]:
            if time.time() > deadline - 5:
                break
            result = random_greedy_search(target_rank, n_trials=3)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
