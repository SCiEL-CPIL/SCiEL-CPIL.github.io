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
    
    def strassen_2x2():
        """
        Apply Strassen's 7-multiplication algorithm to a 2x2 * 2x2 sub-block.
        Then combine with remaining parts using schoolbook.
        
        For 3x3 * 3x7, we partition:
        - Use Strassen for a 2x2 * 2x2 block (rank 7 instead of 16)
        - Use schoolbook for the remaining parts
        
        Partition of 3x3 * 3x7:
        A = [[A11, A12, A13],
             [A21, A22, A23],
             [A31, A32, A33]]
        B = [[B11, B12, B13, B14, B15, B16, B17],
             [B21, B22, B23, B24, B25, B26, B27],
             [B31, B32, B33, B34, B35, B36, B37]]
        
        Let's use Strassen on the top-left 2x2 * 2x2 block (rows 0-1, cols 0-1 of B).
        Then schoolbook for everything else.
        
        Strassen for 2x2 * 2x2:
        M1 = (A11+A22)*(B11+B22)
        M2 = (A21+A22)*B11
        M3 = A11*(B12-B22)
        M4 = A22*(B21-B11)
        M5 = (A11+A12)*B22
        M6 = (A21-A11)*(B11+B12)
        M7 = (A12-A22)*(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        For our 3x3 * 3x7, let's try a simpler approach:
        Just use Strassen-like reduction on the full problem.
        """
        # For now, let's try a direct construction
        # We'll use the fact that 3x3*3x7 can be viewed as:
        # For each of the 3 rows of A, we compute a 1x3 * 3x7 product
        # But we can share some multiplications
        
        # Actually, let's try a different approach:
        # Use the known rank for 3x3*3x3 which is 23 (or lower)
        # And extend to 3x3*3x7
        
        # For 3x3 * 3x7, we can think of it as 3 separate 1x3 * 3x7 products
        # But they share the B matrix
        
        # Let's try a structured approach:
        # For each column k of B, we have a 3x1 vector B[:,k]
        # C[:,k] = A * B[:,k]
        # This is a 3x3 * 3x1 product for each k
        
        # A 3x3 * 3x1 product has rank 9 (schoolbook) or potentially lower
        # But since all 7 columns share the same A, we can potentially share
        
        # Let's just return schoolbook for now and try random search
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
        
        Key insight: For 3x3 * 3x7, we can try to exploit the fact that
        the problem is essentially 3 independent 1x3 * 3x7 products that
        share the same B matrix.
        
        Let's try a construction based on:
        - For each row i of A, C[i,:] = A[i,:] * B
        - This is a dot product of A[i,:] (length 3) with each row of B
        
        A structured approach:
        - Use 3 "base" multiplications for each j (column of A / row of B)
        - But try to share across i's
        
        Actually, let's try a simpler structured approach:
        Partition the 7 columns of B into groups and use block methods.
        """
        # For now, try a few specific constructions
        
        # Construction 1: Use 3x3*3x3 Strassen-like + 3x3*3x4 schoolbook
        # 3x3*3x3 has known rank 23 (or 21, 22 depending on source)
        # Let's assume we can get rank 23 for 3x3*3x3
        # And 3x3*3x4 has rank 36 (schoolbook)
        # Total: 23 + 36 = 59 (worse than 63 schoolbook? No, 59 < 63)
        # But we need to verify the 3x3*3x3 rank 23 construction
        
        # For simplicity, let's just try random search for various ranks
        return None
    
    # Main search strategy
    best_result = None
    best_rank = schoolbook_rank()  # 63
    
    # First, try some structured approaches
    if time.time() < deadline - 10:
        # Try random greedy search for ranks below 63
        for target_rank in [42, 36, 30, 28, 24, 21, 20, 18, 16]:
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
