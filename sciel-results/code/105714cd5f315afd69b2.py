"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_mod2(U, V, W):
        """Verify decomposition over F_2"""
        U_mod = np.array(U, dtype=np.uint8) % 2
        V_mod = np.array(V, dtype=np.uint8) % 2
        W_mod = np.array(W, dtype=np.uint8) % 2
        rank = U_mod.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val = (val + U_mod[i*m+j, c] * V_mod[j*p+k, c] * W_mod[k*n+i, c]) % 2
                    if val != 1:
                        return False
        return True

    def try_rank_over_F2(target_rank, max_iter=50000):
        """
        Try to find a rank-target decomposition over F_2.
        Returns (U, V, W) as lists of lists of 0s and 1s, or None if not found.
        """
        nm = n * m  # 9
        mp = m * p  # 12
        pn = p * n  # 12
        
        # Precompute which (i,j,k) triples need to be satisfied
        # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 mod 2
        
        # Random search: pick random sparse vectors
        random.seed(42)
        
        for iter in range(max_iter):
            if time.time() > deadline - 15:
                return None
            
            # Generate random sparse binary vectors
            # Each vector has about 2-4 non-zero entries
            U = np.zeros((nm, target_rank), dtype=np.uint8)
            V = np.zeros((mp, target_rank), dtype=np.uint8)
            W = np.zeros((pn, target_rank), dtype=np.uint8)
            
            for c in range(target_rank):
                # Choose random positions for U, V, W
                u_pos = random.randint(0, nm - 1)
                v_pos = random.randint(0, mp - 1)
                w_pos = random.randint(0, pn - 1)
                
                # With some probability, make it sparse (multiple non-zeros)
                if random.random() < 0.3:
                    # Add 1-2 extra non-zeros
                    for _ in range(random.randint(1, 2)):
                        u_pos2 = random.randint(0, nm - 1)
                        v_pos2 = random.randint(0, mp - 1)
                        w_pos2 = random.randint(0, pn - 1)
                        U[u_pos2, c] = 1
                        V[v_pos2, c] = 1
                        W[w_pos2, c] = 1
                
                U[u_pos, c] = 1
                V[v_pos, c] = 1
                W[w_pos, c] = 1
            
            if verify_mod2(U, V, W):
                return (U.tolist(), V.tolist(), W.tolist())
        
        return None

    def try_gaussian_elimination_F2(target_rank):
        """
        Use linear algebra over F_2 to find a decomposition.
        This is more systematic than random search.
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        # The tensor equation is: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of 36 equations (n*m*p = 3*3*4 = 36)
        # But it's nonlinear (trilinear), so we can't directly use Gaussian elimination.
        # However, we can fix two of the three matrices and solve for the third.
        
        # Strategy: Fix U and V randomly, then solve for W using the linear system.
        # For fixed U and V, the equation becomes:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is linear in W!
        
        random.seed(123)
        
        for trial in range(100):
            if time.time() > deadline - 15:
                return None
            
            # Generate random U and V
            U = np.zeros((nm, target_rank), dtype=np.uint8)
            V = np.zeros((mp, target_rank), dtype=np.uint8)
            
            for c in range(target_rank):
                # Make U and V sparse
                for _ in range(random.randint(1, 3)):
                    U[random.randint(0, nm-1), c] = 1
                for _ in range(random.randint(1, 3)):
                    V[random.randint(0, mp-1), c] = 1
            
            # Build linear system for W
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Group by (k*n+i): for each row of W, we have equations
            
            # Actually, let's think differently. For each column c of W, let w_c be a vector in F_2^pn.
            # The equation is: sum_c (U_col_c * V_col_c) * w_c = 1 for each (i,j,k)
            
            # Let A[i,j,k,c] = U[i*m+j,c] * V[j*p+k,c]
            # Then sum_c A[i,j,k,c] * W[k*n+i,c] = 1
            
            # This is a system of 36 equations in 12*target_rank unknowns (entries of W)
            # But it's not linear in W because W appears with different indices for different (i,j,k)
            
            # Let me reconsider. For fixed (i,j,k), the equation involves W[k*n+i,c] for all c.
            # So if we group by the index (k*n+i), we get:
            # For each (i,j,k): sum_c A[i,j,k,c] * W[k*n+i,c] = 1
            
            # Let's build a matrix B where B[row, col] represents the coefficient of W[col_row, col_col]
            # Actually, let's just set up the system properly.
            
            # Unknowns: W[0,0], W[0,1], ..., W[pn-1, target_rank-1]
            # Total unknowns: pn * target_rank
            
            # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This equation only involves W[k*n+i, :] for all c
            
            # So we can group equations by the W-row index (k*n+i)
            # For a fixed row r = k*n+i, which equations involve W[r, c]?
            # Equations where k*n+i = r, i.e., i = r % n, k = r // n
            # And j ranges over 0..m-1
            
            # So for each row r of W, we have m equations (one for each j)
            # And each equation involves all target_rank columns of W[r, :]
            
            # Let's build the system:
            # For each row r (0 to pn-1):
            #   i = r % n, k = r // n
            #   For each j (0 to m-1):
            #     sum_c U[i*m+j,c] * V[j*p+k,c] * W[r,c] = 1
            
            # This gives a block diagonal system! Each block is m x target_rank
            # And we need to solve m equations in target_rank unknowns for each block.
            
            # For a solution to exist, we need m <= target_rank (which is true for target_rank >= 3)
            # And the m equations must be consistent.
            
            W = np.zeros((pn, target_rank), dtype=np.uint8)
            
            for r in range(pn):
                i = r % n
                k = r // n
                
                # Build m x target_rank matrix A_r where A_r[j, c] = U[i*m+j,c] * V[j*p+k,c]
                A_r = np.zeros((m, target_rank), dtype=np.uint8)
                for j in range(m):
                    for c in range(target_rank):
                        A_r[j, c] = (U[i*m+j, c] * V[j*p+k, c]) % 2
                
                # We need A_r * W[r, :]^T = [1, 1, ..., 1]^T (vector of m ones)
                # This is a linear system over F_2
                
                # Check if the system is consistent
                # Augment with ones
                A_aug = np.column_stack([A_r, np.ones(m, dtype=np.uint8)])
                
                # Compute rank of A_r and A_aug
                # For a consistent system, rank(A_r) == rank(A_aug)
                
                def rank_F2(M):
                    """Compute rank of binary matrix over F_2"""
                    M = M.copy()
                    rows = M.shape[0]
                    cols = M.shape[1]
                    rank = 0
                    for col in range(cols):
                        # Find pivot
                        pivot = -1
                        for row in range(rank, rows):
                            if M[row, col] == 1:
                                pivot = row
                                break
                        if pivot == -1:
                            continue
                        # Swap rows
                        M[[rank, pivot]] = M[[pivot, rank]]
                        # Eliminate
                        for row in range(rows):
                            if row != rank and M[row, col] == 1:
                                M[row] = (M[row] + M[rank]) % 2
                        rank += 1
                    return rank
                
                rank_A = rank_F2(A_r)
                rank_aug = rank_F2(A_aug)
                
                if rank_A != rank_aug:
                    # Inconsistent system, try different U, V
                    break
                
                # System is consistent, find a solution
                # We can find a particular solution by setting free variables to 0
                # For simplicity, let's just try to find any solution
                # Use the same elimination process to back-substitute
                
                # Actually, let's just try a few random solutions
                found_solution = False
                for _ in range(20):
                    # Try a random solution
                    W_r = np.zeros(target_rank, dtype=np.uint8)
                    
                    # For each column, set it randomly
                    for c in range(target_rank):
                        W_r[c] = random.randint(0, 1)
                    
                    # Check if it satisfies the equations
                    valid = True
                    for j in range(m):
                        val = 0
                        for c in range(target_rank):
                            val = (val + A_r[j, c] * W_r[c]) % 2
                        if val != 1:
                            valid = False
                            break
                    
                    if valid:
                        W[r, :] = W_r
                        found_solution = True
                        break
                
                if not found_solution:
                    # Try systematic approach: set first (target_rank - m) variables to 0,
                    # solve for the rest
                    break
            
            if time.time() > deadline - 15:
                return None
            
            # Verify the complete solution
            if verify_mod2(U, V, W):
                return (U.tolist(), V.tolist(), W.tolist())
        
        return None

    # Try to find a decomposition with rank < 36
    # Start from rank 24 and go up
    best_result = schoolbook()
    best_rank = n * m * p
    
    for target_rank in range(24, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try Gaussian elimination approach
        result = try_gaussian_elimination_F2(target_rank)
        if result is not None:
            U_list, V_list, W_list = result
            # Verify over the reals (should work since entries are 0/1)
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try random search
        result = try_rank_over_F2(target_rank, max_iter=10000)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

# EVOLVE-BLOCK-END
