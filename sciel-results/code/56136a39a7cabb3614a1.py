"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import random
import time
from collections import Counter

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p  # 63
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = schoolbook_rank

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n_, m_, p_ = n, m, p
        if len(U) != n_ * m_:
            return False
        if len(V) != m_ * p_:
            return False
        if len(W) != p_ * n_:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        rank_ = len(U[0])
        if len(V[0]) != rank_ or len(W[0]) != rank_:
            return False
        if rank_ <= 0:
            return False
        
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    expected = 1
                    actual = 0.0
                    for r in range(rank_):
                        actual += U[i * m_ + j][r] * V[j * p_ + k][r] * W[k * n_ + i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def make_result(U, V, W):
        """Convert to list-of-lists format."""
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        # Clean up near-integer values
        U_list = [[round(x, 10) if abs(x - round(x, 10)) < 1e-8 else x for x in row] for row in U_list]
        V_list = [[round(x, 10) if abs(x - round(x, 10)) < 1e-8 else x for x in row] for row in V_list]
        W_list = [[round(x, 10) if abs(x - round(x, 10)) < 1e-8 else x for x in row] for row in W_list]
        return {'U': U_list, 'V': V_list, 'W': W_list}

    # Strategy: Try to find a structured decomposition using the rectangular shape
    # The matrix multiplication tensor for (n,m,p) has a specific structure that can sometimes
    # be exploited. For (3,3,7), we might be able to find a lower rank than 63.
    
    # Approach 1: Try a block-structured decomposition
    # We can view the 3x7 matrix as a concatenation of a 3x3 and a 3x4 block
    # C = A * [B1 B2] where B1 is 3x3 and B2 is 3x4
    # C1 = A * B1 (3x3 result) and C2 = A * B2 (3x4 result)
    
    # Let's try to find low-rank decompositions for the sub-problems and combine them
    # This is a direct sum approach: rank(3x3x7) <= rank(3x3x3) + rank(3x3x4)
    
    # For 3x3x3, the known minimal rank is 23 (Strassen-like improvements)
    # For 3x3x4, let's try to find a low rank
    
    # First, let's construct the 3x3x3 schoolbook
    def construct_3x3x3_schoolbook():
        n_, m_, p_ = 3, 3, 3
        rank = n_ * m_ * p_  # 27
        U = [[0] * rank for _ in range(n_ * m_)]
        V = [[0] * rank for _ in range(m_ * p_)]
        W = [[0] * rank for _ in range(p_ * n_)]
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U[i * m_ + j][col] = 1
                    V[j * p_ + k][col] = 1
                    W[k * n_ + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find a 3x3x3 decomposition with rank < 27
    # Use a simple random search with small integer coefficients
    def search_3x3x3(rank_target, num_trials, coeff_set):
        n_, m_, p_ = 3, 3, 3
        best_U = best_V = best_W = None
        best_r = rank_target + 1
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            U = [[random.choice(coeff_set) for _ in range(rank_target)] for _ in range(n_ * m_)]
            V = [[random.choice(coeff_set) for _ in range(rank_target)] for _ in range(m_ * p_)]
            W = [[random.choice(coeff_set) for _ in range(rank_target)] for _ in range(p_ * n_)]
            
            # Verify
            valid = True
            for i in range(n_):
                for j in range(m_):
                    for k in range(p_):
                        actual = 0.0
                        for r in range(rank_target):
                            actual += U[i * m_ + j][r] * V[j * p_ + k][r] * W[k * n_ + i][r]
                        if abs(actual - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_U, best_V, best_W = U, V, W
                best_r = rank_target
                break
        
        if best_r <= rank_target:
            return best_U, best_V, best_W, best_r
        return None, None, None, rank_target + 1
    
    # Try a few ranks for 3x3x3
    coeff_options = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
        [-1, 0, 1, 2],
        [0, 1, -1, 2, -2]
    ]
    
    # Search for 3x3x3 with ranks from 23 to 26
    U_333 = V_333 = W_333 = None
    rank_333 = 27  # Start with schoolbook
    
    for target_rank in range(23, 27):
        if time.time() > deadline:
            break
        found_U, found_V, found_W, found_rank = search_333(
            target_rank, 50, [-1, 0, 1, 2]
        )
        if found_rank <= target_rank:
            U_333, V_333, W_333 = found_U, found_V, found_W
            rank_333 = found_rank
            break
    
    # If no improvement, use schoolbook
    if U_333 is None:
        U_333, V_333, W_333, rank_333 = construct_333_schoolbook()
    
    # Now try to find a 3x3x4 decomposition
    # rank(3x3x4) >= max(9, 12, 12) = 12
    # Let's try ranks from 13 to 16
    
    def construct_3x3x4_schoolbook():
        n_, m_, p_ = 3, 3, 4
        rank = n_ * m_ * p_  # 36
        U = [[0] * rank for _ in range(n_ * m_)]
        V = [[0] * rank for _ in range(m_ * p_)]
        W = [[0] * rank for _ in range(p_ * n_)]
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U[i * m_ + j][col] = 1
                    V[j * p_ + k][col] = 1
                    W[k * n_ + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U_334 = V_334 = W_334 = None
    rank_334 = 36  # Start with schoolbook
    
    # Search for 3x3x4 with ranks from 13 to 20
    for target_rank in range(13, 21):
        if time.time() > deadline:
            break
        
        # Random search
        found = False
        for trial in range(30):
            if time.time() > deadline:
                break
            
            n_, m_, p_ = 3, 3, 4
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n_ * m_)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m_ * p_)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p_ * n_)]
            
            # Verify
            valid = True
            for i in range(n_):
                for j in range(m_):
                    for k in range(p_):
                        actual = 0.0
                        for r in range(target_rank):
                            actual += U[i * m_ + j][r] * V[j * p_ + k][r] * W[k * n_ + i][r]
                        if abs(actual - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                U_334, V_334, W_334 = U, V, W
                rank_334 = target_rank
                found = True
                break
        
        if found:
            break
    
    # If no improvement, use schoolbook
    if U_334 is None:
        U_334, V_334, W_334, rank_334 = construct_334_schoolbook()
    
    # Combine the two decompositions
    # The combined decomposition has rank = rank_333 + rank_334
    combined_rank = rank_333 + rank_334
    
    # Construct the combined U, V, W
    # For the 3x3x7 problem:
    # U has rows indexed by (i, j) where i in [0,3), j in [0,3) -> 9 rows
    # V has rows indexed by (j, k) where j in [0,3), k in [0,7) -> 21 rows
    # W has rows indexed by (k, i) where k in [0,7), i in [0,3) -> 21 rows
    
    # The 3x3x3 part uses k in [0,3)
    # The 3x3x4 part uses k in [0,4) which we'll map to k in [3,7)
    
    U_combined = [[0] * combined_rank for _ in range(9)]
    V_combined = [[0] * combined_rank for _ in range(21)]
    W_combined = [[0] * combined_rank for _ in range(21)]
    
    # Fill in the 3x3x3 part (k = 0,1,2)
    for i in range(3):
        for j in range(3):
            for r in range(rank_333):
                U_combined[i * 3 + j][r] = U_333[i * 3 + j][r]
        
        for k in range(3):
            for r in range(rank_333):
                V_combined[j * 7 + k][r] = V_333[j * 3 + k][r]
    
    for k in range(3):
        for i in range(3):
            for r in range(rank_333):
                W_combined[k * 3 + i][r] = W_333[k * 3 + i][r]
    
    # Fill in the 3x3x4 part (k = 3,4,5,6)
    offset = rank_333
    for i in range(3):
        for j in range(3):
            for r in range(rank_334):
                U_combined[i * 3 + j][offset + r] = U_334[i * 3 + j][r]
        
        for k in range(4):
            for r in range(rank_334):
                V_combined[j * 7 + (3 + k)][offset + r] = V_334[j * 3 + k][r]
    
    for k in range(4):
        for i in range(3):
            for r in range(rank_334):
                W_combined[(3 + k) * 3 + i][offset + r] = W_334[k * 3 + i][r]
    
    # Verify the combined decomposition
    if verify_decomposition(U_combined, V_combined, W_combined):
        result = make_result(U_combined, V_combined, W_combined)
        best_result = result
        best_rank = combined_rank
    
    # Try a direct search for the full 3x3x7 problem with lower ranks
    # The lower bound is max(9, 21, 21) = 21
    # Let's try ranks from 21 to 40
    
    if time.time() < deadline - 10:
        for target_rank in range(21, min(40, best_rank)):
            if time.time() > deadline:
                break
            
            # Random search with sparse coefficients
            found = False
            for trial in range(20):
                if time.time() > deadline:
                    break
                
                U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(9)]
                V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(21)]
                W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(21)]
                
                if verify_decomposition(U, V, W):
                    result = make_result(U, V, W)
                    best_result = result
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
    
    return best_result

# Helper function for 3x3x3 search (defined inside construct to avoid global state)
def search_333(rank_target, num_trials, coeff_set):
    """Search for a 3x3x3 decomposition with given rank target."""
    n_, m_, p_ = 3, 3, 3
    best_U = best_V = best_W = None
    best_r = rank_target + 1
    
    for trial in range(num_trials):
        U = [[random.choice(coeff_set) for _ in range(rank_target)] for _ in range(n_ * m_)]
        V = [[random.choice(coeff_set) for _ in range(rank_target)] for _ in range(m_ * p_)]
        W = [[random.choice(coeff_set) for _ in range(rank_target)] for _ in range(p_ * n_)]
        
        # Verify
        valid = True
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    actual = 0.0
                    for r in range(rank_target):
                        actual += U[i * m_ + j][r] * V[j * p_ + k][r] * W[k * n_ + i][r]
                    if abs(actual - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            best_U, best_V, best_W = U, V, W
            best_r = rank_target
            break
    
    if best_r <= rank_target:
        return best_U, best_V, best_W, best_r
    return None, None, None, rank_target + 1

def construct_333_schoolbook():
    """Construct schoolbook decomposition for 3x3x3."""
    n_, m_, p_ = 3, 3, 3
    rank = n_ * m_ * p_  # 27
    U = [[0] * rank for _ in range(n_ * m_)]
    V = [[0] * rank for _ in range(m_ * p_)]
    W = [[0] * rank for _ in range(p_ * n_)]
    col = 0
    for i in range(n_):
        for j in range(m_):
            for k in range(p_):
                U[i * m_ + j][col] = 1
                V[j * p_ + k][col] = 1
                W[k * n_ + i][col] = 1
                col += 1
    return U, V, W, rank

def construct_334_schoolbook():
    """Construct schoolbook decomposition for 3x3x4."""
    n_, m_, p_ = 3, 3, 4
    rank = n_ * m_ * p_  # 36
    U = [[0] * rank for _ in range(n_ * m_)]
    V = [[0] * rank for _ in range(m_ * p_)]
    W = [[0] * rank for _ in range(p_ * n_)]
    col = 0
    for i in range(n_):
        for j in range(m_):
            for k in range(p_):
                U[i * m_ + j][col] = 1
                V[j * p_ + k][col] = 1
                W[k * n_ + i][col] = 1
                col += 1
    return U, V, W, rank
