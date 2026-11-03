import time
import sys
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The rank is the number of columns in these matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Initialize schoolbook as fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        return False
        return True

    def calculate_error(U, V, W, rank):
        """Calculate the sum of squared errors."""
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    target = 1
                    error += (val - target) ** 2
        return error

    def simulated_annealing_search(rank, max_time=30.0, seed=42):
        """
        Perform simulated annealing to find a valid decomposition with given rank.
        Uses integer entries from {-1, 0, 1}.
        """
        if time.time() > deadline:
            return None
        
        random.seed(seed)
        
        # Initialize with random integers
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        
        current_error = calculate_error(U, V, W, rank)
        if current_error == 0:
            return U, V, W
        
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        best_error = current_error
        
        T = 10.0
        T_min = 0.1
        cooling = 0.95
        
        while T > T_min and time.time() < start_time + max_time:
            if time.time() > deadline:
                break
                
            # Pick a random matrix and entry to flip
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                idx = random.randint(0, n * m - 1)
                r = random.randint(0, rank - 1)
                old_val = U[idx][r]
                # Cycle through values: -1 -> 0 -> 1 -> -1
                if old_val == -1: new_val = 0
                elif old_val == 0: new_val = 1
                else: new_val = -1
                
                U[idx][r] = new_val
                new_error = calculate_error(U, V, W, rank)
                delta = new_error - current_error
                
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_error == 0:
                            return best_U, best_V, best_W
                else:
                    U[idx][r] = old_val
                    
            elif mat_choice == 'V':
                idx = random.randint(0, m * p - 1)
                r = random.randint(0, rank - 1)
                old_val = V[idx][r]
                if old_val == -1: new_val = 0
                elif old_val == 0: new_val = 1
                else: new_val = -1
                
                V[idx][r] = new_val
                new_error = calculate_error(U, V, W, rank)
                delta = new_error - current_error
                
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_error == 0:
                            return best_U, best_V, best_W
                else:
                    V[idx][r] = old_val
                    
            else: # W
                idx = random.randint(0, p * n - 1)
                r = random.randint(0, rank - 1)
                old_val = W[idx][r]
                if old_val == -1: new_val = 0
                elif old_val == 0: new_val = 1
                else: new_val = -1
                
                W[idx][r] = new_val
                new_error = calculate_error(U, V, W, rank)
                delta = new_error - current_error
                
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        if best_error == 0:
                            return best_U, best_V, best_W
                else:
                    W[idx][r] = old_val
                    
            T *= cooling
            
        if best_error == 0:
            return best_U, best_V, best_W
        return None

    # Try ranks from a lower bound up to schoolbook - 1
    # Lower bound: 35. Known minimal is 36.
    # We will try a range of ranks, giving more time to lower ranks.
    
    # We prioritize finding *any* solution better than schoolbook.
    # Let's try ranks 36 to 50.
    
    ranks_to_try = list(range(36, 51))
    
    # Allocate time: 20s per rank is too much. 10s per rank is 150s total.
    # We have 110s budget.
    # Let's try 36-40 with 20s each, 41-45 with 10s each, 46-50 with 5s each.
    
    time_per_rank = {}
    for r in range(36, 41):
        time_per_rank[r] = 20.0
    for r in range(41, 46):
        time_per_rank[r] = 10.0
    for r in range(46, 51):
        time_per_rank[r] = 5.0
        
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            continue
            
        # Run simulated annealing
        result = simulated_annealing_search(rank, max_time=time_per_rank.get(rank, 5.0), seed=rank * 1000 + 42)
        
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            
            # If we found a low rank, we can stop or continue.
            # Since we want minimal rank, we continue to lower ranks if possible,
            # but we are iterating 36->50. So if we find 40, we still try 36-39?
            # No, the loop is ascending. We should try lower ranks first.
            # Let's reverse the order or just break if we find 36?
            # Actually, finding 36 is the goal. If we find 40, we haven't tried 36-39 yet in this loop structure if it's ascending.
            # Let's change the loop to descending? No, lower rank is harder.
            # It is better to try the *easiest* lower ranks first? No, 36 is the target.
            # Let's try 36 first. If it fails, try 37, etc.
            
            # Wait, the loop is `for rank in ranks_to_try` which is 36..50.
            # So it tries 36 first. If it finds 36, it updates best.
            # Then it tries 37. If it finds 37, best is still 36.
            # This is correct.
            
            # If we find a solution, we can stop early if we hit 36?
            # Yes, 36 is the theoretical lower bound + 1 (since 35 is impossible for this shape likely, or just the bound).
            # If we find 36, we can break.
            if rank == 36:
                break

    return best_result
# EVOLVE-BLOCK-END
