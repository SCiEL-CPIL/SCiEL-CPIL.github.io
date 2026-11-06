"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 78.0  # safety margin before 90s kill
    
    N = 128
    # Precompute gcd for all pairs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def count_conflicts(S, x):
        """Count how many pairs (a,b) in S form a forbidden triple with x."""
        cnt = 0
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                if gcd_table[a][b] == gcd_table[a][x] == gcd_table[b][x]:
                    cnt += 1
        return cnt
    
    def is_valid_addition(S, x):
        """Check if adding x to S creates any forbidden triple."""
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                if gcd_table[a][b] == gcd_table[a][x] == gcd_table[b][x]:
                    return False
        return True
    
    def greedy_construct(order):
        """Build a valid set by processing elements in given order."""
        S = []
        for x in order:
            if is_valid_addition(S, x):
                S.append(x)
        return S
    
    def local_search(S):
        """Try to improve S by adding elements or swapping."""
        S = list(S)
        improved = True
        while improved:
            improved = False
            # Try to add any element not in S
            for x in range(1, N + 1):
                if x not in S:
                    if is_valid_addition(S, x):
                        S.append(x)
                        improved = True
                        break
            # If no addition possible, try swap: remove one, add two
            if not improved:
                for i in range(len(S)):
                    a = S[i]
                    S_temp = [x for j, x in enumerate(S) if j != i]
                    added = 0
                    for x in range(1, N + 1):
                        if x not in S_temp and x != a:
                            if is_valid_addition(S_temp, x):
                                S_temp.append(x)
                                added += 1
                                if added >= 2:
                                    break
                    if added >= 2:
                        S = S_temp
                        improved = True
                        break
        return S
    
    best_S = []
    
    # Multiple random greedy initializations + local search
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_construct(order)
        S = local_search(S)
        
        if len(S) > len(best_S):
            best_S = S
    
    # Additional pass: try to add elements to best_S
    if time.time() - start_time < DEADLINE - 5:
        for x in range(1, N + 1):
            if x not in best_S:
                if is_valid_addition(best_S, x):
                    best_S.append(x)
    
    # Final local search on best
    if time.time() - start_time < DEADLINE - 3:
        best_S = local_search(best_S)
    
    return best_S
# EVOLVE-BLOCK-END
