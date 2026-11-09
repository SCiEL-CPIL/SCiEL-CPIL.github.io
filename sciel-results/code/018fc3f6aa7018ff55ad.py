import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 85.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    known_sol = [8, 31, 248]
    best_max = 248
    best_list = known_sol
    
    # Try to find solutions with smaller max
    # Use a priority queue or iterative deepening
    
    # For each candidate max M (from 7 to best_max-1), check feasibility
    # Use DFS with strong pruning
    
    def dfs(rem, min_d, max_d, terms_left):
        if time.time() > deadline:
            return None
        if rem == 0:
            return []
        if terms_left <= 0:
            return None
        if rem < 0:
            return None
        if min_d > max_d:
            return None
        
        # Strong pruning: 
        # Max achievable sum: terms_left * (1/min_d)
        if rem > Fraction(terms_left, min_d):
            return None
        
        # Min achievable sum with at least 1 term: 1/max_d
        # If rem < 1/max_d and we need at least 1 more term, we can't do it with just 1 term
        # But we might use more terms. The min sum with k terms is sum of k smallest unit fractions in [min_d, max_d]
        # which is 1/max_d + 1/(max_d-1) + ... 
        # This is hard to compute quickly, so let's use a weaker bound:
        # If rem < 1/max_d and terms_left == 1, return None
        if terms_left == 1 and rem < Fraction(1, max_d):
            return None
        
        # Try each d
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            if d < min_d:
                continue
            term = Fraction(1, d)
            if term > rem:
                continue
            rem_new = rem - term
            if rem_new == 0:
                return [d]
            result = dfs(rem_new, d + 1, max_d, terms_left - 1)
            if result is not None:
                return [d] + result
        return None
    
    for M in range(best_max - 1, 6, -1):
        if time.time() > deadline:
            break
        result = dfs(TARGET, 1, M, MAX_TERMS)
        if result is not None:
            total = sum(Fraction(1, d) for d in result)
            if total == TARGET and len(set(result)) == len(result):
                best_max = M
                best_list = result
    
    return best_list
