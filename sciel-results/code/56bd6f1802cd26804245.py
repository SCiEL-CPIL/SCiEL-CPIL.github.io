import time
import math
import numpy as np
from itertools import combinations

def compute_score(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    
    # Efficiently compute sum and difference sets using numpy or sets
    # For small to medium U, set operations are fine. 
    # U-U is symmetric around 0, range [-maxU, maxU].
    # U+U is symmetric? No, U is non-negative, so U+U is [0, 2maxU].
    
    sum_set = set()
    diff_set = set()
    
    # Optimization: If U is large, O(N^2) might be slow. 
    # But N is likely small enough (hundreds) for 150s if we are smart.
    # Let's use a boolean array for sums and diffs if maxU is not too huge, 
    # or just sets. Sets are safer for memory if maxU is large but |U| is small.
    
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Strategy: Greedy construction.
    # We want to maximize |U-U| / |U+U|.
    # Note that |U-U| <= (2*maxU + 1).
    # |U+U| >= |U|.
    # For a set of size N, max possible |U-U| is 2*maxU+1.
    # If we can make U-U cover the entire range [-maxU, maxU], then |U-U| = 2*maxU+1.
    # Then the score is 1 + log((2*maxU+1)/|U+U|) / log(2*maxU+1).
    # To maximize this, we want |U+U| to be as small as possible relative to maxU.
    # However, |U+U| is at least 2*maxU+1 - something? No.
    # Actually, for any set U, |U+U| + |U-U| >= 2*maxU + 1 is not a standard tight bound, 
    # but typically |U+U| is proportional to maxU for "dense" sets.
    # For "sparse" sets like powers of 2, |U+U| is small (O(N^2)) while maxU is large (exponential).
    # Let's test the structure of powers of 2.
    
    # Candidate 1: Powers of 2
    # U = {0, 1, 2, 4, ..., 2^k}
    # maxU = 2^k
    # |U| = k+1
    # U+U: Sums of two powers of 2. Distinct sums are 0, 1, 2, 3, 4, 5, ... up to 2^(k+1)-1?
    # No. 1+2=3, 2+4=6, 1+4=5. 
    # The sumset of powers of 2 has size roughly k(k+1)/2? 
    # Let's just generate and score.
    
    best_score = -1
    best_U = [0]
    
    # Try powers of 2
    for k in range(1, 50):
        if time.time() > deadline:
            break
        U = [2**i for i in range(k)]
        U.insert(0, 0) # Ensure 0 is included (2^0=1, so 0 is not in powers of 2 usually unless we start at 0? 2^0=1. 0 is not a power of 2.)
        # The problem requires 0 in U.
        # If U = {0, 1, 2, 4...}, maxU = 2^(k-1).
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # Try other bases?
    # Base 3?
    for k in range(1, 30):
        if time.time() > deadline:
            break
        U = [3**i for i in range(k)]
        if 0 not in U:
            U.insert(0, 0)
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # Try a greedy addition approach
    # Start with [0, 1]
    # At each step, try adding a new number x.
    # We want to maximize the score.
    # The score depends on maxU. Increasing maxU increases the denominator log(2*maxU+1).
    # It also potentially increases |U-U| and |U+U|.
    # If we add a very large number, maxU jumps.
    
    # Let's try a simple local search on the last element of the power of 2 sequence.
    # Or just stick with the best found so far.
    
    # The paper mentions size 2003 and 54265.
    # Powers of 2 with k=11 gives size 12 (0 to 1024). 
    # k=20 gives size 21 (0 to 2^19 ~ 500k).
    # Let's check the score for k=20 powers of 2.
    
    # Let's also try a "greedy max difference" set.
    # To maximize |U-U|, we want U to be a "Sidon-like" or "B-heteromorphic" set?
    # Actually, we want |U-U| to be large. The maximum possible is 2*maxU+1.
    # If U is an arithmetic progression, |U-U| = 2*maxU+1.
    # But |U+U| is also large (approx 2*maxU).
    # Ratio is ~ 1. Score is ~ 1.
    # For powers of 2, |U-U| is not full range, but maxU is huge.
    # Let's verify the score calculation for powers of 2.
    
    # If U = {0, 1, 2, 4, 8, 16} (k=5)
    # maxU = 16. Denom = log(33) ~ 3.49.
    # U-U: differences between powers of 2.
    # U+U: sums.
    
    # Let's run a quick local optimization on the exponent k for base 2.
    # And maybe mix?
    
    # What if we use a set that is a union of powers of 2 and some others?
    # Given the time limit, the powers of 2 construction is a strong baseline for "sparse" sets
    # which tend to have high maxU relative to |U|, and potentially favorable ratio.
    
    # Let's check if a specific known construction from the paper (size 2003) is better.
    # We don't have the paper's exact set, but we can guess it's a specific sequence.
    # However, without the paper, powers of 2 is the most standard "sparse" set.
    
    # Let's try to optimize the base.
    for base in [2, 3, 4, 5]:
        for k in range(1, 40):
            if time.time() > deadline:
                break
            U = [base**i for i in range(k)]
            if 0 not in U:
                U.insert(0, 0)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    # If time remains, try a random local search around the best_U
    if time.time() < deadline - 10:
        # Simple random perturbations
        import random
        random.seed(42)
        current_best = best_U[:]
        current_score = best_score
        
        for _ in range(500):
            if time.time() > deadline:
                break
            U = current_best[:]
            # Add a random number
            max_val = max(U) if U else 1
            new_val = random.randint(max_val//2, max_val*2 + 10)
            if new_val not in U:
                U.append(new_val)
                s = compute_score(U)
                if s > current_score:
                    current_score = s
                    current_best = U[:]
            # Remove a random number (except 0)
            elif len(U) > 2:
                idx = random.randint(1, len(U)-1)
                U.pop(idx)
                s = compute_score(U)
                if s > current_score:
                    current_score = s
                    current_best = U[:]
                    
        best_U = current_best

    return best_U

# Keep the function name exactly as required
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program had: def construct(): return [0, 1, 3]
# So I must define construct().

# Note: The seed program didn't have compute_score, but I added it. 
# The evaluator calls construct().
# I should ensure compute_score is defined before construct or inside it if needed, 
# but it's cleaner outside.

if __name__ == "__main__":
    # Quick self-test
    print(construct())
