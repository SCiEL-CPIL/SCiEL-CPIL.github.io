import time
import math
import random
import numpy as np

def calculate_score(U):
    if not U:
        return 1.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Use bitsets for efficient sumset/difference set calculation
    # Range of sums: 0 to 2*max_u
    # Range of diffs: -max_u to max_u
    # We can represent the set U as a boolean array of length max_u + 1
    
    n = max_u + 1
    u_bits = np.zeros(n, dtype=bool)
    for u in U:
        u_bits[u] = True
        
    # Calculate Sumset U+U
    # Sums range from 0 to 2*max_u
    # We can use convolution or simple loops if N is small, but bitset shifts are faster
    # Actually, for M up to ~500, O(M^2) with numpy operations is fine, but let's try to be smart.
    # A simple O(N^2) loop in Python is slow. 
    # Let's use the property that |U+U| can be computed by checking all pairs? No, that's slow.
    # Let's stick to a slightly optimized Python set logic if N is small, or use numpy convolution?
    # Numpy convolution of boolean arrays is fast.
    
    # Sumset
    # u_bits is length n. Convolution of u_bits with u_bits gives counts of sums.
    # sums_counts = np.convolve(u_bits, u_bits)
    # sums = np.nonzero(sums_counts)[0]
    # len(sums) = |U+U|
    
    # Difference set
    # Diffs range from -max_u to max_u.
    # We can compute |U-U| by checking for each d in [-max_u, max_u] if there exists u1, u2 in U such that u1-u2=d.
    # This is equivalent to checking if (u_bits & (u_bits shifted by d)) has any True.
    # Alternatively, we can just use sets if N is very small, but for M=500, N can be up to 500.
    # O(N^2) with N=500 is 250,000 ops. In Python, this is ~0.01-0.05s. 
    # 120s budget allows for ~2000-10000 such evaluations.
    # But we can do better.
    
    # Let's use a hybrid: if max_u < 100, use set logic (simple). If max_u >= 100, use numpy.
    # Actually, let's just use numpy always for consistency and speed on larger sets.
    
    # Sumset via convolution
    sums_counts = np.convolve(u_bits, u_bits)
    len_sums = np.count_nonzero(sums_counts)
    
    # Difference set
    # For each shift d in -max_u to max_u, check if intersection is non-empty
    # We can iterate d from 0 to max_u. By symmetry, |U-U| = 1 + 2 * |{d in U-U : d > 0}|
    # Let's check d=0 to max_u.
    count_pos_diffs = 0
    for d in range(1, max_u + 1):
        # Shift u_bits by d. 
        # u_bits[d:] & u_bits[:-d]
        # If any True, then d is in U-U.
        if np.any(u_bits[d:] & u_bits[:-d]):
            count_pos_diffs += 1
            
    len_diffs = 1 + 2 * count_pos_diffs
    
    ratio = len_diffs / len_sums
    if ratio <= 1:
        return 1.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_U = {0}
    best_score = 1.0
    
    # 1. Exhaustive search for small ranges
    # We can exhaustively check all subsets of {0, 1, ..., M} for small M?
    # 2^M is too big for M>20.
    # Instead, we can use a greedy approach with a wide search window, or a local search.
    # Let's do a greedy construction first.
    
    current_U = {0}
    current_score = 1.0
    
    # Greedy Expansion
    # Try to add elements that improve the score.
    # We'll search up to a certain max value.
    max_val_limit = 300
    
    # Initialize with a small set to help the greedy step?
    # Maybe start with {0} and add 1, 2, etc.
    
    # To make the greedy step faster, we can pre-compute or use a heuristic.
    # Let's try adding elements in increasing order, and if it doesn't help, skip.
    # But the optimal set might not be contiguous.
    
    # Let's use a "Simulated Annealing" style local search with the fast scorer.
    
    # Initialize with a random set or a greedy one.
    # Greedy:
    current_U = {0}
    current_score = calculate_score(current_U)
    
    # Try to build a set of size ~20-50 with values up to 300.
    # We will iterate through potential additions.
    
    # To avoid getting stuck in local minima, we'll do multiple restarts with different random seeds/initializations.
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        init_size = random.randint(5, 15)
        init_max = random.randint(50, 150)
        current_U = {0}
        # Add random elements
        while len(current_U) < init_size:
            x = random.randint(1, init_max)
            current_U.add(x)
            
        current_score = calculate_score(current_U)
        
        if current_score > best_score:
            best_score = current_score
            best_U = set(current_U)
            
        # Local Search
        T = 0.5
        T_min = 0.01
        cooling = 0.99
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            if T < T_min:
                T = T_min
                
            new_U = set(current_U)
            move_type = random.random()
            
            if move_type < 0.4:
                # Add
                current_max = max(current_U)
                # Search range: up to current_max + 50 or max_val_limit
                search_max = min(current_max + 50, max_val_limit)
                k = random.randint(1, search_max)
                if k not in new_U:
                    new_U.add(k)
                else:
                    continue
            elif move_type < 0.7:
                # Remove
                removable = [x for x in current_U if x != 0]
                if removable:
                    k = random.choice(removable)
                    new_U.discard(k)
                else:
                    continue
            else:
                # Replace
                removable = [x for x in current_U if x != 0]
                if removable:
                    k = random.choice(removable)
                    current_max = max(current_U)
                    search_max = min(current_max + 50, max_val_limit)
                    y = random.randint(1, search_max)
                    if y != k:
                        new_U.discard(k)
                        new_U.add(y)
                    else:
                        continue
                else:
                    continue
                    
            new_score = calculate_score(new_U)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_U = new_U
                current_score = new_score
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_U = set(current_U)
            else:
                no_improve_count += 1
                
            T *= cooling
            
    return sorted(list(best_U))
