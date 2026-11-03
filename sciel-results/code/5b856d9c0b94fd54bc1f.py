import time
import random
from itertools import combinations

def compute_score(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
        
    # Compute sumset and difference set
    # Using sets for efficiency
    U_set = set(U)
    
    sumset = set()
    diffset = set()
    
    # Optimization: if |U| is small, nested loops are fine.
    # If |U| is large, we might need FFT, but for this search, |U| is small (< 10).
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    import math
    # Score: 1 + log(|U-U|/|U+U|) / log(2 max U + 1)
    # Note: log base e is fine as ratio is invariant to base
    try:
        val = 1.0 + math.log(size_diff / size_sum) / math.log(2 * max_u + 1)
        return val
    except (ValueError, ZeroDivisionError):
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Search parameters
    # Based on "symmetric core" strategy
    # U = {0} U {x, M-x | x in S}
    # We search for M and S
    
    # We can also try non-symmetric sets if symmetric doesn't yield enough, 
    # but the directive says to investigate the symmetric family.
    
    # Let's try a range of M and sizes k for S
    # k is the number of elements in S (excluding 0 and M-x pairs)
    # Total size of U will be 1 + 2*k
    
    # We will use a local search for each (M, k) configuration
    # Start with a random S and improve it
    
    max_M = 150
    max_k = 5
    
    # Random seed for reproducibility in this run, but we want diversity
    # random.seed(42) 
    
    # Pre-compute some small sets that might be good?
    # The seed [0, 1, 3] has score 1.0.
    
    # Iterate over M
    for M in range(5, max_M + 1):
        if time.time() > deadline:
            break
            
        # Iterate over possible sizes of S
        for k in range(1, max_k + 1):
            if time.time() > deadline:
                break
                
            # We need to choose k elements from {1, ..., M-1}
            # Such that x != M-x (i.e., x != M/2 if M is even)
            # And x < M-x to avoid duplicates in the pair generation?
            # Actually, U = {0} U {x, M-x}. If we pick x, M-x is generated.
            # We should pick x such that x < M/2 to ensure distinct pairs.
            # Candidates for S: {1, ..., floor((M-1)/2)}
            
            half_M = M // 2
            if half_M < k:
                continue
                
            candidates = list(range(1, half_M + 1))
            
            # Perform a few random restarts for local search
            num_restarts = 5
            
            for _ in range(num_restarts):
                if time.time() > deadline:
                    break
                    
                # Initialize S randomly
                S = set(random.sample(candidates, k))
                
                # Local search: swap elements in S
                # Score calculation is expensive, so we limit iterations
                improved = True
                local_iter = 0
                max_local_iter = 100
                
                while improved and local_iter < max_local_iter:
                    improved = False
                    local_iter += 1
                    
                    if time.time() > deadline:
                        break
                    
                    # Build current U
                    U_curr = {0}
                    for x in S:
                        U_curr.add(x)
                        U_curr.add(M - x)
                    
                    curr_score = compute_score(list(U_curr))
                    
                    # Try swapping one element in S with one not in S
                    # To speed up, we can just try adding/removing or swapping
                    # Let's try swapping
                    for i, x_in in enumerate(list(S)):
                        for x_out in candidates:
                            if x_out in S:
                                continue
                            
                            new_S = set(S)
                            new_S.remove(x_in)
                            new_S.add(x_out)
                            
                            U_new = {0}
                            for x in new_S:
                                U_new.add(x)
                                U_new.add(M - x)
                            
                            new_score = compute_score(list(U_new))
                            
                            if new_score > curr_score:
                                S = new_S
                                curr_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    
                    # If we found an improvement, restart the inner loop
                    # Also try changing the size? No, k is fixed here.
                    
                    # Update global best
                    if curr_score > best_score:
                        best_score = curr_score
                        best_set = list(U_curr)
                        
    return best_set

# EVOLVE-BLOCK-START
def construct():
    return construct_impl()
# EVOLVE-BLOCK-END

def construct_impl():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Search parameters
    max_M = 200
    max_k = 6
    
    # We will search for sets of the form U = {0} U {x, M-x | x in S}
    # S is a subset of {1, ..., floor((M-1)/2)}
    
    for M in range(3, max_M + 1):
        if time.time() > deadline:
            break
            
        half_M = M // 2
        if half_M < 1:
            continue
            
        candidates = list(range(1, half_M + 1))
        
        # Try different sizes k for S
        for k in range(1, max_k + 1):
            if time.time() > deadline:
                break
                
            if len(candidates) < k:
                continue
                
            # Random restarts
            # We do a few restarts with local search
            num_restarts = 3
            
            for _ in range(num_restarts):
                if time.time() > deadline:
                    break
                    
                S = set(random.sample(candidates, k))
                
                # Local search
                improved = True
                local_iter = 0
                max_local_iter = 50
                
                while improved and local_iter < max_local_iter:
                    improved = False
                    local_iter += 1
                    
                    if time.time() > deadline:
                        break
                    
                    U_curr = {0}
                    for x in S:
                        U_curr.add(x)
                        U_curr.add(M - x)
                    
                    curr_score = compute_score(list(U_curr))
                    
                    # Try swapping
                    # We can also try adding/removing if we allow k to change, 
                    # but here k is fixed.
                    # Swapping is a good move.
                    
                    # To make it faster, we can sample a few swaps instead of all
                    # But k is small, so all swaps are fine.
                    swaps_tried = 0
                    max_swaps = 20
                    
                    # Get list of elements in S
                    S_list = list(S)
                    
                    # Try to improve by swapping one element
                    for x_in in S_list:
                        if swaps_tried >= max_swaps:
                            break
                        for x_out in random.sample(candidates, min(10, len(candidates))):
                            if x_out in S:
                                continue
                            swaps_tried += 1
                            
                            new_S = set(S)
                            new_S.remove(x_in)
                            new_S.add(x_out)
                            
                            U_new = {0}
                            for x in new_S:
                                U_new.add(x)
                                U_new.add(M - x)
                            
                            new_score = compute_score(list(U_new))
                            
                            if new_score > curr_score:
                                S = new_S
                                curr_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    
                    if curr_score > best_score:
                        best_score = curr_score
                        best_set = list(U_curr)
                        
    return best_set
