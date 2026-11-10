import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    best_score = (float('inf'), float('inf'))
    best_A = None
    
    # Range of sizes to try
    # Smaller k -> fewer pairs -> lower average multiplicity, but harder to cover
    # Larger k -> more pairs -> higher average multiplicity, easier to cover
    for k in range(14, 28):
        # Try multiple random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            A = list(set(random.sample(range(129), k)))
            
            # Local Search
            current_score, current_A = local_search(A, deadline)
            
            if current_score < best_score:
                best_score = current_score
                best_A = current_A
                
            # If we have a valid set (gaps=0), we can stop looking for larger k 
            # if peak is very low? 
            # But smaller k might yield lower peak. 
            # Actually, we are iterating k from small to large.
            # If we find peak=1, we can stop everything.
            if best_score[0] == 0 and best_score[1] == 1:
                return sorted(best_A)
                
    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
        
    return sorted(best_A)

def calc_score(A):
    A = sorted(A)
    mult = np.zeros(257, dtype=np.int32)
    for i in range(len(A)):
        for j in range(i, len(A)):
            mult[A[i] + A[j]] += 1
    gaps = np.sum(mult[128:257] == 0)
    if gaps > 0:
        return (gaps, float('inf'))
    peak = int(np.max(mult[128:257]))
    return (0, peak)

def local_search(A, deadline):
    A = list(A)
    current_score = calc_score(A)
    
    # If initial is already perfect (0, 1), return
    if current_score == (0, 1):
        return current_score, A
        
    improved = True
    # Max iterations to prevent infinite loops in local minima
    max_iters = 500
    
    while improved and time.time() < deadline and max_iters > 0:
        improved = False
        max_iters -= 1
        
        # Generate neighbors by swapping one element
        # To speed up, we can shuffle A to avoid deterministic bias
        # But we need to check all swaps to find the best one (First Improvement vs Best Improvement)
        # Best Improvement is slower but better. 
        # With k~20, 100 neighbors. Fast enough.
        
        best_neighbor_score = current_score
        best_neighbor_A = A
        
        for i in range(len(A)):
            a = A[i]
            # Try swapping with b not in A
            # Sample a few random b's or iterate all?
            # Iterate all is 129-k ~ 100 swaps per i. Total 2000 swaps.
            # Each swap check is O(k^2) ~ 400 ops. 800,000 ops. ~0.1s in Python? 
            # Maybe too slow for 500 iterations.
            # Let's try random sampling of neighbors.
            
            # Try 10 random swaps per element? Or just global random swaps?
            # Let's try 5 random swaps per element.
            candidates = random.sample([x for x in range(129) if x not in A], min(5, 129-len(A)))
            for b in candidates:
                new_A = A.copy()
                new_A[i] = b
                new_score = calc_score(new_A)
                if new_score < best_neighbor_score:
                    best_neighbor_score = new_score
                    best_neighbor_A = new_A
                    
        if best_neighbor_score < current_score:
            current_score = best_neighbor_score
            A = best_neighbor_A
            improved = True
            if current_score == (0, 1):
                break
                
    return current_score, A
