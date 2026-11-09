import time
import random

START_TIME = time.time()
DEADLINE = 58.0

def _remaining():
    return DEADLINE - (time.time() - START_TIME)

def check_validity(A):
    n = len(A)
    if n == 0:
        return True
    
    # dp[k] is bitmask of sums
    # Max sum 256 * 60 = 15360
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for a in A:
        # Update in reverse
        for k in range(n - 1, -1, -1):
            if dp[k]:
                dp[k+1] |= (dp[k] << a)
                
    # Check for conflicts
    # A sum s is in M_i and M_j (i!=j) iff it is counted more than once in the union of popcounts
    # Union of masks
    union_mask = 0
    total_bits = 0
    for k in range(n + 1):
        total_bits += bin(dp[k]).count('1')
        union_mask |= dp[k]
        
    if total_bits != bin(union_mask).count('1'):
        return False
    return True

def construct():
    if _remaining() < 0.5:
        return [1, 2, 4, 8, 16, 32, 64, 128]
        
    # 1. Greedy Initialization
    # Start with powers of 2, then try to add smallest available
    A = [1, 2, 4, 8, 16, 32, 64, 128]
    A_set = set(A)
    
    # Try to add more greedily
    candidates = [x for x in range(1, 257) if x not in A_set]
    candidates.sort()
    
    improved = True
    while improved and _remaining() > 1.0:
        improved = False
        for c in candidates:
            if c in A_set:
                continue
            new_A = A + [c]
            if check_validity(new_A):
                A = new_A
                A_set.add(c)
                improved = True
                break # Restart candidate loop with new set
                
    best_A = list(A)
    best_size = len(A)
    
    # 2. Local Search with Swaps
    # We will try to swap elements to find a better configuration
    # that allows more elements to be added.
    
    # To speed up, we can pre-compute something? No, check is fast enough.
    # We'll run a loop for a certain number of iterations or until time runs out.
    
    rng = random.Random(42)
    
    # We'll use a "stagnation" counter. If we don't improve size, we might need to perturb.
    # However, the goal is just to maximize size.
    
    # Strategy: 
    # While time remains:
    #   1. Try all swaps (a in A, c not in A). If valid, accept and try to add more.
    #   2. If no swap works, try a random perturbation (swap with random c) and accept if valid (even if size doesn't increase immediately, it might open up paths).
    #   3. Keep track of best size.
    
    A = list(best_A)
    A_set = set(A)
    
    while _remaining() > 0.5:
        # Try to add elements first
        candidates = [x for x in range(1, 257) if x not in A_set]
        candidates.sort()
        added = False
        for c in candidates:
            if _remaining() < 0.5:
                break
            if check_validity(A + [c]):
                A.append(c)
                A_set.add(c)
                added = True
                break
        
        if len(A) > best_size:
            best_A = list(A)
            best_size = len(A)
            
        if added:
            continue # Restart search with new set
        
        # If no addition, try swaps
        found_swap = False
        # Shuffle A to avoid bias? Or iterate deterministically?
        # Deterministic is faster but might loop. Random is better for escaping.
        # Let's iterate through a random permutation of A for swaps.
        A_perm = A[:]
        rng.shuffle(A_perm)
        
        for a in A_perm:
            if _remaining() < 0.5:
                break
            # Try swapping a with a random candidate
            # To be efficient, we can pick a few random candidates for each a
            cands_for_swap = [x for x in range(1, 257) if x not in A_set]
            rng.shuffle(cands_for_swap)
            
            # Limit attempts per 'a' to avoid spending too much time on one element
            # If we find a valid swap, we accept it and restart the whole process.
            for c in cands_for_swap[:20]: # Try up to 20 random swaps for this 'a'
                if _remaining() < 0.5:
                    break
                new_A = [x for x in A if x != a]
                new_A.append(c)
                if check_validity(new_A):
                    # Accept swap
                    A = new_A
                    A_set = set(A)
                    found_swap = True
                    break
            if found_swap:
                break
                
        if not found_swap:
            # Stuck. Try a more aggressive perturbation?
            # Or just accept that we are in a local optimum and try to restart from best_A?
            # Or try to remove an element and see if we can add two?
            # Let's try a random restart from best_A with a different seed/order
            # But first, let's try "remove and add" which is effectively a swap.
            # If simple swaps don't work, maybe we need to remove 1 and add 1 (same as swap) 
            # or remove 2 and add 2?
            
            # Let's try a random restart from a subset of best_A
            if _remaining() > 5.0:
                # Pick a random subset of best_A of size best_size - 2
                # And try to grow it
                if best_size > 4:
                    subset_size = best_size - 2
                    subset = rng.sample(best_A, subset_size)
                    A = subset
                    A_set = set(A)
                    # Try to grow
                    # This effectively resets the search trajectory
                else:
                    # If best size is small, just break
                    break
            else:
                break
                
    # Final check
    if check_validity(best_A):
        return [int(x) for x in best_A]
    else:
        # Fallback
        return [1, 2, 4, 8, 16, 32, 64, 128]
