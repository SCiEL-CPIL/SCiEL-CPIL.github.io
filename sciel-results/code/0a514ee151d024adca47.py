import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Bitset representation
    # B[m] is an integer where bit s is 1 if sum s is achievable with m elements.
    # Max sum is 256 * 60 = 15360.
    # Python ints handle this easily.
    
    def is_valid(B, k):
        # Check if B[0..k] are pairwise disjoint
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if B[i] & B[j]:
                    return False
        return True

    def try_add(x, B, k):
        # Returns new B and new k if valid, else None
        # B is list of ints.
        # We update in reverse order to avoid using x multiple times in one step?
        # No, we are adding x to the set.
        # New sums of size m+1 come from old sums of size m.
        # We need to compute new_B.
        
        new_B = B[:]  # Copy
        # Update from high m to low m?
        # B_new[m] = B_old[m] | (B_old[m-1] << x)
        # If we update in place, B_old[m-1] must be available.
        # So iterate m from k down to 0.
        
        # Actually, we just need to check validity.
        # Let's compute the new bitsets.
        
        # Optimization: Only update up to k+1
        for m in range(k, -1, -1):
            if m > 0:
                new_B[m] = B[m] | (B[m-1] << x)
            else:
                new_B[0] = B[0] # 0 is always there
        
        # Check validity for m=0 to k+1
        if not is_valid(new_B, k + 1):
            return None
        
        return new_B, k + 1

    # Greedy decreasing
    candidates = list(range(256, 0, -1))
    
    best_set = []
    best_size = 0
    
    # We can try multiple random permutations if time allows
    # But first, try the deterministic one
    
    B = [0] * 61
    B[0] = 1 # Sum 0
    k = 0
    current_set = []
    
    for x in candidates:
        res = try_add(x, B, k)
        if res:
            B, k = res
            current_set.append(x)
            if k > best_size:
                best_size = k
                best_set = current_set[:]
        
        if time.time() > deadline:
            break

    # If we have time, try some random restarts
    # Or local search: try to swap elements?
    # Given the greedy nature, it's hard to improve without breaking validity.
    # But maybe starting with a different order helps?
    
    # Let's try a few random shuffles of the candidates
    # We must ensure we finish before deadline.
    
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        # Try 10 random restarts
        for _ in range(10):
            if time.time() > deadline - 0.5:
                break
            random.shuffle(candidates)
            B = [0] * 61
            B[0] = 1
            k = 0
            current_set = []
            
            for x in candidates:
                res = try_add(x, B, k)
                if res:
                    B, k = res
                    current_set.append(x)
                    if k > best_size:
                        best_size = k
                        best_set = current_set[:]
                
                if time.time() > deadline - 0.5:
                    break

    # Ensure types are correct
    return [int(x) for x in best_set]
