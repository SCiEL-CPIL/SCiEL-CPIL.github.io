import time
import random
from itertools import combinations

def exact_is_valid(A):
    """
    Checks if set A is valid using exact subset-sum DP.
    Returns True if valid, False otherwise.
    """
    A = list(A)
    n = len(A)
    if n < 3:
        return True
        
    # For each element a, check if it is the mean of any subset of others
    for i, a in enumerate(A):
        others = [x for j, x in enumerate(A) if j != i]
        # Shift values by -a to check for sum 0
        # y = x - a
        Y = [x - a for x in others]
        
        # dp[k] is a set of sums achievable with k elements from Y
        # We only care about sums that could be 0.
        # Max sum magnitude: 512 * 48 ~ 24576
        # We use a dictionary or set of sums.
        
        # Optimization: If 0 is in dp[k] for any k >= 2, it's invalid.
        # Initialize dp
        # dp[0] = {0}
        # We will iterate through Y and update dp.
        
        # To save memory and time, we can use a list of sets.
        # Max size of others is n-1 <= 47.
        dp = [set() for _ in range(len(Y) + 1)]
        dp[0].add(0)
        
        for y in Y:
            # Update from high count to low count to avoid using y multiple times in same step
            for k in range(len(Y), 0, -1):
                if dp[k-1]:
                    # Add y to all sums in dp[k-1]
                    # We only care if 0 is reached, but we need the sums for future steps.
                    # We can prune sums that are too large? 
                    # Since we are checking for 0, and values are bounded, we can keep all.
                    new_sums = set()
                    for s in dp[k-1]:
                        ns = s + y
                        new_sums.add(ns)
                    dp[k].update(new_sums)
                    
                    # Early exit: if 0 is in dp[k] and k >= 2
                    if 0 in dp[k] and k >= 2:
                        return False
                        
    return True

def heuristic_is_valid(A):
    """
    Fast check for subsets of size 2, 3, 4.
    """
    A = list(A)
    n = len(A)
    if n < 3:
        return True
        
    for i, a in enumerate(A):
        others = [x for j, x in enumerate(A) if j != i]
        
        # Check size 2
        # x + y = 2a
        others_set = set(others)
        for x in others:
            if (2*a - x) in others_set and (2*a - x) != x:
                return False
            # Note: if 2*a - x == x, then 2x = 2a => x=a, but x in others so x!=a.
            # So x and 2*a-x are distinct.
            
        # Check size 3
        # x + y + z = 3a
        # Iterate pairs
        m = len(others)
        for p in range(m):
            x = others[p]
            for q in range(p+1, m):
                y = others[q]
                z = 3*a - x - y
                if z in others_set and z != x and z != y:
                    return False
                # Note: z could be equal to x or y?
                # If z == x, then 2x + y = 3a. x, y, x is not a subset of distinct elements.
                # The subset must consist of distinct elements from A.
                # So we need distinct x, y, z.
        
        # Check size 4
        # x + y + z + w = 4a
        # Iterate triples
        # This is O(n^3) per element, so O(n^4) total.
        # For n=40, 40^4 = 2.5M. Might be slow in Python if done many times.
        # We will skip size 4 in the heuristic to keep it fast, or limit it.
        # Given the risk, let's include it but optimize.
        
        # Alternative for size 4:
        # Precompute pair sums of others?
        # If we have pair sums, we check if 4a - (x+y) is a pair sum.
        pair_sums = set()
        for p in range(m):
            for q in range(p+1, m):
                pair_sums.add(others[p] + others[q])
                
        for p in range(m):
            x = others[p]
            for q in range(p+1, m):
                y = others[q]
                target = 4*a - x - y
                # Check if target can be formed by two distinct elements z, w from others, 
                # distinct from x and y.
                # This is tricky with just a set of sums.
                # We'll skip size 4 in heuristic to ensure speed, relying on exact check at the end.
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_set = []
    best_size = 0
    
    # Candidate pool: 1..512
    candidates = list(range(1, 513))
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        random.shuffle(candidates)
        
        current_set = []
        
        for x in candidates:
            # Check if adding x is valid using heuristic
            # We use a temporary list to avoid modification issues
            test_set = current_set + [x]
            
            if heuristic_is_valid(test_set):
                current_set = test_set
                
        # After greedy, verify with exact check
        if len(current_set) > best_size:
            if exact_is_valid(current_set):
                best_set = current_set
                best_size = len(current_set)
                # If we hit a high size, we might want to continue to see if we can beat it
                # But if we are close to deadline, we should break
                if time.time() > deadline - 5:
                    break
            else:
                # If invalid, we can try to trim it
                # Find offending elements and remove them?
                # For simplicity, we just discard this candidate for now
                pass
                
        # If we've done a few restarts and found a large set, we can stop early
        # if we are confident, but random restarts are cheap.
        
    return best_set

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    print(f"Found set of size {len(s)}")
