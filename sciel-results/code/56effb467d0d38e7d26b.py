import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    best_set = []
    max_val = 512
    target_size = 48
    
    # We will run multiple random restarts
    # Each restart tries to build a set greedily
    
    num_restarts = 200  # High number, but each is fast
    
    for r in range(num_restarts):
        if time.time() - start_time > deadline:
            break
        
        # Create a random order of candidates
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        S = []
        # dp[k] is an integer bitset where bit s is 1 if sum s is achievable with k elements
        # Max sum approx 48 * 512 = 24576
        # We need dp up to size len(S)
        # Initialize dp for empty set
        dp = [0] * (target_size + 1)
        dp[0] = 1  # bit 0 set for 0 elements
        
        # Precomputed pair sums for fast check
        # Since S changes, we can just iterate S for pair checks if |S| is small
        # Or maintain a set of pair sums.
        # Set of pair sums is O(N^2) to build, O(1) to check.
        # Updating is O(N).
        pair_sums = set()
        
        for x in candidates:
            if time.time() - start_time > deadline - 1:
                break
            
            if len(S) >= target_size:
                break
                
            # 1. Check if x is valid with current S
            # 1a. x is not mean of any subset of S (size >= 2)
            valid = True
            
            # Check pair condition for x: 2x not in pair_sums
            if 2 * x in pair_sums:
                valid = False
            
            # Check subset sums for x (size >= 2)
            if valid:
                for k in range(2, len(S) + 1):
                    target_sum = k * x
                    if target_sum > 25000: # Optimization: max possible sum
                        break
                    if (dp[k] >> target_sum) & 1:
                        valid = False
                        break
            
            # 1b. Check if any y in S becomes mean of a subset involving x
            if valid:
                for y in S:
                    # y is mean of subset T' U {x}, T' subset of S \ {y}
                    # Sum(T') + x = (|T'| + 1) * y
                    # Sum(T') = (k+1)*y - x, where k = |T'|
                    # k can range from 1 to len(S)-1
                    # Note: T' must not contain y. 
                    # Using dp of S might include y. 
                    # If it does, it's a false positive (reject valid x).
                    # This is acceptable for now.
                    
                    # Also check pair condition: 2y = x + b => b = 2y - x
                    # If b in S and b != x and b != y, then y is mean of {x, b}
                    b = 2 * y - x
                    if b in S and b != x and b != y:
                        valid = False
                        break
                    
                    if not valid:
                        break
                    
                    # Check larger subsets for y
                    for k in range(1, len(S)):
                        target_sum = (k + 1) * y - x
                        if target_sum < 0:
                            continue
                        if target_sum > 25000:
                            break
                        if (dp[k] >> target_sum) & 1:
                            # Potential violation
                            # We assume T' does not contain y
                            valid = False
                            break
                    
                    if not valid:
                        break
            
            if valid:
                # Add x to S
                S.append(x)
                
                # Update pair_sums
                for a in S[:-1]: # All elements except x
                    pair_sums.add(a + x)
                
                # Update dp
                # Iterate k in reverse to avoid using x multiple times
                for k in range(len(S) - 1, -1, -1):
                    # dp[k+1] |= dp[k] << x
                    # Only if dp[k] is non-zero
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
        
        if len(S) > len(best_set):
            best_set = S[:]
            
        # Optional: If we found 48, we can stop early? 
        # The task says "Maximize cardinality". 48 is the max possible?
        # "At most 48 integers". So 48 is the ceiling.
        if len(best_set) >= 48:
            break
            
    # Final verification and return
    # The problem asks to return the set.
    # We should ensure the returned set is valid.
    # The greedy check was conservative (false positives allowed), so it should be valid.
    # But let's do a quick sanity check if time permits?
    # The checks were sufficient to prevent known violations.
    
    return best_set

# Helper for local testing or if we want to be extra sure
def _is_valid_set(S):
    """
    Full verification.
    """
    if len(S) < 3:
        return True
        
    S = list(S)
    N = len(S)
    
    for i in range(N):
        a = S[i]
        others = S[:i] + S[i+1:]
        n_others = len(others)
        
        if n_others < 2:
            continue
            
        max_sum = sum(others)
        
        dp = [0] * (n_others + 1)
        dp[0] = 1
        
        for val in others:
            for k in range(n_others - 1, -1, -1):
                if dp[k] == 0:
                    continue
                dp[k+1] |= (dp[k] << val)
                
        for k in range(2, n_others + 1):
            target_sum = k * a
            if target_sum <= max_sum:
                if (dp[k] >> target_sum) & 1:
                    return False
            else:
                # If target_sum > max_sum, impossible
                pass
                
    return True

# Run once to ensure it works? No, construct() is the entry point.
# But we can add a fallback if construct returns invalid set?
# The greedy method with false positives only should return a valid set.
# If it returns an invalid set, it means there was a logic error in the check.
# Given the checks cover all subset sizes and pair sums, it should be fine.

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
    print(_is_valid_set(res))
