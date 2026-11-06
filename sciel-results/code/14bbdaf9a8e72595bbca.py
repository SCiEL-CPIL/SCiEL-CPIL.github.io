import time
import random

def _all_subset_sums_distinct(seq):
    """Check if all subset sums are distinct. Returns True if valid."""
    n = len(seq)
    bits = 1  # only sum 0
    for a in seq:
        bits |= (bits << a)
    cnt = bin(bits).count('1')
    return cnt == (1 << n)

def construct():
    """
    Construct 12 distinct positive integers <= 1048576 with all subset sums distinct.
    Minimize the largest element.
    """
    n = 12
    max_val = 1048576
    
    # Theoretical minimum is 2^(n-1) = 2048.
    # We start with the optimal candidate: powers of 2.
    best_seq = [1 << i for i in range(n)]
    best_max = max(best_seq)
    
    deadline = 85.0
    start_time = time.time()
    
    # Try to find a better solution (max < 2048) using local search.
    # This is unlikely to succeed if the theoretical minimum holds, 
    # but we attempt it to be safe.
    
    # We will try to reduce the largest element.
    # If we can't beat 2048, we return 2048.
    
    # Simple local search: try to replace the largest element with a smaller one
    # and adjust others if necessary? 
    # Actually, just random restarts with a bias towards smaller numbers.
    
    rng = random.Random(123)
    
    # If we have time, try random construction with a hard cap of best_max - 1
    target_max = best_max - 1
    
    if time.time() - start_time < deadline - 5:
        for _ in range(100): # 100 restarts
            if time.time() - start_time > deadline - 5:
                break
            
            seq = []
            valid = True
            # Greedy random: pick smallest available that doesn't break constraint
            # but with some randomness
            current_sum = 0
            for i in range(n):
                # Try values from 1 up to target_max
                # We focus on smaller values to keep max low
                found = False
                # Sort candidates: prioritize smaller values, but add some noise
                # To find a valid set, we need to be careful.
                # If we pick too small, we might get stuck.
                # If we pick too large, we exceed target.
                
                # Strategy: Try all values from 1 to target_max? 
                # 2048 * 12 is too slow for 100 restarts.
                
                # Instead, use a heuristic: 
                # The sum of previous elements is S.
                # The next element must be such that no collision occurs.
                # Collisions happen if x is in current sums or x+s is in current sums.
                
                # For speed, we just try a few random candidates in the range [1, target_max]
                # and check validity.
                
                candidates = set()
                # Add small numbers
                for v in range(1, min(50, target_max + 1)):
                    candidates.add(v)
                # Add random numbers
                for _ in range(20):
                    candidates.add(rng.randint(1, target_max))
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for val in cand_list:
                    if val in seq:
                        continue
                    test_seq = seq + [val]
                    if _all_subset_sums_distinct(test_seq):
                        seq.append(val)
                        found = True
                        break
                
                if not found:
                    valid = False
                    break
            
            if valid:
                cur_max = max(seq)
                if cur_max < best_max:
                    best_seq = seq
                    best_max = cur_max
                    if best_max == 2047: # Theoretical limit check
                        break

    return best_seq
