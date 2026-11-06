import time
import random

def _all_subset_sums_distinct(seq):
    """
    Check if all subset sums are distinct using a bitset.
    Returns True if valid.
    """
    n = len(seq)
    bits = 1
    for a in seq:
        bits |= (bits << a)
    return bin(bits).count('1') == (1 << n)

def _construct_valid(n, deadline):
    """
    Construct a valid sequence using randomized local search.
    """
    best_seq = [1 << i for i in range(n)]
    best_max = max(best_seq)
    
    # Try multiple random restarts
    while time.time() < deadline:
        # Start with a random valid-like sequence
        seq = [1]
        s = 1
        for i in range(1, n):
            # Pick a value that maintains distinctness potential
            # Use superincreasing-like but with randomization
            min_val = 1
            max_val = min(s + 1, 2048)  # Upper bound
            if max_val <= min_val:
                max_val = min_val + 1
            val = random.randint(min_val, max_val)
            # Ensure distinct
            while val in seq:
                val = random.randint(min_val, max_val)
                if max_val <= min_val:
                    break
            seq.append(val)
            s += val
        
        # Local search to improve
        current_seq = list(seq)
        current_max = max(current_seq)
        
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, n - 1)
            old_val = current_seq[idx]
            
            # Try various changes
            for delta in random.sample(range(1, 50), min(10, 49)):
                for sign in [-1, 1]:
                    new_val = old_val + sign * delta
                    if new_val <= 0 or new_val in current_seq:
                        continue
                    
                    current_seq[idx] = new_val
                    if _all_subset_sums_distinct(current_seq):
                        new_max = max(current_seq)
                        if new_max < current_max:
                            current_max = new_max
                            break
                    else:
                        current_seq[idx] = old_val
                        break
            
            if time.time() > deadline:
                break
        
        if current_max < best_max:
            best_max = current_max
            best_seq = list(current_seq)
    
    return best_seq

def construct():
    n = 12
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Start with powers of 2
    seq = [1 << i for i in range(n)]
    best_seq = list(seq)
    best_max = max(seq)  # 2048
    
    # Try randomized local search with multiple restarts
    improved_seq = _construct_valid(n, deadline)
    improved_max = max(improved_seq)
    
    if improved_max < best_max:
        best_max = improved_max
        best_seq = improved_seq
    
    return best_seq
