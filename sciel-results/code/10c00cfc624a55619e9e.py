import time
import random

DEADLINE_SECONDS = 85.0

def _try_greedy(order, max_size=60):
    """
    Attempt to build a valid set by adding elements in the specified order.
    Returns the set if a valid set of size > 0 is found, else None.
    """
    N = max_size
    # masks[k] holds sums achievable with exactly k elements
    # We use Python ints as bitsets.
    # Max sum approx 256 * 60 = 15360. Bit length ~ 15360.
    masks = [0] * (N + 1)
    masks[0] = 1  # Sum 0 with 0 elements
    
    S = []
    
    for x in order:
        if x < 1 or x > 256:
            continue
        if x in S:
            continue
            
        # Calculate new sums P_k = masks[k-1] << x
        # We only need to check up to len(S) + 1
        current_len = len(S)
        if current_len >= max_size:
            break
            
        P = [0] * (current_len + 2)
        for k in range(1, current_len + 2):
            if k - 1 <= current_len:
                P[k] = masks[k-1] << x
            else:
                P[k] = 0
        
        # Check validity
        # 1. New sums P_i must not intersect with old masks M_j for i != j
        # 2. New sums P_i must not intersect with new sums P_j for i != j
        
        valid = True
        
        # Check P_i vs M_j
        for i in range(1, current_len + 2):
            if P[i] == 0:
                continue
            for j in range(0, current_len + 1):
                if i == j:
                    continue
                if masks[j] & P[i]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Check P_i vs P_j
            for i in range(1, current_len + 2):
                if P[i] == 0:
                    continue
                for j in range(i + 1, current_len + 2):
                    if P[j] == 0:
                        continue
                    if P[i] & P[j]:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Update masks
            for k in range(1, current_len + 2):
                masks[k] |= P[k]
            S.append(x)
            
    return S

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Standard ascending order
    if time.time() < deadline - 1.0:
        S = _try_greedy(list(range(1, 257)))
        if S and len(S) > best_size:
            best_set = S
            best_size = len(S)
            
    # Strategy 2: Descending order
    if time.time() < deadline - 1.0:
        S = _try_greedy(list(range(256, 0, -1)))
        if S and len(S) > best_size:
            best_set = S
            best_size = len(S)
            
    # Strategy 3: Randomized Greedy Restarts
    # We can run many random permutations or random subset selections.
    # Since N is small, full permutation is expensive.
    # Instead, we can randomize the order of candidates.
    
    # Try a few random orders
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        # Shuffle 1..256
        order = list(range(1, 257))
        random.shuffle(order)
        
        S = _try_greedy(order)
        if S and len(S) > best_size:
            best_set = S
            best_size = len(S)
            
        # If we found a very large set, we might stop early or continue to try for even larger
        if best_size >= 60:
            break
            
    # Ensure return type is list of ints
    if best_set:
        return [int(x) for x in best_set]
    else:
        # Fallback: A known valid small set
        # {1, 2, 4, 8} is valid?
        # 0: 0
        # 1: 1, 2, 4, 8
        # 2: 3, 5, 9, 6, 10, 12
        # 3: 7, 11, 13, 14
        # 4: 15
        # Sums are distinct across cardinalities?
        # 0: {0}
        # 1: {1,2,4,8}
        # 2: {3,5,6,9,10,12}
        # 3: {7,11,13,14}
        # 4: {15}
        # No overlaps.
        return [1, 2, 4, 8, 16, 32, 64, 128]

# EVOLVE-BLOCK-END
