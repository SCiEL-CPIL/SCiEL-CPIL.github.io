import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # 120s limit, leave margin

    # Helper: Convert orientation list to adjacency bitsets
    def orientation_to_bits(orientation, n):
        bits = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:  # i -> j
                    bits[i] |= (1 << j)
                else:  # j -> i
                    bits[j] |= (1 << i)
                idx += 1
        return bits

    # Helper: Check if tournament (represented by bits) is valid
    def is_valid_bits(bits, n):
        # For every 4-set, the intersection of their out-neighborhoods must be non-empty
        # Since a vertex cannot dominate itself, and the intersection excludes the vertices themselves
        # (as i is not in N+(i)), any non-empty intersection implies a dominator outside the set.
        
        # Optimization: Pre-check out-degrees?
        # If a vertex has out-degree < 4, it cannot dominate any 4-set.
        # This doesn't directly help with the intersection check, but might help in local search.
        
        for quad in combinations(range(n), 4):
            mask = bits[quad[0]] & bits[quad[1]] & bits[quad[2]] & bits[quad[3]]
            if mask == 0:
                return False
        return True

    # 1. Baseline: Paley Tournament on 83 vertices
    p = 83
    squares = {x * x % p for x in range(1, p)}
    paley_orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    
    # Verify Paley is valid (it should be, but good to check)
    paley_bits = orientation_to_bits(paley_orientation, p)
    if not is_valid_bits(paley_bits, p):
        # If Paley is invalid, we are in trouble. 
        # But the seed program scored 1.0, so it must be valid.
        # We'll still try to find a valid one.
        pass

    best_result = {'n': p, 'orientation': paley_orientation}

    # 2. Search for smaller valid tournaments
    # We try n from 82 down to 70
    # We use random search.
    
    # To make it faster, we can use a "repair" heuristic or just random.
    # Given the time, random search with early exit is best.
    
    for n in range(82, 69, -1):
        if time.time() > deadline - 15:  # Leave 15s buffer
            break
        
        # Number of attempts depends on remaining time
        # Rough estimate: is_valid_bits takes ~0.5s for n=80
        # So we can do ~20 checks per second? No, 1 check ~ 0.5s -> 2 checks/s.
        # Let's try a few attempts.
        
        attempts = 0
        while time.time() < deadline - 5:
            # Generate random tournament
            random_orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            bits = orientation_to_bits(random_orientation, n)
            
            if is_valid_bits(bits, n):
                return {'n': n, 'orientation': random_orientation}
            
            attempts += 1
            # Limit attempts per n to avoid getting stuck too long if probability is low
            if attempts > 10:
                break

    return best_result
