"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
import time

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcds for speed
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        S_list = list(S)
        for i in range(len(S_list)):
            y = S_list[i]
            g_xy = gcd_table[x][y]
            for j in range(i + 1, len(S_list)):
                z = S_list[j]
                g_xz = gcd_table[x][z]
                if g_xy == g_xz:
                    if gcd_table[y][z] == g_xy:
                        return False
        return True

    def greedy_fill(S, candidates):
        """Greedily add elements from candidates to S if possible."""
        S = S.copy()
        random.shuffle(candidates)
        for x in candidates:
            if x not in S:
                if can_add(x, S):
                    S.add(x)
        return S

    best_set = set()
    max_size = 0
    
    start_time = time.time()
    timeout = 55.0
    
    # Run multiple random restarts
    while time.time() - start_time < timeout:
        # Start with an empty set
        current_set = set()
        
        # Greedy fill with random order
        candidates = numbers.copy()
        current_set = greedy_fill(current_set, candidates)
        
        # Local Search / Hill Climbing
        improved = True
        while improved and (time.time() - start_time < timeout):
            improved = False
            
            # Try to add any single element
            missing = [x for x in numbers if x not in current_set]
            if missing:
                random.shuffle(missing)
                for x in missing:
                    if can_add(x, current_set):
                        current_set.add(x)
                        improved = True
                        break # Restart check to find more additions
                
            if not improved:
                # Stuck. Perform a kick: remove a random element and refill
                if current_set:
                    # Remove a random element
                    r = random.choice(list(current_set))
                    current_set.remove(r)
                    
                    # Refill greedily
                    missing = [x for x in numbers if x not in current_set]
                    new_set = greedy_fill(current_set, missing)
                    
                    # If new set is better, keep it
                    if len(new_set) > len(current_set):
                        current_set = new_set
                        improved = True # Continue trying to add more
                    # Else, we stay at the current (smaller) set and try another kick?
                    # Or revert? If we don't revert, we might drift.
                    # Let's revert to avoid getting stuck in a smaller set if it was a bad kick.
                    # Actually, if new_set is not larger, it's likely the same size or smaller.
                    # If smaller, it's bad. If same, it's a neutral move.
                    # We'll just try another kick in the next iteration.
                else:
                    break

        if len(current_set) > max_size:
            max_size = len(current_set)
            best_set = current_set.copy()
            
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
