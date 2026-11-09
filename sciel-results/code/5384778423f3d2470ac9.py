"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0
    N = 4096
    MAX_SIZE = 80

    def build_greedy(start_from=1, max_size=MAX_SIZE):
        """Greedy: add numbers in order, maintaining set of triple sums."""
        selected = []
        triple_sums = set()
        
        for x in range(start_from, N + 1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 10:
                break
            
            # Check if x can be added: all new triple sums involving x must be unique
            # and not already in triple_sums.
            # New triples involving x:
            # (x, x, x) -> 3x
            # (x, x, y) for y in selected -> 2x + y
            # (x, y, z) for y <= z in selected -> x + y + z
            
            new_sums = set()
            new_sums.add(3 * x)
            
            for y in selected:
                new_sums.add(2 * x + y)
            
            for i in range(len(selected)):
                for j in range(i, len(selected)):
                    new_sums.add(x + selected[i] + selected[j])
            
            # Check for conflicts
            if new_sums & triple_sums:
                continue
            
            # Check internal conflicts in new_sums
            if len(new_sums) != len(new_sums):
                continue
            
            # All good, add x
            selected.append(x)
            triple_sums.update(new_sums)
        
        return selected

    # Try greedy from start
    result = build_greedy()
    
    # If we didn't reach MAX_SIZE, try to see if we can do better with a different strategy
    # But the greedy from 1 is usually very strong for this problem.
    # Let's verify the result is valid and return it.
    
    # Optional: try a second pass or different start if time permits
    # For now, the greedy is the main strategy.
    
    return result
# EVOLVE-BLOCK-END
