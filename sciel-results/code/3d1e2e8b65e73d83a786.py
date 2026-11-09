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
            if time.time() > DEADLINE - 20:
                break
            
            # Check if x can be added: all new triple sums involving x must be unique
            # and not conflict with existing triple sums
            # New triples involving x: (x,x,x), (x,x,y) for y in selected, (x,y,z) for y<=z in selected
            
            # (x,x,x)
            s1 = 3 * x
            if s1 in triple_sums:
                continue
            
            # (x,x,y) for y in selected
            valid = True
            for y in selected:
                s = 2 * x + y
                if s in triple_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            # (x,y,z) for y <= z in selected
            for i in range(len(selected)):
                y = selected[i]
                for j in range(i, len(selected)):
                    z = selected[j]
                    s = x + y + z
                    if s in triple_sums:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # All checks passed, add x
            selected.append(x)
            triple_sums.add(s1)
            for y in selected[:-1]:
                triple_sums.add(2 * x + y)
            for i in range(len(selected) - 1):
                y = selected[i]
                for j in range(i, len(selected) - 1):
                    z = selected[j]
                    triple_sums.add(x + y + z)
        
        return selected

    # Try greedy from 1
    result = build_greedy(start_from=1)
    
    # Try to improve by local search: try swapping elements to fit more
    if time.time() < DEADLINE - 30:
        # Try different starting points
        for start in [2, 3, 4, 5]:
            if time.time() > DEADLINE - 30:
                break
            candidate = build_greedy(start_from=start)
            if len(candidate) > len(result):
                result = candidate
    
    return result
# EVOLVE-BLOCK-END
