"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85
    n = 12
    MAX_BOUND = 1048576
    
    # Start with powers of 2 as the known feasible solution
    best = [2**i for i in range(n)]
    best_max = 2**11  # 2048
    
    # Try to find a better solution via backtracking
    # Key insight: use a set of subset sums, and for each new element x,
    # check that {x+s for s in sums} is disjoint from sums.
    
    def backtrack(idx, current_set, subset_sums, current_max):
        nonlocal best, best_max
        if time.time() > deadline:
            return
        if idx == n:
            if current_max < best_max:
                best_max = current_max
                best = list(current_set)
            return
        
        # Prune: can't beat best_max
        if current_max >= best_max:
            return
        
        start = 1
        if current_set:
            start = current_set[-1] + 1
        
        end = min(best_max, MAX_BOUND + 1)
        
        for x in range(start, end):
            if time.time() > deadline:
                return
            # Quick check: x must not be in subset_sums (since x+0=x)
            if x in subset_sums:
                continue
            # Full check
            valid = True
            for s in subset_sums:
                if s != 0 and x + s in subset_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            new_sums = set(subset_sums)
            for s in subset_sums:
                new_sums.add(x + s)
            
            new_max = max(current_max, x)
            backtrack(idx + 1, current_set + [x], new_sums, new_max)
    
    backtrack(0, [], {0}, 0)
    return best
# EVOLVE-BLOCK-END
