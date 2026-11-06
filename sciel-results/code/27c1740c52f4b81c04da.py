"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def is_valid(subset):
    """Check if no subset of 'subset' sums to 200."""
    reachable = {0}
    for x in subset:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 in new:
            return False
        reachable |= new
    return True

def find_conflict(subset, x):
    """Find a subset of 'subset' that sums to 200 - x."""
    target = 200 - x
    if target < 0:
        return []
    
    # DP to find one subset
    # dp[s] = list of elements used to reach sum s
    # Since T=200 is small, we can just store boolean and backtrack
    reachable = {0: []}
    for y in subset:
        new_reachable = {}
        for s, elems in reachable.items():
            ns = s + y
            if ns <= target:
                if ns not in new_reachable:
                    new_reachable[ns] = elems + [y]
        # Keep existing reachable sums if not improved (or just merge)
        # Actually, we just need ONE path.
        for s, elems in new_reachable.items():
            if s not in reachable:
                reachable[s] = elems
    
    if target in reachable:
        return reachable[target]
    return []

def construct():
    # 1. Greedy initialization
    selected = []
    reachable = {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    
    # 2. Local Search to improve
    # We will try to add elements or swap
    improved = True
    while improved:
        improved = False
        # Randomize order for diversity
        candidates = [x for x in range(1, 101) if x not in selected]
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if we can just add x
            if is_valid(selected + [x]):
                selected.append(x)
                improved = True
                break # Restart with new state
            
            # If not, try to swap
            # Find conflict subset
            conflict = find_conflict(selected, x)
            if not conflict:
                # Should not happen if is_valid failed, but safety
                continue
                
            # Try removing one element from conflict
            random.shuffle(conflict)
            for y in conflict:
                new_selected = [z for z in selected if z != y] + [x]
                if is_valid(new_selected):
                    selected = new_selected
                    improved = True
                    break
            
            if improved:
                break
                
    return selected
# EVOLVE-BLOCK-END
