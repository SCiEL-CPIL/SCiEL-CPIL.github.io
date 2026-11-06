"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 512
    deadline = time.time() + 55.0  # Safety margin for 90s limit
    
    # Precompute conflict lists: for each x, list of (d, position) such that
    # if x is at position i in AP (a, a+d, a+2d, a+3d), we need the other 3 terms.
    # conflicts[x] = list of tuples (y1, y2, y3) where {x, y1, y2, y3} forms a 4-AP
    # If all of y1, y2, y3 are in selected, then x cannot be added.
    # Actually, we need: adding x creates a 4-AP iff there exist d and position i
    # such that the other three terms of the AP are in selected.
    
    # For efficiency, precompute for each x the list of "forbidden triples"
    # forbidden[x] = list of (a, b, c) such that {x, a, b, c} is a 4-AP
    # Adding x is forbidden iff any (a,b,c) with a,b,c in selected exists.
    
    forbidden = [[] for _ in range(n + 1)]
    for a in range(1, n - 2):
        for d in range(1, (n - a) // 3 + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2*d
            x4 = a + 3*d
            if x4 > n:
                break
            # x1 is first: forbidden triple is (x2, x3, x4)
            forbidden[x1].append((x2, x3, x4))
            # x2 is second: forbidden triple is (x1, x3, x4)
            forbidden[x2].append((x1, x3, x4))
            # x3 is third: forbidden triple is (x1, x2, x4)
            forbidden[x3].append((x1, x2, x4))
            # x4 is fourth: forbidden triple is (x1, x2, x3)
            forbidden[x4].append((x1, x2, x3))
    
    def can_add(x, selected_set):
        """Check if x can be added to selected_set without forming a 4-AP."""
        for (a, b, c) in forbidden[x]:
            if a in selected_set and b in selected_set and c in selected_set:
                return False
        return True
    
    def greedy_construct(order):
        """Greedy construction given an ordering of candidates."""
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    def local_search(selected, max_iters=100):
        """Try to improve selected by adding/removing/swapping elements."""
        current = set(selected)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding any element that can be added
            for x in range(1, n + 1):
                if x not in current and can_add(x, current):
                    current.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try removing one element to allow adding two or more
                for x in list(current):
                    trial = current - {x}
                    added_count = 0
                    for y in range(1, n + 1):
                        if y not in trial and can_add(y, trial):
                            trial.add(y)
                            added_count += 1
                    if added_count > 1:
                        current = trial
                        improved = True
                        break
            
            if not improved:
                break
        
        return current
    
    # Phase 1: Multiple randomized greedy passes with local search
    best_set = set()
    best_size = 0
    
    # Deterministic baseline
    selected = greedy_construct(range(1, n + 1))
    selected = local_search(selected, max_iters=30)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
    
    # Randomized greedy with multiple shuffles
    candidates = list(range(1, n + 1))
    
    while time.time() < deadline - 8:  # Reserve 8s for final polish
        random.shuffle(candidates)
        selected = greedy_construct(candidates)
        selected = local_search(selected, max_iters=40)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    # Phase 2: More aggressive local search on best found
    if time.time() < deadline - 2:
        improved = local_search(best_set, max_iters=300)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
