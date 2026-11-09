# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses randomized greedy construction with multiple restarts and local search refinement.
    """
    m = 205
    deadline = time.time() + 590  # 10s safety margin before 600s kill
    
    # Precompute forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    
    # Precompute conflict matrix: conflict[i][j] = True if |i-j| mod m is in forbidden
    # Actually, we need: (i-j) mod m in forbidden OR (j-i) mod m in forbidden
    # Since forbidden is symmetric (if d is a QR, then -d is also a QR mod 205? Not necessarily)
    # Let's just check both directions
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i, j] = True
    
    best_set = set()
    best_size = 0
    
    def is_valid(s):
        """Check if set s is valid (no conflicts)."""
        arr = list(s)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if conflict[arr[i], arr[j]]:
                    return False
        return True
    
    def greedy_construct(order):
        """Build a set using greedy method with given order of candidates."""
        s = set()
        for candidate in order:
            ok = True
            for existing in s:
                if conflict[candidate, existing]:
                    ok = False
                    break
            if ok:
                s.add(candidate)
        return s
    
    def local_search(s, max_iters=1000):
        """Try to improve set s via local search: add/remove/swap operations."""
        s = set(s)
        current_size = len(s)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a new element
            candidates_to_add = [x for x in range(m) if x not in s]
            random.shuffle(candidates_to_add)
            for x in candidates_to_add:
                ok = True
                for y in s:
                    if conflict[x, y]:
                        ok = False
                        break
                if ok:
                    s.add(x)
                    current_size += 1
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one element, add another
                s_list = list(s)
                random.shuffle(s_list)
                for y in s_list:
                    s_without_y = s - {y}
                    candidates_to_add = [x for x in range(m) if x not in s_without_y]
                    random.shuffle(candidates_to_add)
                    for x in candidates_to_add:
                        if x == y:
                            continue
                        ok = True
                        for z in s_without_y:
                            if conflict[x, z]:
                                ok = False
                                break
                        if ok:
                            s = s_without_y
                            s.add(x)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try remove one and add two (net gain of 1)
                s_list = list(s)
                random.shuffle(s_list)
                for y in s_list:
                    s_without_y = s - {y}
                    # Find two elements that can be added
                    candidates = [x for x in range(m) if x not in s_without_y and x != y]
                    random.shuffle(candidates)
                    found_pair = False
                    for i, x1 in enumerate(candidates):
                        if found_pair:
                            break
                        ok1 = True
                        for z in s_without_y:
                            if conflict[x1, z]:
                                ok1 = False
                                break
                        if not ok1:
                            continue
                        for x2 in candidates[i+1:]:
                            if found_pair:
                                break
                            ok2 = True
                            for z in s_without_y:
                                if conflict[x2, z]:
                                    ok2 = False
                                    break
                            if not ok2:
                                continue
                            if conflict[x1, x2]:
                                continue
                            # Both can be added
                            s = s_without_y
                            s.add(x1)
                            s.add(x2)
                            found_pair = True
                            break
                    if found_pair:
                        improved = True
                        break
            
            if not improved:
                break
        
        return s
    
    # Run multiple randomized greedy constructions
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 12345)
        
        # Create a random order
        order = list(range(m))
        random.shuffle(order)
        
        # Greedy construction
        s = greedy_construct(order)
        
        # Local search refinement
        s = local_search(s)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # Additional local search passes on the best set
    for _ in range(5):
        if time.time() > deadline:
            break
        s = local_search(best_set)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
