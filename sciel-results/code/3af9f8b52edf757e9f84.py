import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    
    n = 512
    k = 4  # 4-term AP
    
    def is_valid_4ap_free(selected_set):
        """Check if a set contains no nonconstant 4-term AP."""
        lst = sorted(selected_set)
        # For each pair (a, b), check if b+d and b+2d are in the set where d = b-a > 0
        # Actually, 4-term AP: a, a+d, a+2d, a+3d
        # For each a in set, and each d >= 1, check if a+d, a+2d, a+3d are in set
        for a in lst:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in selected_set and (a + 2*d) in selected_set and (a + 3*d) in selected_set:
                    return False
        return True
    
    def greedy_seed():
        """Fast greedy construction to get a valid starting set."""
        selected = set()
        for x in range(1, n + 1):
            # Check if adding x creates a 4-AP
            valid = True
            # x could be the 1st, 2nd, 3rd, or 4th term of a 4-AP
            # Case 1: x is 1st term: x, x+d, x+2d, x+3d -> check if x+d, x+2d, x+3d in selected
            for d in range(1, (n - x) // 3 + 1):
                if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # Case 2: x is 2nd term: a, x, x+d, x+2d where a = x-d
            for d in range(1, (x - 1) // 3 + 1):
                a = x - d
                if a in selected and (x + d) in selected and (x + 2*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # Case 3: x is 3rd term: a, a+d, x, x+d where a = x-2d
            for d in range(1, (x - 1) // 2 + 1):
                a = x - 2*d
                if a in selected and (a + d) in selected and (x + d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # Case 4: x is 4th term: a, a+d, a+2d, x where a = x-3d
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3*d
                if a in selected and (a + d) in selected and (a + 2*d) in selected:
                    valid = False
                    break
            if valid:
                selected.add(x)
        return selected
    
    def find_4aps_in_set(selected_set):
        """Return list of 4-APs (as tuples) present in the set."""
        aps = []
        lst = sorted(selected_set)
        for a in lst:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in selected_set and (a + 2*d) in selected_set and (a + 3*d) in selected_set:
                    aps.append((a, a+d, a+2*d, a+3*d))
        return aps
    
    # Start with greedy seed
    best_set = greedy_seed()
    best_size = len(best_set)
    
    # Local search: try to add elements
    unselected = set(range(1, n + 1)) - best_set
    random.seed(42)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        if not unselected:
            break
        
        # Try random additions
        improved = False
        for _ in range(min(100, len(unselected))):
            if time.time() > deadline - 5:
                break
            x = random.choice(list(unselected))
            test_set = best_set | {x}
            if is_valid_4ap_free(test_set):
                best_set = test_set
                unselected.remove(x)
                best_size += 1
                improved = True
                break
        
        # If we improved, continue
        if improved:
            continue
        
        # Try swaps: remove one element, add one or two elements
        # First, try single swaps: remove y, add x
        # But this doesn't increase size. We need to increase size.
        # So try: remove y, add x1 and x2 (net +1)
        
        # Find elements that are in many 4-APs (bad elements)
        # For now, try random swaps: remove y, add x1, x2
        if len(best_set) > 0 and len(unselected) >= 2:
            for _ in range(min(50, len(unselected))):
                if time.time() > deadline - 5:
                    break
                # Pick a random element to remove
                y = random.choice(list(best_set))
                # Pick two random elements to add
                candidates = list(unselected)
                random.shuffle(candidates)
                x1 = candidates[0]
                x2 = candidates[1]
                
                test_set = (best_set - {y}) | {x1, x2}
                if is_valid_4ap_free(test_set):
                    best_set = test_set
                    unselected = (unselected - {x1, x2}) | {y}
                    best_size += 1
                    improved = True
                    break
        
        if not improved:
            # Try more aggressive: remove up to 3, add up to 4
            # But keep it simple for now
            pass
    
    return sorted(best_set)
