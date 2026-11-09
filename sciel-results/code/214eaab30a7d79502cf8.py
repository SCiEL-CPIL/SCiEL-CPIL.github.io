import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe under 90s
    
    n = 192
    
    def is_valid_addition(candidate, selected_set, selected_list):
        """Check if candidate can be added to selected_list without violating constraints."""
        # Check: candidate must not divide sum of any 3 distinct OTHER elements
        others = [v for v in selected_list if v != candidate]
        for combo in combinations(others, 3):
            if sum(combo) % candidate == 0:
                return False
        # Check: each existing element must not divide sum of any 3 distinct OTHER elements (including candidate)
        for a in selected_list:
            # 3 distinct others from (selected_list + [candidate]) excluding a
            # This is expensive, so let's be smart:
            # The new sums involving candidate: for each pair (b,c) in selected_list where b,c != a,
            # check if (b + c + candidate) % a == 0
            # Also existing sums were already valid, so only need to check sums that include candidate
            for b in selected_list:
                if b == a:
                    continue
                for c in selected_list:
                    if c == a or c == b:
                        continue
                    if (b + c + candidate) % a == 0:
                        return False
        return True
    
    def full_validate(selected_list):
        """Full validation of the entire set."""
        for a in selected_list:
            others = [v for v in selected_list if v != a]
            for combo in combinations(others, 3):
                if sum(combo) % a == 0:
                    return False
        return True
    
    # Phase 1: Greedy from largest to smallest
    selected = []
    selected_set = set()
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    
    best = list(selected)
    
    # Phase 2: Local search - try to add excluded elements
    excluded = [x for x in range(1, n+1) if x not in selected_set]
    
    # Try adding each excluded element (just adding, not swapping)
    for x in excluded:
        if time.time() > deadline:
            break
        if is_valid_addition(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    
    if len(selected) > len(best):
        best = list(selected)
    
    # Phase 3: Try swaps - remove one element, try to add two excluded elements
    # This is more complex, let's try a simpler approach first
    
    # Phase 4: Try different greedy orders
    # Try greedy from a different starting point
    import random
    random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        # Try a perturbed order
        order = list(range(n, 0, -1))
        # Perturb: move some elements
        for _ in range(10):
            i = random.randint(0, len(order)-1)
            j = random.randint(0, len(order)-1)
            order[i], order[j] = order[j], order[i]
        
        sel = []
        sel_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(x, sel_set, sel):
                sel.append(x)
                sel_set.add(x)
        
        if len(sel) > len(best):
            best = list(sel)
    
    # Phase 5: Local search on best - try to add more by removing conflicting elements
    best_set = set(best)
    for _ in range(3):
        if time.time() > deadline:
            break
        improved = False
        for x in sorted(range(1, n+1), reverse=True):
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            # Try adding x
            if is_valid_addition(x, best_set, best):
                best.append(x)
                best_set.add(x)
                improved = True
                break
        if not improved:
            break
    
    return best
