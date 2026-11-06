import random
import time

def construct():
    deadline = time.time() + 50  # Safety margin
    
    def greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            if 200 - x in reachable:
                continue
            selected.append(x)
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
        return selected

    def is_valid(subset):
        """Check if subset has no sub-sum equal to 200"""
        reachable = {0}
        for x in subset:
            if 200 - x in reachable:
                return False
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
        return True

    def local_search(initial_subset, max_iter=5000):
        """Try to improve cardinality by adding/removing elements"""
        current = set(initial_subset)
        best = set(current)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add a random element not in current
            candidates_to_add = [x for x in range(1, 101) if x not in current]
            if candidates_to_add:
                x = random.choice(candidates_to_add)
                trial = current | {x}
                if is_valid(trial):
                    current = trial
                    if len(current) > len(best):
                        best = set(current)
            
            # Try to remove a random element from current (to allow adding others later)
            if current:
                x = random.choice(list(current))
                trial = current - {x}
                # Only accept removal if it doesn't decrease size (we're maximizing)
                # But we might want to remove to allow adding more later
                # For now, just accept removals to explore the space
                current = trial
                if len(current) > len(best):
                    best = set(current)
        
        return list(best)

    best = []
    
    # 1. Deterministic Largest First
    order1 = list(range(100, 0, -1))
    res1 = greedy(order1)
    if len(res1) > len(best):
        best = res1
        
    # 2. Deterministic Smallest First
    order2 = list(range(1, 101))
    res2 = greedy(order2)
    if len(res2) > len(best):
        best = res2
        
    # 3. Randomized Greedy
    random.seed(42)
    
    for i in range(2000):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        res = greedy(order)
        if len(res) > len(best):
            best = res
            
    # 4. Local search from best greedy solution
    if best:
        improved = local_search(best, max_iter=10000)
        if len(improved) > len(best):
            best = improved
    
    return best
