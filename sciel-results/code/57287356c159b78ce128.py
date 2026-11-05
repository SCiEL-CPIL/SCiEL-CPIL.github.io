"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    
    from random import Random
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True
    
    def can_add(S_set, x):
        """Check if x can be added to S_set while keeping it sum-free."""
        if x in S_set:
            return False
        # Check: for all a in S_set, a + x not in S_set and x + a not in S_set
        # and 2*x not in S_set
        for a in S_set:
            if a + x in S_set or x + a in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True
    
    def greedy_build(elements, time_budget=5.0):
        """Greedy sum-free subset construction with multiple strategies."""
        start = time.time()
        best = set()
        
        # Strategy 1: All odd numbers
        odds = [x for x in elements if x % 2 != 0]
        if odds:
            S = set(odds)
            if is_sum_free(S):
                if len(S) > len(best):
                    best = set(S)
        
        # Strategy 2: All even numbers divided by 2, then check
        evens = [x for x in elements if x % 2 == 0]
        if evens:
            S = set(evens)
            if is_sum_free(S):
                if len(S) > len(best):
                    best = set(S)
        
        # Strategy 3: Numbers with large absolute value (top half by |x|)
        sorted_by_abs = sorted(elements, key=lambda x: -abs(x))
        S = set(sorted_by_abs[:len(elements)//2])
        if is_sum_free(S):
            if len(S) > len(best):
                best = set(S)
        
        # Strategy 4: Greedy by adding elements that don't create conflicts
        # Try different orderings
        orderings = [
            sorted(elements),
            sorted(elements, reverse=True),
            sorted(elements, key=lambda x: abs(x)),
            sorted(elements, key=lambda x: -abs(x)),
            sorted(elements, key=lambda x: x % 7),  # modular grouping
        ]
        
        for ordering in orderings:
            if time.time() - start > time_budget:
                break
            S = set()
            for x in ordering:
                if can_add(S, x):
                    S.add(x)
            if len(S) > len(best):
                best = set(S)
        
        # Strategy 5: Randomized greedy multiple times
        rng = Random(42)
        for trial in range(20):
            if time.time() - start > time_budget:
                break
            shuffled = list(elements)
            rng.shuffle(shuffled)
            S = set()
            for x in shuffled:
                if can_add(S, x):
                    S.add(x)
            if len(S) > len(best):
                best = set(S)
        
        return best
    
    def local_search_improve(S, elements, max_iters=200):
        """Try to improve sum-free subset via local search (add/remove/swap)."""
        S = set(S)
        elements_set = set(elements)
        candidates = [x for x in elements if x not in S]
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try adding one element
            for x in candidates:
                if can_add(S, x):
                    S.add(x)
                    candidates.remove(x)
                    improved = True
                    break
            
            # Try swapping: remove one, add one or two
            if not improved:
                best_swap = None
                best_gain = 0
                for x in list(S):
                    S_without = S - {x}
                    for y in candidates:
                        if y in S:
                            continue
                        if can_add(S_without, y):
                            new_S = S_without | {y}
                            # Check if we can add more
                            extra = 0
                            for z in candidates:
                                if z != y and can_add(new_S, z):
                                    extra += 1
                                    if extra >= 1:
                                        break
                            gain = 1 + extra - 1  # net gain
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = (x, y)
                
                if best_swap and best_gain > 0:
                    x, y = best_swap
                    S.discard(x)
                    S.add(y)
                    if x in candidates:
                        candidates.append(x)
                    if y in candidates:
                        candidates.remove(y)
                    improved = True
        
        return S
    
    def solve_instance(elements, time_budget=30.0):
        """Solve one instance with time budget."""
        start = time.time()
        
        # Phase 1: Greedy construction
        S = greedy_build(elements, time_budget=min(10.0, time_budget * 0.3))
        
        # Phase 2: Local search improvement
        remaining = time_budget - (time.time() - start)
        if remaining > 2:
            max_iters = int(remaining * 50)  # scale iterations with time
            S = local_search_improve(S, elements, max_iters=max(10, max_iters))
        
        # Phase 3: Try to add more elements greedily after local search
        remaining = time_budget - (time.time() - start)
        if remaining > 1:
            elements_set = set(elements)
            candidates = [x for x in elements if x not in S]
            for x in candidates:
                if time.time() - start > time_budget - 0.5:
                    break
                if can_add(S, x):
                    S.add(x)
        
        return S
    
    # Solve each instance
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        if time.time() > deadline - 5:
            # Not enough time, return best effort (empty or odd numbers)
            elems = instance(size, seed)
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        elems = instance(size, seed)
        
        # Allocate time: more time for larger instances
        if size <= 60:
            budget = 20.0
        elif size <= 120:
            budget = 35.0
        else:
            budget = 50.0
        
        S = solve_instance(elems, time_budget=budget)
        results.append(sorted(S))
    
    return results
# EVOLVE-BLOCK-END
