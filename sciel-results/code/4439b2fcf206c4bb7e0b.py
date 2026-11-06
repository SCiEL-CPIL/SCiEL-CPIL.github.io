"""Sum-free subset finder using greedy construction with multiple restarts."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def greedy_build(elements, order_seed=None):
        """Greedy build: add elements in given order if they don't create conflicts."""
        if order_seed is not None:
            rng = random.Random(order_seed)
            elements = list(elements)
            rng.shuffle(elements)
        
        S = []
        S_set = set()
        for x in elements:
            can_add = True
            for a in S:
                if a + x in S_set:
                    can_add = False
                    break
                if x + x == a:
                    can_add = False
                    break
            if can_add:
                S.append(x)
                S_set.add(x)
        return S
    
    def try_add_back(S, all_elements):
        """Try to add back elements that don't create conflicts."""
        S = list(S)
        S_set = set(S)
        candidates = [x for x in all_elements if x not in S_set]
        
        improved = True
        while improved:
            improved = False
            for x in candidates[:]:
                can_add = True
                for a in S:
                    if a + x in S_set:
                        can_add = False
                        break
                    if x + x == a:
                        can_add = False
                        break
                if can_add:
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    improved = True
        return S
    
    def solve_instance(elements, time_budget):
        """Find a large sum-free subset of elements."""
        elements = list(elements)
        best_size = 0
        best_set = None
        
        # Strategy 1: Odd numbers (baseline, always sum-free)
        odds = [x for x in elements if x % 2 != 0]
        best_size = len(odds)
        best_set = odds[:]
        
        # Strategy 2: Greedy build with multiple random orderings
        for trial in range(20):
            if time.time() >= time_budget:
                break
            S = greedy_build(elements, order_seed=trial * 7919 + 12345)
            S = try_add_back(S, elements)
            if is_sum_free(S) and len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        # Strategy 3: Start from odds, greedily add evens
        if time.time() < time_budget:
            S = best_set[:]
            S_set = set(S)
            evens = [x for x in elements if x % 2 == 0 and x not in S_set]
            for x in evens:
                if time.time() >= time_budget:
                    break
                can_add = True
                for a in S:
                    if a + x in S_set:
                        can_add = False
                        break
                    if x + x == a:
                        can_add = False
                        break
                if can_add:
                    S.append(x)
                    S_set.add(x)
            if is_sum_free(S) and len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        return best_set
    
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        elements = instance(size, seed)
        time_budget = min(time.time() + 40, deadline - 10)
        result = solve_instance(elements, time_budget)
        results.append(result)
    
    return results
