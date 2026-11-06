import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S_set):
        """Check if x can be added to S_set while maintaining sum-free property."""
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True

    def can_swap_remove_add(u, v, S_set):
        """Check if we can remove u and add v to S_set while maintaining sum-free."""
        # First, remove u from consideration
        # Check if v conflicts with any remaining element
        for a in S_set:
            if a == u:
                continue
            if a + v in S_set and a + v != u:
                return False
            if v - a in S_set and v - a != u:
                return False
        if 2 * v in S_set and 2 * v != u:
            return False
        return True

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_add(candidates, current_set, rng):
        """Greedy add elements from candidates to current_set."""
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        for e in shuffled:
            if can_add(e, current_set):
                current_set.add(e)
        return current_set

    def local_search_2opt(current_set, S, rng, max_iters=50):
        """2-opt local search: try swapping one element for another."""
        S_set = set(S)
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            elements_to_try = list(current_set)
            rng.shuffle(elements_to_try)
            
            for u in elements_to_try:
                if time.time() > deadline - 10:
                    break
                    
                # Find candidates that could replace u
                candidates = [e for e in S if e not in current_set]
                rng.shuffle(candidates)
                
                for v in candidates:
                    if can_swap_remove_add(u, v, current_set):
                        new_set = set(current_set)
                        new_set.remove(u)
                        new_set.add(v)
                        current_set = new_set
                        improved = True
                        break
                        
                if improved:
                    break
                    
            if not improved:
                break
                
        return current_set

    def solve_instance(S):
        S_set = set(S)
        odds = [x for x in S if x % 2 != 0]
        evens = [x for x in S if x % 2 == 0]
        
        best_set = set(odds)
        best_size = len(best_set)
        
        num_restarts = 25
        rng = Random(42)
        
        for r in range(num_restarts):
            if time.time() > deadline - 15:
                break
                
            current_set = set(odds)
            
            # Greedy add evens
            shuffled_evens = evens[:]
            rng.shuffle(shuffled_evens)
            
            for e in shuffled_evens:
                if time.time() > deadline - 15:
                    break
                if can_add(e, current_set):
                    current_set.add(e)
            
            # 2-opt local search
            current_set = local_search_2opt(current_set, S, rng, max_iters=30)
            
            if len(current_set) > best_size:
                best_set = current_set
                best_size = len(best_set)
                
        return list(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Fallback if time is tight
        if time.time() > deadline - 10:
             results.append([x for x in S if x % 2])
             continue
             
        res = solve_instance(S)
        
        # Verify
        if not is_sum_free(res):
            # Fallback to parity
            res = [x for x in S if x % 2]
            
        results.append(res)
        
    return results
