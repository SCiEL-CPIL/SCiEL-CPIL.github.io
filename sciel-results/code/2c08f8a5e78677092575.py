"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
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
    
    def count_violations(S):
        """Count number of violating triples."""
        S_set = set(S)
        count = 0
        for a in S:
            for b in S:
                if a + b in S_set:
                    count += 1
        return count
    
    def can_add(S_set, x):
        """Check if x can be added to S without creating violations."""
        for a in S_set:
            if a + x in S_set or x + a in S_set or 2*x in S_set:
                return False
        return True
    
    def greedy_build(values, order):
        """Build sum-free set greedily in given order."""
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    def local_search(S, values, rng, max_iters=200):
        """Improve sum-free set via local search."""
        S = set(S)
        S_set = S
        values_set = set(values)
        best_size = len(S)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try adding elements
            candidates = [x for x in values_set if x not in S]
            rng.shuffle(candidates)
            for x in candidates[:50]:
                if can_add(S_set, x):
                    S_set.add(x)
                    best_size = max(best_size, len(S_set))
                    improved = True
                    break
            
            # Try removing elements to allow more additions
            if not improved:
                S_list = list(S_set)
                rng.shuffle(S_list)
                for x in S_list[:20]:
                    S_set.remove(x)
                    added = 0
                    for y in values_set:
                        if y not in S_set and can_add(S_set, y):
                            S_set.add(y)
                            added += 1
                    if added > 1:
                        best_size = max(best_size, len(S_set))
                        improved = True
                    else:
                        S_set.add(x)
                    if improved:
                        break
            
            if not improved:
                break
        
        return S_set
    
    def solve_instance(values, seed):
        """Find large sum-free subset for one instance."""
        if time.time() > deadline - 10:
            # Fallback: odd numbers
            return [x for x in values if x % 2]
        
        best = set()
        best_size = 0
        rng = Random(seed)
        
        n = len(values)
        
        # Strategy 1: Greedy with different orderings
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            order = values[:]
            rng.shuffle(order)
            S = greedy_build(values, order)
            if len(S) > best_size:
                best = S
                best_size = len(S)
        
        # Strategy 2: Local search from best
        if best_size > 0:
            S = local_search(best, values, rng, max_iters=100)
            if len(S) > best_size:
                best = S
                best_size = len(S)
        
        # Strategy 3: Start from odd numbers, then local search
        odd_set = set(x for x in values if x % 2)
        S = local_search(odd_set, values, rng, max_iters=100)
        if len(S) > best_size:
            best = S
            best_size = len(S)
        
        # Strategy 4: More aggressive local search with restarts
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            S = set()
            rng.shuffle(values)
            for x in values:
                if can_add(S, x):
                    S.add(x)
            
            S = local_search(S, values, rng, max_iters=150)
            if len(S) > best_size:
                best = S
                best_size = len(S)
        
        return list(best)
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        values = instance(size, seed)
        subset = solve_instance(values, seed)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
