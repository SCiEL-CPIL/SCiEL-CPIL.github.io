"""Improved sum-free subset finder using greedy + local search."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
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
                if (a + b) in S_set:
                    return False
        return True
    
    def count_violations(S):
        """Count number of (a,b,c) with a+b=c in S."""
        S_set = set(S)
        count = 0
        for a in S:
            for b in S:
                if (a + b) in S_set:
                    count += 1
        return count
    
    def greedy_add(S, candidates):
        """Greedily add elements from candidates to S maintaining sum-free."""
        S_set = set(S)
        for x in candidates:
            # Check if adding x creates any violation
            ok = True
            for a in S_set:
                if (x - a) in S_set:
                    ok = False
                    break
                if (a - x) in S_set:
                    ok = False
                    break
            if ok:
                S_set.add(x)
        return list(S_set)
    
    def local_search(S, all_elements, max_iters=5000):
        """Local search: try swaps to increase size."""
        S_set = set(S)
        remaining = [x for x in all_elements if x not in S_set]
        best_size = len(S_set)
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            improved = False
            # Try to add a random element from remaining
            if remaining:
                idx = random.randint(0, len(remaining) - 1)
                x = remaining[idx]
                # Check if x can be added
                can_add = True
                for a in S_set:
                    if (x - a) in S_set or (a - x) in S_set:
                        can_add = False
                        break
                if can_add:
                    S_set.add(x)
                    remaining.pop(idx)
                    if len(S_set) > best_size:
                        best_size = len(S_set)
                    improved = True
            
            # Try swaps: remove one from S, add one from remaining
            if not improved and remaining and S_set:
                s_list = list(S_set)
                remove_idx = random.randint(0, len(s_list) - 1)
                to_remove = s_list[remove_idx]
                
                for add_idx in range(min(50, len(remaining))):
                    x = remaining[add_idx]
                    # Check if S - {to_remove} + {x} is sum-free
                    new_set = S_set - {to_remove}
                    new_set.add(x)
                    if is_sum_free(list(new_set)):
                        S_set = new_set
                        remaining.pop(add_idx)
                        remaining.append(to_remove)
                        if len(S_set) > best_size:
                            best_size = len(S_set)
                        improved = True
                        break
            
            if not improved:
                # Shuffle remaining and try again
                random.shuffle(remaining)
        
        return list(S_set)
    
    def solve_instance(elements):
        """Find large sum-free subset of elements."""
        n = len(elements)
        if n == 0:
            return []
        
        # Strategy 1: Odds
        odds = [x for x in elements if x % 2 != 0]
        
        # Strategy 2: Upper half (large magnitude)
        max_abs = max(abs(x) for x in elements)
        upper = [x for x in elements if abs(x) > max_abs / 2]
        
        # Strategy 3: Greedy from scratch with random order
        best_solution = []
        
        # Try multiple random orderings
        for trial in range(30):
            if time.time() > DEADLINE:
                break
            
            shuffled = elements[:]
            random.shuffle(shuffled)
            
            # Greedy construction
            S = []
            S_set = set()
            for x in shuffled:
                ok = True
                for a in S_set:
                    if (x - a) in S_set or (a - x) in S_set:
                        ok = False
                        break
                if ok:
                    S.append(x)
                    S_set.add(x)
            
            # Local search to improve
            S = local_search(S, elements, max_iters=2000)
            
            if len(S) > len(best_solution):
                best_solution = S
        
        # Compare with odds and upper
        if len(odds) > len(best_solution):
            best_solution = odds
        if len(upper) > len(best_solution):
            best_solution = upper
        
        return best_solution
    
    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            # Fallback to odds if out of time
            elements = instance(size, seed)
            results.append([x for x in elements if x % 2 != 0])
            continue
        
        elements = instance(size, seed)
        solution = solve_instance(elements)
        results.append(solution)
    
    return results
# EVOLVE-BLOCK-END
