"""Improved sum-free subset finder using multi-start greedy with different orderings."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def greedy_add_order(elements, order, max_time=30.0):
        """Greedy construction following a specific order."""
        S = []
        S_set = set()
        
        for x in order:
            if time.time() > max_time:
                break
            conflict = False
            for a in S:
                # Case 1: a + b = x => b = x - a
                if (x - a) in S_set:
                    conflict = True
                    break
                # Case 2: a + x = c => c = a + x
                if (a + x) in S_set:
                    conflict = True
                    break
                # Case 3: x + x = c => c = 2x
                if (2 * x) in S_set:
                    conflict = True
                    break
            
            if not conflict:
                S.append(x)
                S_set.add(x)
        
        # Local search: try to add remaining elements
        remaining = [x for x in elements if x not in S_set]
        rng_local = Random(12345)
        rng_local.shuffle(remaining)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i, x in enumerate(remaining):
                if time.time() > max_time:
                    break
                conflict = False
                for a in S:
                    if (x - a) in S_set or (a + x) in S_set:
                        conflict = True
                        break
                if (2 * x) in S_set:
                    conflict = True
                    break
                
                if not conflict:
                    S.append(x)
                    S_set.add(x)
                    remaining.pop(i)
                    improved = True
        
        return S
    
    def swap_improve(S, elements, max_time=10.0):
        """Try 1-for-2 swaps: remove one element, add two others if possible."""
        S_set = set(S)
        remaining = [x for x in elements if x not in S_set]
        
        if not remaining:
            return S
        
        # Shuffle for variety
        rng = Random(777)
        rng.shuffle(remaining)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i, x in enumerate(remaining):
                if time.time() > max_time:
                    break
                
                # Try removing each element from S and see if we can add x and another
                # This is expensive, so limit the number of removals to try
                for j in range(min(len(S), 50)):  # Limit to first 50 elements
                    if time.time() > max_time:
                        break
                    
                    # Try removing S[j]
                    y = S[j]
                    
                    # Check if x can be added without y
                    conflict_x = False
                    for a in S:
                        if a == y:
                            continue
                        if (x - a) in S_set or (a + x) in S_set:
                            if a == y or (x - a) == y or (a + x) == y:
                                continue
                            if (x - a) in S_set or (a + x) in S_set:
                                conflict_x = True
                                break
                    if (2 * x) in S_set:
                        if 2 * x != y:
                            conflict_x = True
                    
                    if conflict_x:
                        continue
                    
                    # Now try to add another element z from remaining (or x itself if possible)
                    # For simplicity, just check if x fits
                    # If it does, we've improved by 0 (removed 1, added 1)
                    # We want to add 2 to net gain 1
                    # This is complex, so let's just do 1-for-1 swaps first
                    
                    # Actually, let's do a simpler improvement: 
                    # Try to swap S[j] with x if x fits and S[j] doesn't
                    # But we already know x fits, so this would be neutral or worse
                    # unless removing S[j] allows adding more later
                    
                    # For now, skip complex swaps and rely on multi-start greedy
                    pass
        
        return S
    
    def multi_start_greedy(elements, max_time=30.0):
        """Try multiple greedy orderings and return the best result."""
        n = len(elements)
        if n <= 3:
            return list(elements)
        
        best_result = []
        best_size = 0
        
        # Define different orderings
        orderings = []
        
        # 1. Sort by absolute value descending
        order1 = sorted(elements, key=lambda x: abs(x), reverse=True)
        orderings.append(order1)
        
        # 2. Sort by value descending
        order2 = sorted(elements, reverse=True)
        orderings.append(order2)
        
        # 3. Sort by value ascending
        order3 = sorted(elements)
        orderings.append(order3)
        
        # 4. Sort by absolute value ascending
        order4 = sorted(elements, key=lambda x: abs(x))
        orderings.append(order4)
        
        # 5. Random orderings
        rng = Random(42)
        for _ in range(3):
            order_rand = elements.copy()
            rng.shuffle(order_rand)
            orderings.append(order_rand)
        
        # Time per ordering
        time_per_order = max(1.0, max_time / len(orderings))
        
        for order in orderings:
            if time.time() > max_time:
                break
            
            result = greedy_add_order(elements, order, max_time)
            
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
        
        # Final local search on the best result
        if time.time() < max_time:
            remaining_time = max_time - time.time()
            # Try to improve with additional local search
            best_set = set(best_result)
            remaining = [x for x in elements if x not in best_set]
            rng_local = Random(999)
            rng_local.shuffle(remaining)
            
            improved = True
            while improved and time.time() < max_time:
                improved = False
                for i, x in enumerate(remaining):
                    if time.time() > max_time:
                        break
                    conflict = False
                    for a in best_result:
                        if (x - a) in best_set or (a + x) in best_set:
                            conflict = True
                            break
                    if (2 * x) in best_set:
                        conflict = True
                        break
                    
                    if not conflict:
                        best_result.append(x)
                        best_set.add(x)
                        remaining.pop(i)
                        improved = True
        
        return best_result
    
    def solve_instance(elements, max_time=30.0):
        """Solve a single instance using multi-start greedy."""
        return multi_start_greedy(elements, max_time)
    
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        if time.time() > deadline:
            results.append([])
            continue
        
        elems = instance(size, seed)
        
        # Allocate time based on size
        # Larger instances get more time
        if size <= 60:
            time_budget = 30.0
        elif size <= 120:
            time_budget = 50.0
        else:
            time_budget = 60.0
        
        remaining_time = deadline - time.time()
        if time_budget > remaining_time:
            time_budget = max(1.0, remaining_time - 1.0)
        
        subset = solve_instance(elems, time_budget)
        
        # Verify sum-free property
        if not is_sum_free(subset):
            # Fallback to a simple greedy
            subset = greedy_add_order(elems, sorted(elems, key=lambda x: abs(x), reverse=True), 5.0)
        
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
