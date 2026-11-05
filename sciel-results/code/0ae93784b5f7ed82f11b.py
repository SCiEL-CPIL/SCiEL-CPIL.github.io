"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter, defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if S is sum-free."""
        S_set = set(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                if a + b in S_set:
                    return False
        return True

    def find_violations(S_list):
        """Return list of (a, b, c) such that a+b=c, all in S."""
        S_set = set(S_list)
        viol = []
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                c = a + b
                if c in S_set:
                    viol.append((a, b, c))
        return viol

    def can_add(S_set, S_list, x):
        """Check if x can be added to sum-free set S without violation."""
        # x + a in S for any a in S_list
        # x - a in S for any a in S_list (i.e., a + x in S)
        # 2*x in S
        for a in S_list:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True

    def repair_set(S, V, time_limit=5.0):
        """Remove elements to make S sum-free, then greedily expand."""
        start_repair = time.time()
        S_list = list(S)
        S_set = set(S_list)
        V_set = set(V)
        
        # Phase 1: Remove elements to fix violations
        while True:
            if time.time() - start_repair > time_limit:
                break
            viol = find_violations(S_list)
            if not viol:
                break
            
            counts = Counter()
            for a, b, c in viol:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
            
            if counts:
                worst = max(counts, key=counts.get)
                if worst in S_set:
                    S_list.remove(worst)
                    S_set.discard(worst)
                else:
                    break
            else:
                break

        # Phase 2: Greedily add safe elements
        improved = True
        while improved and time.time() - start_repair < time_limit:
            improved = False
            candidates = [x for x in V if x not in S_set]
            # Sort candidates by some heuristic - maybe by how many conflicts they'd have
            for x in candidates:
                if can_add(S_set, S_list, x):
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                    break
        return S_list

    def local_search_improve(S, V, max_iter=50, time_budget=3.0):
        """Try to improve S by adding elements and doing 1-opt swaps."""
        start_ls = time.time()
        S_list = list(S)
        S_set = set(S_list)
        V_set = set(V)
        
        for _ in range(max_iter):
            if time.time() - start_ls > time_budget:
                break
            
            # Try adding
            added = False
            candidates = [x for x in V if x not in S_set]
            for x in candidates:
                if can_add(S_set, S_list, x):
                    S_list.append(x)
                    S_set.add(x)
                    added = True
                    break
            
            if added:
                continue
            
            # Try 1-opt swap: remove one element, try to add more
            best_swap = None
            best_gain = 0
            
            for i, elem in enumerate(S_list):
                if time.time() - start_ls > time_budget:
                    break
                    
                temp_set = set(S_list)
                temp_set.discard(elem)
                temp_list = [x for x in S_list if x != elem]
                
                # Try adding candidates
                added_count = 0
                for x in V:
                    if x not in temp_set:
                        if can_add(temp_set, temp_list, x):
                            temp_set.add(x)
                            temp_list.append(x)
                            added_count += 1
                
                gain = added_count - 1
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (i, added_count)
            
            if best_swap and best_gain > 0:
                i, _ = best_swap
                removed_elem = S_list.pop(i)
                S_set.discard(removed_elem)
                
                # Re-add elements that were addable
                for x in V:
                    if x not in S_set:
                        if can_add(S_set, S_list, x):
                            S_list.append(x)
                            S_set.add(x)
            else:
                break
        
        return S_list

    def search_instance(elements, seed_offset=0, time_budget=40.0):
        """Search for a large sum-free subset of elements."""
        start_search = time.time()
        rng = random.Random(seed_offset)
        best = []
        best_size = 0
        n = len(elements)
        
        # Generate multiple initial candidates
        candidates = []
        
        # 1. Odds
        odd = [x for x in elements if x % 2 != 0]
        candidates.append(odd)
        
        # 2. Largest values
        sorted_desc = sorted(elements, reverse=True)
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            k = int(n * frac)
            candidates.append(sorted_desc[:k])
        
        # 3. Smallest absolute values
        sorted_abs = sorted(elements, key=lambda x: abs(x))
        for frac in [0.3, 0.4, 0.5, 0.6]:
            k = int(n * frac)
            candidates.append(sorted_abs[:k])
        
        # 4. Random samples
        for _ in range(12):
            k = int(n * rng.uniform(0.3, 0.8))
            candidates.append(rng.sample(elements, k))
            
        # 5. Positive numbers
        pos = [x for x in elements if x > 0]
        candidates.append(pos)
        
        # 6. Negative numbers
        neg = [x for x in elements if x < 0]
        candidates.append(neg)
        
        # 7. High values (top 1/3)
        high = sorted(elements, reverse=True)[:n//3]
        candidates.append(high)
        
        # 8. Low values (bottom 1/3)
        low = sorted(elements)[:n//3]
        candidates.append(low)
        
        # 9. Values above median
        median_val = elements[n//2]
        above_median = [x for x in elements if x > median_val]
        candidates.append(above_median)
        
        # 10. Values below median
        below_median = [x for x in elements if x < median_val]
        candidates.append(below_median)
        
        # Process each candidate
        for cand in candidates:
            if time.time() - start_search > time_budget:
                break
            S = repair_set(cand, elements, time_limit=2.0)
            if len(S) > best_size:
                best_size = len(S)
                best = S
        
        # Additional random restarts if time permits
        restart_budget = min(10.0, time_budget - (time.time() - start_search))
        if restart_budget > 2.0:
            for _ in range(30):
                if time.time() - start_search > time_budget - 2.0:
                    break
                k = int(n * rng.uniform(0.3, 0.8))
                cand = rng.sample(elements, k)
                S = repair_set(cand, elements, time_limit=1.0)
                if len(S) > best_size:
                    best_size = len(S)
                    best = S
        
        # Local search on best
        if time.time() - start_search < time_budget - 1.0:
            improved = local_search_improve(best, elements, max_iter=50, time_budget=3.0)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
        
        # Second round of local search with different strategy
        if time.time() - start_search < time_budget - 0.5:
            improved2 = local_search_improve(best, elements, max_iter=30, time_budget=2.0)
            if len(improved2) > best_size:
                best = improved2
                best_size = len(improved2)
        
        return best

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    total_time = 0
    for i, elems in enumerate(instances):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        # Allocate time based on instance size
        if i == 0:
            time_budget = min(25.0, remaining - 10)
        elif i == 1:
            time_budget = min(35.0, remaining - 10)
        else:
            time_budget = min(50.0, remaining)
        
        result = search_instance(elems, seed_offset=801 + i, time_budget=time_budget)
        results.append(result)
    
    # Final verification and fallback
    for i, result in enumerate(results):
        if not is_sum_free(result):
            # Try to fix by removing violating elements
            S_list = list(result)
            S_set = set(S_list)
            changed = True
            while changed:
                changed = False
                viol = find_violations(S_list)
                if viol:
                    counts = Counter()
                    for a, b, c in viol:
                        counts[a] += 1
                        counts[b] += 1
                        counts[c] += 1
                    worst = max(counts, key=counts.get)
                    if worst in S_set:
                        S_list.remove(worst)
                        S_set.discard(worst)
                        changed = True
            results[i] = S_list
    
    return results
# EVOLVE-BLOCK-END
