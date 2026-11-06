"""Sum-free subset finder using Conflict-Graph Maximum Independent Set Heuristic with 2-Opt Local Search."""
import time
import random
import math
from collections import defaultdict


def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(elems):
        """Verify a list is sum-free."""
        s = set(elems)
        for a in elems:
            for b in elems:
                if a + b in s:
                    return False
        return True

    def find_sum_free(elements, time_budget=45.0):
        """Find a large sum-free subset using Conflict-Graph MIS Heuristic with 2-Opt."""
        if not elements:
            return []

        n = len(elements)
        elem_set = set(elements)
        elem_to_idx = {x: i for i, x in enumerate(elements)}
        my_deadline = time.time() + time_budget

        # Precompute: for each element k, which pairs (i,j) sum to k
        # And for each element k, which pairs (d,j) satisfy k+j=d
        sum_pairs = [[] for _ in range(n)]  # sum_pairs[k]: list of (i,j) where elements[i]+elements[j]==elements[k]
        add_conflicts = [[] for _ in range(n)]  # add_conflicts[k]: list of (d,j) where elements[k]+elements[j]==elements[d]

        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    sum_pairs[k].append((i, j))

        for i in range(n):
            for j in range(n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    add_conflicts[i].append((k, j))

        def can_add(frozen, k):
            """Check if element k can be added to the set represented by frozen (a set of indices)."""
            # Check 1: no two elements in frozen sum to k
            for (i, j) in sum_pairs[k]:
                if i in frozen and j in frozen:
                    return False
            # Check 2: adding k doesn't create a sum with existing elements
            for (d, j) in add_conflicts[k]:
                if d in frozen and j in frozen:
                    return False
            return True

        def count_conflicts(frozen):
            """Count violated sum-free constraints."""
            count = 0
            for k in frozen:
                for (i, j) in sum_pairs[k]:
                    if i in frozen and j in frozen:
                        count += 1
            return count

        def greedy_completion(frozen, order=None):
            """Greedy add any element that fits, until no more can be added."""
            if order is None:
                order = list(range(n))
            result = set(frozen)
            changed = True
            while changed:
                changed = False
                for k in order:
                    if k not in result and can_add(result, k):
                        result.add(k)
                        changed = True
            return result

        def local_search_2opt(frozen, max_iters=3000):
            """2-opt local search: try swapping one element out and one in."""
            current = set(frozen)
            best = set(current)
            best_size = len(current)
            rng = random.Random(42)
            
            # Precompute conflict neighborhoods for faster evaluation
            # For each element, what's the "cost" of removing it and "benefit" of adding others
            
            for iteration in range(max_iters):
                if time.time() > my_deadline:
                    break
                
                # Try to add any element that fits
                added = False
                inactive = [k for k in range(n) if k not in current]
                if inactive:
                    # Sort inactive by some heuristic (e.g., fewer conflicts with current set)
                    def conflict_count_with_current(k):
                        cnt = 0
                        for (i, j) in sum_pairs[k]:
                            if i in current and j in current:
                                cnt += 1
                        for (d, j) in add_conflicts[k]:
                            if d in current and j in current:
                                cnt += 1
                        return cnt
                    
                    inactive.sort(key=conflict_count_with_current)
                    for k in inactive:
                        if can_add(current, k):
                            current.add(k)
                            added = True
                            break
                
                if not added:
                    # Try 2-opt: remove one, add one
                    if len(current) > 1:
                        active_list = list(current)
                        # Try a few random swaps
                        for _ in range(min(5, len(active_list))):
                            rem = rng.choice(active_list)
                            base = current - {rem}
                            # Find best element to add
                            best_add = None
                            best_new_size = len(base)
                            for k in range(n):
                                if k in current or k == rem:
                                    continue
                                if can_add(base, k):
                                    new_set = base | {k}
                                    # Check if this improves or maintains
                                    if len(new_set) >= best_new_size:
                                        # Prefer adding if it doesn't create conflicts
                                        if count_conflicts(new_set) == 0:
                                            best_add = k
                                            break
                            
                            if best_add is not None:
                                current = base | {best_add}
                            else:
                                # Just remove
                                current = base
                            
                            if len(current) > best_size:
                                best_size = len(current)
                                best = set(current)
                
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
                
                # Check if stuck and need to restart
                if not added and len(current) == best_size:
                    # Try a more aggressive restart: remove 2 random elements
                    if len(current) > 4:
                        to_remove = rng.sample(list(current), 2)
                        for r in to_remove:
                            current.discard(r)
                        # Greedy complete
                        current = greedy_completion(current)
                        if len(current) > best_size:
                            best_size = len(current)
                            best = set(current)

            return best

        def multi_start_search():
            """Try multiple different initial solutions and local search from each."""
            best_result = set()
            best_size = 0
            
            # Strategy 1: Sign partition
            pos = set(i for i in range(n) if elements[i] > 0)
            neg = set(i for i in range(n) if elements[i] < 0)
            for s in [pos, neg]:
                completed = greedy_completion(s)
                if len(completed) > best_size:
                    best_size = len(completed)
                    best_result = completed
            
            # Strategy 2: Residue classes mod 3, 5, 7
            for mod in [3, 5, 7]:
                for r in range(mod):
                    s = set(i for i in range(n) if elements[i] % mod == r)
                    completed = greedy_completion(s)
                    if len(completed) > best_size:
                        best_size = len(completed)
                        best_result = completed
            
            # Strategy 3: Greedy by absolute value (largest first)
            order_abs = sorted(range(n), key=lambda i: -abs(elements[i]))
            completed = greedy_completion(set(), order_abs)
            if len(completed) > best_size:
                best_size = len(completed)
                best_result = completed
            
            # Strategy 4: Greedy by absolute value (smallest first)
            order_abs_small = sorted(range(n), key=lambda i: abs(elements[i]))
            completed = greedy_completion(set(), order_abs_small)
            if len(completed) > best_size:
                best_size = len(completed)
                best_result = completed
            
            # Strategy 5: Randomized greedy (multiple tries)
            for t in range(10):
                rng = random.Random(10000 + t)
                order = list(range(n))
                rng.shuffle(order)
                completed = greedy_completion(set(), order)
                if len(completed) > best_size:
                    best_size = len(completed)
                    best_result = completed
            
            # Strategy 6: Conflict-density greedy
            # Sort by number of conflicts (fewer conflicts first)
            def conflict_degree(i):
                cnt = 0
                for (a, b) in sum_pairs[i]:
                    cnt += 2  # Both a and b could conflict
                for (d, j) in add_conflicts[i]:
                    cnt += 2
                return cnt
            
            order_conflict = sorted(range(n), key=conflict_degree)
            completed = greedy_completion(set(), order_conflict)
            if len(completed) > best_size:
                best_size = len(completed)
                best_result = completed
            
            # Now apply 2-opt local search from the best starting point
            if best_result:
                improved = local_search_2opt(best_result)
                if len(improved) > best_size:
                    best_result = improved
                    best_size = len(improved)
            
            # If we have time, try local search from other good starting points
            if time.time() < my_deadline - 10:
                # Try from sign partitions
                for s in [pos, neg]:
                    if time.time() > my_deadline - 5:
                        break
                    completed = greedy_completion(s)
                    if len(completed) >= best_size - 2:  # Only if close to best
                        improved = local_search_2opt(completed)
                        if len(improved) > best_size:
                            best_result = improved
                            best_size = len(improved)
            
            return best_result

        # Run the multi-start search
        best_set = multi_start_search()
        
        # Final polish: greedy completion
        best_set = greedy_completion(best_set)
        
        # Convert to values
        best_result = [elements[i] for i in sorted(best_set)]
        
        # Final Verification
        if not is_sum_free(best_result):
            result_set = set(best_result)
            changed = True
            while changed:
                changed = False
                for a in list(result_set):
                    for b in list(result_set):
                        if a + b in result_set:
                            result_set.discard(a + b)
                            changed = True
            best_result = list(result_set)
            if not is_sum_free(best_result):
                # Fallback: take largest by absolute value
                best_result = sorted(best_result, key=lambda x: -abs(x))[:len(best_result)//2]
        
        return best_result

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = deadline - time.time()
        if remaining < 5:
            elems = instance(size, seed)
            results.append([x for x in elems if abs(x) % 2 == 1])
            continue

        time_budget = min(remaining - 3, 50.0)
        elems = instance(size, seed)
        result = find_sum_free(elems, time_budget=time_budget)
        results.append(result)

    return results
