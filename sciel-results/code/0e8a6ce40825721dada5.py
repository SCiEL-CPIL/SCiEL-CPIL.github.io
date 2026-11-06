"""Improved sum-free subset finder using local search."""
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def find_sum_free(elements, time_budget=40.0):
        """Find a large sum-free subset using local search."""
        if not elements:
            return []
        
        n = len(elements)
        elem_set = set(elements)
        elem_to_idx = {x: i for i, x in enumerate(elements)}
        
        # Precompute: for each pair (i,j), check if elements[i]+elements[j] is in the set
        # This is O(n^2) which is fine for n up to 240
        sum_in_set = {}  # (i,j) -> index of sum, or None
        for i in range(n):
            for j in range(i, n):  # a=b allowed, so i<=j
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    sum_in_set[(i, j)] = elem_to_idx[s]
        
        def is_sum_free(active):
            """Check if active indices form a sum-free set."""
            for (i, j), k in sum_in_set.items():
                if i in active and j in active and k in active:
                    return False
            return True
        
        def conflict_count(active):
            """Count number of violated constraints."""
            count = 0
            for (i, j), k in sum_in_set.items():
                if i in active and j in active and k in active:
                    count += 1
            return count
        
        # Strategy 1: Greedy removal
        def greedy_removal():
            active = set(range(n))
            # Remove elements involved in conflicts, preferring high-degree
            while conflict_count(active) > 0:
                # Find all conflicts
                conflicts = []
                for (i, j), k in sum_in_set.items():
                    if i in active and j in active and k in active:
                        conflicts.append((i, j, k))
                if not conflicts:
                    break
                # Count how many conflicts each element is in
                degree = defaultdict(int)
                for i, j, k in conflicts:
                    degree[i] += 1
                    degree[j] += 1
                    degree[k] += 1
                # Remove the element with highest degree
                to_remove = max(degree, key=degree.get)
                active.remove(to_remove)
            return active
        
        # Strategy 2: Start from a large independent set and try to add
        def greedy_addition(order):
            active = set()
            # Try adding elements in given order
            for idx in order:
                # Check if adding idx creates a conflict
                can_add = True
                for (i, j), k in sum_in_set.items():
                    # Check if idx participates in a new conflict
                    if k == idx:
                        if i in active and j in active:
                            can_add = False
                            break
                    elif i == idx:
                        if j in active and k in active:
                            can_add = False
                            break
                    elif j == idx:
                        if i in active and k in active:
                            can_add = False
                            break
                if can_add:
                    active.add(idx)
            return active
        
        # Strategy 3: Local search - start from greedy, then swap
        def local_search(initial_active, max_iters=5000):
            active = set(initial_active)
            current_conflicts = conflict_count(active)
            best_active = set(active)
            best_size = len(active)
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if current_conflicts == 0:
                    # Try to add more elements
                    improved = False
                    for idx in range(n):
                        if idx not in active:
                            # Check if adding creates conflict
                            can_add = True
                            for (i, j), k in sum_in_set.items():
                                if k == idx:
                                    if i in active and j in active:
                                        can_add = False
                                        break
                                elif i == idx:
                                    if j in active and k in active:
                                        can_add = False
                                        break
                                elif j == idx:
                                    if i in active and k in active:
                                        can_add = False
                                        break
                            if can_add:
                                active.add(idx)
                                current_conflicts = conflict_count(active)
                                if len(active) > best_size:
                                    best_size = len(active)
                                    best_active = set(active)
                                improved = True
                                break
                    if not improved:
                        break
                    continue
                
                # Try to fix conflicts by removing or swapping
                # Find a conflict
                conflicts = []
                for (i, j), k in sum_in_set.items():
                    if i in active and j in active and k in active:
                        conflicts.append((i, j, k))
                
                if not conflicts:
                    break
                
                # Try removing one element from a random conflict
                i, j, k = random.choice(conflicts)
                candidates = [i, j, k]
                random.shuffle(candidates)
                
                best_new_conflicts = current_conflicts
                best_remove = None
                for c in candidates:
                    new_active = active - {c}
                    new_conflicts = conflict_count(new_active)
                    if new_conflicts < best_new_conflicts:
                        best_new_conflicts = new_conflicts
                        best_remove = c
                
                if best_remove is not None:
                    active.remove(best_remove)
                    current_conflicts = best_new_conflicts
                    if len(active) > best_size and is_sum_free(active):
                        best_size = len(active)
                        best_active = set(active)
                else:
                    # Try swapping: remove one, add another
                    i, j, k = random.choice(conflicts)
                    for c in [i, j, k]:
                        new_active = active - {c}
                        # Try to add a random inactive element
                        inactive = [x for x in range(n) if x not in new_active]
                        if not inactive:
                            continue
                        add_candidate = random.choice(inactive)
                        test_active = new_active | {add_candidate}
                        new_conflicts = conflict_count(test_active)
                        if new_conflicts < current_conflicts:
                            active = test_active
                            current_conflicts = new_conflicts
                            if len(active) > best_size and is_sum_free(active):
                                best_size = len(active)
                                best_active = set(active)
                            break
        
        # Try multiple strategies and keep the best
        best_result = []
        
        # Strategy 1: Greedy removal
        start_time = time.time()
        active1 = greedy_removal()
        if len(active1) > len(best_result):
            best_result = [elements[i] for i in active1]
        
        # Strategy 2: Greedy addition with different orders
        rng = random.Random(42)
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            order = list(range(n))
            rng.shuffle(order)
            active2 = greedy_addition(order)
            if len(active2) > len(best_result):
                best_result = [elements[i] for i in active2]
        
        # Strategy 3: Local search from best so far
        if best_result:
            best_indices = {elements.index(x) for x in best_result}
            # Run local search
            max_iters = 2000 if n <= 120 else 1000
            improved = local_search(best_indices, max_iters)
            if improved:
                best_result = [elements[i] for i in improved]
        
        # Verify result is sum-free
        result_set = set(best_result)
        for a in best_result:
            for b in best_result:
                if a + b in result_set:
                    # Should not happen, but just in case
                    pass
        
        return best_result
    
    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            # Fallback to simple odd-number heuristic
            elems = instance(size, seed)
            results.append([x for x in elems if x % 2])
            continue
        
        elems = instance(size, seed)
        # Allocate time based on instance size
        if size <= 60:
            time_budget = 30.0
        elif size <= 120:
            time_budget = 40.0
        else:
            time_budget = 50.0
        
        result = find_sum_free(elems, time_budget)
        results.append(result)
    
    return results
