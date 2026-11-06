"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def greedy_remove(arr):
        """Remove elements causing violations, starting with those involved in most violations."""
        s = set(arr)
        if not s:
            return list(s)
        
        # Count how many violations each element is involved in
        violation_count = {x: 0 for x in s}
        for a in s:
            for b in s:
                if a + b in s:
                    violation_count[a] += 1
                    violation_count[b] += 1
                    violation_count[a + b] += 1
        
        # Iteratively remove the element with the most violations
        current = list(s)
        current_set = set(s)
        
        while True:
            if time.time() > deadline:
                break
            # Find element with max violations
            if not current_set:
                break
            max_elem = max(current_set, key=lambda x: violation_count.get(x, 0))
            if violation_count.get(max_elem, 0) == 0:
                break
            current_set.remove(max_elem)
            current.remove(max_elem)
            
            # Recalculate violations after removal
            violation_count = {x: 0 for x in current_set}
            for a in current_set:
                for b in current_set:
                    if a + b in current_set:
                        violation_count[a] += 1
                        violation_count[b] += 1
                        violation_count[a + b] += 1
        
        return list(current_set)

    def local_search_add(arr, all_elements, max_iterations=2000):
        """Try to add back removed elements to improve size while maintaining sum-free."""
        s = set(arr)
        removed = list(all_elements - s)
        
        # Shuffle for diversity
        random.shuffle(removed)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            for elem in removed:
                if time.time() > deadline:
                    break
                if elem in s:
                    continue
                new_s = s | {elem}
                if is_sum_free(new_s):
                    s.add(elem)
                    removed.remove(elem)
                    improved = True
            if not improved:
                break
        
        return list(s)

    def local_search_swap(arr, all_elements, max_iterations=500):
        """Try swapping elements to potentially get a larger sum-free set."""
        s = set(arr)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            for elem_in in list(s):
                if time.time() > deadline:
                    break
                for elem_out in (all_elements - s):
                    if time.time() > deadline:
                        break
                    new_s = (s - {elem_in}) | {elem_out}
                    if is_sum_free(new_s):
                        s = new_s
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return list(s)

    def solve_instance(arr):
        """Solve a single instance to find a large sum-free subset."""
        n = len(arr)
        all_elements = set(arr)
        
        # Strategy 1: Start with odd numbers
        odd_subset = [x for x in arr if x % 2 != 0]
        current1 = set(odd_subset)
        
        # Try to add even numbers greedily
        even_numbers = [x for x in arr if x % 2 == 0]
        for even in even_numbers:
            if time.time() > deadline:
                break
            test_set = current1 | {even}
            if is_sum_free(test_set):
                current1.add(even)
        
        # Local search to add more
        current1 = local_search_add(list(current1), all_elements, max_iterations=500)
        
        # Strategy 2: Greedy from largest absolute value
        sorted_by_abs = sorted(arr, key=lambda x: -abs(x))
        current2 = set()
        for x in sorted_by_abs:
            if time.time() > deadline:
                break
            test_set = current2 | {x}
            if is_sum_free(test_set):
                current2.add(x)
        
        current2 = local_search_add(list(current2), all_elements, max_iterations=500)
        
        # Strategy 3: Greedy from smallest absolute value
        sorted_by_abs_asc = sorted(arr, key=lambda x: abs(x))
        current3 = set()
        for x in sorted_by_abs_asc:
            if time.time() > deadline:
                break
            test_set = current3 | {x}
            if is_sum_free(test_set):
                current3.add(x)
        
        current3 = local_search_add(list(current3), all_elements, max_iterations=500)
        
        # Choose the best
        best = max([list(current1), list(current2), list(current3)], key=len)
        
        # Try swap-based local search on the best
        swapped = local_search_swap(best, all_elements, max_iterations=200)
        if len(swapped) > len(best):
            best = swapped
        
        # Final verification
        if not is_sum_free(best):
            best = greedy_remove(best)
        
        return sorted(best)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            results.append([])
            continue
        
        arr = instance(size, seed)
        best_subset = solve_instance(arr)
        results.append(best_subset)
    
    return results
# EVOLVE-BLOCK-END
