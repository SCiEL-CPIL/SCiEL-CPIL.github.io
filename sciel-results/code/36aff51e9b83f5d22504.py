"""Rewritten program: structural greedy + local search for divisibility-incomparable subset sums."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute divisor counts and proper divisors for 1..256
    div_count = [0] * 257
    proper_divs = [[] for _ in range(257)]
    for i in range(1, 257):
        for j in range(1, i):
            if i % j == 0:
                div_count[i] += 1
                proper_divs[i].append(j)
    
    # Precompute multiples table: multiples[i] = list of multiples of i in 1..256 (excluding i)
    multiples = [[] for _ in range(257)]
    for i in range(1, 257):
        m = 2 * i
        while m <= 256:
            multiples[i].append(m)
            m += i
    
    def validate_set(elements):
        """Check if elements form a valid set. Returns True if valid."""
        n = len(elements)
        if n == 0:
            return True
        # Compute all nonempty subset sums
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False
            sums.add(s)
        # Check divisibility: no two distinct sums divide each other
        sl = sorted(sums)
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                if sl[j] % sl[i] == 0:
                    return False
        return True
    
    def compute_subset_sums(elements):
        """Return the set of all nonempty subset sums."""
        n = len(elements)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums
    
    def check_divisibility(sums):
        """Check if no two distinct sums in the set divide each other."""
        sl = sorted(sums)
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                if sl[j] % sl[i] == 0:
                    return False
        return True
    
    def try_add_element(elements, current_sums, x):
        """Try to add x to elements. Returns (new_elements, new_sums) or None if invalid."""
        # Compute new subset sums: old sums + x, plus x itself
        new_sums = set()
        for s in current_sums:
            ns = s + x
            if ns in current_sums:
                return None  # collision
            if ns in new_sums:
                return None  # collision
            new_sums.add(ns)
        new_sums.add(x)
        
        if x in current_sums:
            return None  # collision
        
        all_sums = current_sums | new_sums
        if not check_divisibility(all_sums):
            return None
        
        return elements + [x], all_sums
    
    def greedy_construct(order, max_size=30):
        """Build a valid set by adding elements in the given order."""
        current = []
        current_sums = set()
        
        for x in order:
            if x in current:
                continue
            result = try_add_element(current, current_sums, x)
            if result is not None:
                current, current_sums = result
                if len(current) >= max_size:
                    break
        
        return current, current_sums
    
    def local_search(elements, current_sums, candidates, max_iterations=200):
        """Try to improve the set using insert and swap moves."""
        best_elements = list(elements)
        best_sums = set(current_sums)
        best_size = len(best_elements)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try insert moves: add a candidate element
            for x in candidates:
                if x in best_elements:
                    continue
                result = try_add_element(best_elements, best_sums, x)
                if result is not None:
                    new_elements, new_sums = result
                    if len(new_elements) > best_size:
                        best_elements = new_elements
                        best_sums = new_sums
                        best_size = len(best_elements)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swap moves: replace one element with a candidate
            for i, old_x in enumerate(best_elements):
                # Remove old_x
                remaining = best_elements[:i] + best_elements[i+1:]
                remaining_sums = compute_subset_sums(remaining)
                
                for x in candidates:
                    if x in remaining:
                        continue
                    result = try_add_element(remaining, remaining_sums, x)
                    if result is not None:
                        new_elements, new_sums = result
                        if len(new_elements) > best_size:
                            best_elements = new_elements
                            best_sums = new_sums
                            best_size = len(best_elements)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_elements, best_sums
    
    # Phase 1: Multiple greedy attempts with different orderings
    candidates_all = list(range(1, 257))
    
    # Ordering 1: Descending (large numbers first)
    desc_order = sorted(candidates_all, reverse=True)
    
    # Ordering 2: Ascending
    asc_order = sorted(candidates_all)
    
    # Ordering 3: By divisor count (fewer divisors first)
    div_order = sorted(candidates_all, key=lambda x: (div_count[x], x))
    
    # Ordering 4: Random with bias toward larger numbers
    rng = random.Random(12345)
    biased_order = sorted(candidates_all, key=lambda x: (-x, rng.random()))
    
    orderings = [desc_order, asc_order, div_order, biased_order]
    
    # Add some random orderings
    for _ in range(30):
        shuffled = candidates_all[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    best_set = []
    best_sums = set()
    best_size = 0
    
    for order in orderings:
        if time.time() > deadline:
            break
        elements, sums = greedy_construct(order)
        if len(elements) > best_size:
            best_size = len(elements)
            best_set = elements
            best_sums = sums
    
    # Phase 2: Local search to improve
    if time.time() < deadline:
        remaining = [x for x in candidates_all if x not in best_set]
        improved_set, improved_sums = local_search(best_set, best_sums, remaining, max_iterations=100)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_sums = improved_sums
            best_size = len(best_set)
    
    # Phase 3: More local search iterations with different candidate orderings
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            remaining = [x for x in candidates_all if x not in best_set]
            rng.shuffle(remaining)
            improved_set, improved_sums = local_search(best_set, best_sums, remaining, max_iterations=50)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_sums = improved_sums
                best_size = len(best_set)
    
    # Phase 4: Final greedy pass to see if we can add more
    if time.time() < deadline:
        remaining = [x for x in candidates_all if x not in best_set]
        rng.shuffle(remaining)
        elements, sums = greedy_construct(remaining, max_size=30 - len(best_set))
        if len(elements) > 0:
            # Try to merge
            merged_elements = list(best_set) + elements
            merged_sums = compute_subset_sums(merged_elements)
            if len(merged_elements) > best_size and check_divisibility(merged_sums) and len(merged_sums) == (1 << len(merged_elements)) - 1:
                best_set = merged_elements
                best_sums = merged_sums
                best_size = len(best_set)
    
    # Final verification
    if not validate_set(best_set):
        # Fallback to a known valid small set
        best_set = [31, 32, 34]
    
    return list(best_set)
# EVOLVE-BLOCK-END
