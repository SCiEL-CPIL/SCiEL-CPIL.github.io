import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    def has_subset_sum_200(elements):
        """Check if any subset of elements sums to 200 using bitset."""
        reachable = 1  # bit 0 set
        mask = (1 << 201) - 1
        for x in elements:
            if (reachable >> (200 - x)) & 1:
                return True
            reachable = (reachable | (reachable << x)) & mask
        return False
    
    # Structural insight: In [51, 100], only triples can sum to 200
    # (2 elements: max 100+99=199, 4 elements: min 51+52+53+54=210)
    
    # Generate all bad triples in [51, 100]
    triples = []
    for z in range(100, 67, -1):
        target_xy = 200 - z
        min_x = max(51, 200 - 2*z + 1)
        max_x = (target_xy - 1) // 2
        for x in range(min_x, max_x + 1):
            y = target_xy - x
            if x < y < z:
                triples.append((x, y, z))
    
    # Greedy hitting set: remove elements that break the most remaining triples
    def greedy_hitting_set(triples_list):
        if not triples_list:
            return set()
        
        H = set()
        active = set(triples_list)
        
        while active:
            if time.time() > deadline:
                break
            # Count frequency of each element in active triples
            counts = {}
            for t in active:
                for v in t:
                    counts[v] = counts.get(v, 0) + 1
            
            if not counts:
                break
            
            # Pick element with max frequency
            best_elem = max(counts, key=lambda v: counts[v])
            H.add(best_elem)
            
            # Remove all triples containing best_elem
            active = {t for t in active if best_elem not in t}
        
        return H
    
    # Strategy 1: Use triple-based hitting set on [51,100]
    H = greedy_hitting_set(triples)
    base_set = [v for v in range(51, 101) if v not in H]
    
    # Try to add elements from [1, 50] using greedy
    def try_add_elements(base, candidates):
        current = list(base)
        reachable = 1
        mask = (1 << 201) - 1
        added = []
        
        for x in candidates:
            if (reachable >> (200 - x)) & 1:
                continue
            current.append(x)
            reachable = (reachable | (reachable << x)) & mask
            added.append(x)
        
        return current, added
    
    # Try adding small elements in various orders
    best_set = base_set
    best_size = len(base_set)
    
    # Verify base set is valid
    if not has_subset_sum_200(base_set):
        best_set = base_set
    else:
        # Fallback: use pure greedy
        best_set = []
    
    small_elements = list(range(1, 51))
    
    # Try descending order for small elements
    order = sorted(small_elements, reverse=True)
    result, _ = try_add_elements(best_set, order)
    if not has_subset_sum_200(result) and len(result) > best_size:
        best_size = len(result)
        best_set = result
    
    # Try ascending order
    order = sorted(small_elements)
    result, _ = try_add_elements(best_set, order)
    if not has_subset_sum_200(result) and len(result) > best_size:
        best_size = len(result)
        best_set = result
    
    # Randomized trials
    rng = random.Random(42)
    base_list = list(range(1, 101))
    
    for i in range(50000):
        if time.time() > deadline:
            break
        
        rng.shuffle(base_list)
        
        current = []
        reachable = 1
        mask = (1 << 201) - 1
        
        for x in base_list:
            if (reachable >> (200 - x)) & 1:
                continue
            current.append(x)
            reachable = (reachable | (reachable << x)) & mask
        
        if len(current) > best_size:
            if not has_subset_sum_200(current):
                best_size = len(current)
                best_set = current
    
    # Local search: try to add back excluded elements
    if best_set:
        excluded = [x for x in range(1, 101) if x not in best_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur = list(best_set)
            for x in excluded:
                if time.time() > deadline:
                    break
                if x in cur:
                    continue
                cand = cur + [x]
                if not has_subset_sum_200(cand):
                    cur = cand
                    excluded.remove(x)
                    improved = True
            
            if len(cur) > best_size:
                best_size = len(cur)
                best_set = cur
    
    # Ensure non-empty
    if not best_set:
        best_set = [100]
    
    return best_set
