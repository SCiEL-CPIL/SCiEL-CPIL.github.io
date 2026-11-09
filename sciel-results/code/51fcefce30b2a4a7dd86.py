"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free_subset(elements, subset_set):
        """Check if subset_set is sum-free within the context of elements (actually just subset_set itself)."""
        s = subset_set
        for a in s:
            if (2 * a) in s:
                return False
            for b in s:
                if b != a and (a + b) in s:
                    return False
        return True

    def can_add_element(x, s):
        """Check if x can be added to sum-free set s without creating conflicts."""
        if 2 * x in s:
            return False
        for y in s:
            if y != x:
                if x + y in s:
                    return False
                if x - y in s:
                    return False
        return True

    def greedy_sum_free(elements, order_key=None):
        """
        Strategy:
        1. Start with 'High Magnitude' candidates:
           - Positives > max_pos / 2
           - Negatives < min_neg / 2
        2. Resolve conflicts in this initial set.
        3. Greedily add remaining elements using specified ordering.
        """
        if not elements:
            return []
        
        pos = [x for x in elements if x > 0]
        neg = [x for x in elements if x < 0]
        
        max_pos = max(pos) if pos else 0
        min_neg = min(neg) if neg else 0
        
        # Initial candidates
        # For positives, we want x such that 2x > max_pos.
        # For negatives, we want x such that 2x < min_neg.
        candidates = []
        for x in pos:
            if 2 * x > max_pos:
                candidates.append(x)
        for x in neg:
            if 2 * x < min_neg:
                candidates.append(x)
        
        selected_set = set(candidates)
        
        # Resolve conflicts in selected_set
        def has_conflict(s):
            for a in s:
                if 2*a in s:
                    return True
                for b in s:
                    if b != a and a+b in s:
                        return True
            return False

        while has_conflict(selected_set):
            removed = False
            for a in list(selected_set):
                if 2*a in selected_set:
                    selected_set.remove(2*a)
                    removed = True
                    break
                for b in selected_set:
                    if b != a:
                        s_val = a + b
                        if s_val in selected_set:
                            selected_set.remove(s_val)
                            removed = True
                            break
                if removed:
                    break
            if not removed:
                break

        # Now greedily add remaining elements
        remaining = [x for x in elements if x not in selected_set]
        if order_key:
            remaining.sort(key=order_key)
        else:
            remaining.sort(key=lambda x: -abs(x))
        
        for x in remaining:
            if can_add_element(x, selected_set):
                selected_set.add(x)
        
        return list(selected_set)

    def local_search_improve(elements, initial_subset, max_iterations=5000):
        """
        Try to improve the subset by swapping elements.
        For each element in the subset, try removing it and adding a different element.
        """
        s = set(initial_subset)
        remaining = [x for x in elements if x not in s]
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for x in list(s):
                # Try removing x and adding some element from remaining
                s.remove(x)
                remaining.append(x)
                
                best_add = None
                for y in remaining:
                    if y not in s and can_add_element(y, s):
                        # Check if this gives us a larger set (it does: -1 + 1 = 0, but maybe enables more)
                        # Actually, swapping doesn't change size. We need a different approach.
                        # Let's try: remove x, then greedily add from remaining
                        pass
                
                # Better approach: remove x, then try to add multiple elements
                s_no_x = set(s)
                added_any = False
                for y in list(remaining):
                    if can_add_element(y, s_no_x):
                        s_no_x.add(y)
                        added_any = True
                
                if len(s_no_x) > len(s):
                    s = s_no_x
                    remaining = [x for x in elements if x not in s]
                    improved = True
                    break
                else:
                    # Put x back
                    s.add(x)
                    remaining.remove(x)
        
        return list(s)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > timeout:
            # Return best effort (empty or previous)
            results.append([])
            continue
            
        elems = instance(size, seed)
        
        # Try multiple orderings and pick the best
        orderings = [
            lambda x: -abs(x),           # Largest absolute value first
            lambda x: x,                 # Smallest value first
            lambda x: -x,                # Largest value first
            lambda x: abs(x),            # Smallest absolute value first
            lambda x: (0 if x > 0 else 1, -abs(x)),  # Positives first by magnitude
            lambda x: (1 if x > 0 else 0, abs(x)),   # Negatives first by magnitude
        ]
        
        best_subset = None
        best_size = -1
        
        for order in orderings:
            if time.time() - start_time > timeout:
                break
            subset = greedy_sum_free(elems, order_key=order)
            if len(subset) > best_size:
                best_size = len(subset)
                best_subset = subset
        
        # Local search improvement
        if best_subset is not None and time.time() - start_time < timeout - 5:
            improved = local_search_improve(elems, best_subset, max_iterations=2000)
            if len(improved) > best_size:
                best_subset = improved
        
        results.append(best_subset if best_subset is not None else [])

    return results
# EVOLVE-BLOCK-END
