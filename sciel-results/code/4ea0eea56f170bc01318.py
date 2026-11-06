import time
import random

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin before 60s kill

    def elapsed():
        return time.time() - start_time

    def check_and_add(selected, current_sums, x):
        """Check if x can be added. Returns (True, new_sums) or (False, None)."""
        n = len(selected)
        # Generate all new triple sums involving x
        # Type 1: (x, x, x) -> 3x
        # Type 2: (x, x, s) for s in selected -> 2x + s
        # Type 3: (x, s1, s2) for s1 <= s2 in selected (with replacement) -> x + s1 + s2
        
        new_sums = []
        
        # Type 1
        s = 3 * x
        if s in current_sums:
            return False, None
        new_sums.append(s)
        
        # Type 2
        for s in selected:
            val = 2 * x + s
            if val in current_sums:
                return False, None
            new_sums.append(val)
        
        # Type 3: x + s1 + s2 where s1 <= s2 from selected (combinations with replacement)
        for i in range(n):
            s1 = selected[i]
            for j in range(i, n):
                s2 = selected[j]
                val = x + s1 + s2
                if val in current_sums:
                    return False, None
                new_sums.append(val)
        
        # Check for duplicates among new_sums themselves
        if len(new_sums) != len(set(new_sums)):
            return False, None
        
        return True, new_sums

    def run_greedy(order):
        """Run greedy construction with given order."""
        selected = []
        current_sums = set()
        
        for x in order:
            if elapsed() > time_limit - 2.0:
                break
            ok, new_sums = check_and_add(selected, current_sums, x)
            if ok:
                selected.append(x)
                current_sums.update(new_sums)
        
        return selected

    def try_local_improvement(selected, current_sums, max_time=5.0):
        """Try to improve by removing one element and adding others."""
        if elapsed() > time_limit - 3.0:
            return selected, current_sums
        
        improved = False
        original_len = len(selected)
        
        # Try removing each element and see if we can add more
        for remove_idx in range(len(selected) - 1, -1, -1):
            if elapsed() > time_limit - 3.0:
                break
            
            x_remove = selected[remove_idx]
            new_selected = selected[:remove_idx] + selected[remove_idx+1:]
            
            # Rebuild sums for the reduced set
            new_sums = set()
            from itertools import combinations_with_replacement
            for triple in combinations_with_replacement(new_selected, 3):
                new_sums.add(sum(triple))
            
            # Try to add back x_remove and other candidates
            # First try adding back x_remove
            ok, added_sums = check_and_add(new_selected, new_sums, x_remove)
            if ok:
                new_selected.append(x_remove)
                new_sums.update(added_sums)
            
            # Try adding other candidates from 1 to 4096
            added_any = False
            for candidate in range(1, 4097):
                if elapsed() > time_limit - 3.0:
                    break
                if candidate in new_selected:
                    continue
                ok, added_sums = check_and_add(new_selected, new_sums, candidate)
                if ok:
                    new_selected.append(candidate)
                    new_sums.update(added_sums)
                    added_any = True
            
            if len(new_selected) > len(selected):
                selected = new_selected
                current_sums = new_sums
                improved = True
                break
        
        return selected, current_sums

    # Strategy 1: Greedy from 1 upward
    order1 = list(range(1, 4097))
    best_set = run_greedy(order1)
    best_size = len(best_set)
    
    # Build current_sums for best_set
    from itertools import combinations_with_replacement
    best_sums = set()
    for triple in combinations_with_replacement(best_set, 3):
        best_sums.add(sum(triple))
    
    # Try local improvement
    if elapsed() < time_limit - 5.0:
        best_set, best_sums = try_local_improvement(best_set, best_sums)
        best_size = len(best_set)
    
    # Strategy 2: Random orderings
    random.seed(42)
    for attempt in range(50):
        if elapsed() > time_limit - 5.0:
            break
        
        # Create a random ordering with bias toward smaller numbers
        # Use exponential distribution to favor smaller values
        order = list(range(1, 4097))
        random.shuffle(order)
        
        result = run_greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            best_sums = set()
            for triple in combinations_with_replacement(best_set, 3):
                best_sums.add(sum(triple))
    
    # Strategy 3: Try different ranges and step patterns
    if elapsed() < time_limit - 5.0:
        # Try starting from different bases
        for base in [1, 2, 3, 5, 7, 10]:
            if elapsed() > time_limit - 5.0:
                break
            order = list(range(base, 4097, 1))
            random.shuffle(order)
            result = run_greedy(order)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
                best_sums = set()
                for triple in combinations_with_replacement(best_set, 3):
                    best_sums.add(sum(triple))
    
    # Final local improvement
    if elapsed() < time_limit - 3.0:
        best_set, best_sums = try_local_improvement(best_set, best_sums)
        best_size = len(best_set)
    
    return best_set
