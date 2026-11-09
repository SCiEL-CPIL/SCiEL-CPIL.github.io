import time
import random

def construct():
    n = 12
    deadline = time.time() + 55  # Safety margin
    
    def is_valid_set(lst):
        """Check if all subset sums are distinct."""
        subset_sums = {0}
        for x in lst:
            new_sums = {s + x for s in subset_sums}
            if subset_sums & new_sums:
                return False
            subset_sums = subset_sums | new_sums
        return True
    
    def find_min_replacement(prefix, exclude_val, deadline):
        """Find smallest valid replacement for exclude_val given prefix."""
        prefix_sums = {0}
        for p in prefix:
            new_sums = {s + p for s in prefix_sums}
            if prefix_sums & new_sums:
                return None  # prefix itself is invalid
            prefix_sums = prefix_sums | new_sums
        
        candidate = 1
        while candidate < exclude_val:
            if time.time() > deadline:
                return None
            if candidate not in prefix:
                new_sums = {s + candidate for s in prefix_sums}
                if not (prefix_sums & new_sums):
                    return candidate
            candidate += 1
        return None
    
    # Initial greedy solution
    elements = []
    subset_sums = {0}
    used = set()
    for i in range(n):
        if time.time() > deadline:
            break
        candidate = 1
        while True:
            if time.time() > deadline:
                break
            if candidate not in used:
                new_sums = {s + candidate for s in subset_sums}
                if not (subset_sums & new_sums):
                    elements.append(candidate)
                    used.add(candidate)
                    subset_sums = subset_sums | new_sums
                    break
            candidate += 1
    if len(elements) == n:
        best_set = elements
        best_max = max(elements)
    else:
        best_set = None
        best_max = float('inf')
        
    # Local search
    while time.time() < deadline:
        # Start from best_set
        if not best_set:
            break
        current = list(best_set)
        current.sort()
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current.sort()
            
            # Move 1: Replace largest element with smaller valid one
            largest_idx = n - 1
            largest_val = current[largest_idx]
            prefix = current[:n-1]
            
            replacement = find_min_replacement(prefix, largest_val, deadline)
            if replacement is not None:
                new_set = prefix + [replacement]
                new_set.sort()
                if max(new_set) < max(current):
                    current = new_set
                    improved = True
                    continue
            
            # Move 2: Try swapping two elements to reduce max
            # For each pair (i, j) where i < j, try replacing current[j] with a smaller value
            # while keeping current[i] fixed, but also consider removing current[i]
            # and replacing both with two smaller values.
            
            # Simpler: try to reduce each element in order from largest
            for idx in range(n-1, -1, -1):
                if time.time() > deadline:
                    break
                val = current[idx]
                # Build prefix without this element
                prefix = current[:idx] + current[idx+1:]
                
                replacement = find_min_replacement(prefix, val, deadline)
                if replacement is not None:
                    new_set = prefix + [replacement]
                    new_set.sort()
                    if max(new_set) < max(current):
                        current = new_set
                        improved = True
                        break
            
            # Move 3: Random perturbation - replace a random element with a random valid one
            if not improved and time.time() < deadline:
                # Try a random move: pick a random index, try to replace with random smaller value
                idx = random.randint(0, n-1)
                val = current[idx]
                prefix = current[:idx] + current[idx+1:]
                
                # Try a few random candidates
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    candidate = random.randint(1, max(1, val - 1))
                    if candidate not in prefix:
                        prefix_sums = {0}
                        for p in prefix:
                            new_sums = {s + p for s in prefix_sums}
                            prefix_sums = prefix_sums | new_sums
                        
                        new_sums = {s + candidate for s in prefix_sums}
                        if not (prefix_sums & new_sums):
                            new_set = prefix + [candidate]
                            new_set.sort()
                            # Accept if it reduces max or is same max (for diversity)
                            if max(new_set) <= max(current):
                                current = new_set
                                improved = True
                                break
        
        if max(current) < best_max:
            best_set = current
            best_max = max(current)
            
    return best_set if best_set else [2**i for i in range(n)]
