import time
import random

def construct():
    deadline = time.time() + 85.0
    best_set = None
    best_len = 0
    
    def is_valid(s):
        s_sorted = sorted(s)
        n = len(s_sorted)
        for i in range(n):
            a = s_sorted[i]
            for j in range(i+1, n):
                b = s_sorted[j]
                for k in range(j, n):
                    c = s_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(x, current_set):
        if x in current_set:
            return False
        new_set = current_set | {x}
        return is_valid(new_set)

    def resolve_conflicts(s):
        """
        Greedily remove elements to make the set valid.
        Returns the modified set.
        """
        current = set(s)
        # Precompute conflicts to guide removal?
        # For simplicity, just loop until valid
        while True:
            if time.time() > deadline:
                return current
            
            # Find a conflict
            s_sorted = sorted(current)
            n = len(s_sorted)
            conflict_found = None
            # Count conflicts for each element to decide what to remove
            conflict_count = {x: 0 for x in current}
            
            for i in range(n):
                a = s_sorted[i]
                for j in range(i+1, n):
                    b = s_sorted[j]
                    for k in range(j, n):
                        c = s_sorted[k]
                        if (b + c) % a == 0:
                            if conflict_found is None:
                                conflict_found = (a, b, c)
                            conflict_count[a] += 1
                            conflict_count[b] += 1
                            conflict_count[c] += 1
                            
            if conflict_found is None:
                return current
            
            # Remove the element in the conflict with the highest count
            a, b, c = conflict_found
            candidates = [a, b, c]
            # Prefer removing the smallest? Or the one with highest count?
            # Removing the divisor 'a' might be good if 'a' is small and causes many issues.
            # Removing 'b' or 'c' might be better if they are large and unique.
            # Let's try removing the one with the highest conflict count.
            max_count = max(conflict_count[x] for x in candidates)
            to_remove = [x for x in candidates if conflict_count[x] == max_count]
            
            # If tie, remove the smallest?
            random.shuffle(to_remove)
            current.remove(to_remove[0])

    # Try starting suffixes
    # The valid suffix starts at 171.
    # We can try starting lower and resolving.
    for start in range(128, 172):
        if time.time() > deadline:
            break
        
        start_set = set(range(start, 257))
        
        # Resolve conflicts
        resolved_set = resolve_conflicts(start_set)
        
        # Try to add smaller elements
        current = resolved_set
        min_in_current = min(current) if current else 257
        
        candidates = list(range(1, min_in_current))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current):
                current.add(x)
        
        if len(current) > best_len:
            best_set = sorted(current)
            best_len = len(current)

    if best_set is None:
        best_set = list(range(171, 257))
        
    return best_set
