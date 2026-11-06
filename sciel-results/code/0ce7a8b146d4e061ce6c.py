import time

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    n = 256
    
    def is_valid_addition(S, y):
        """Check if y can be added to S without creating a 3AP."""
        # Check if y is the middle term: exists u, w in S such that u + w == 2*y
        for w in S:
            if w == y:
                continue
            u = 2 * y - w
            if u in S and u != w and u != y:
                return False
        # Check if y is the first term: exists v, w in S such that 2*v == y + w
        for v in S:
            if v == y or v <= y:
                continue
            w = 2 * v - y
            if w in S and w != v and w != y:
                return False
        # Check if y is the third term: exists u, v in S such that 2*v == u + y
        for v in S:
            if v == y or v >= y:
                continue
            u = 2 * v - y
            if u in S and u != v and u != y:
                return False
        return True
    
    def is_valid_swap(S, x, y):
        """Check if (S - {x}) U {y} is 3-free. x in S, y not in S."""
        if x == y:
            return True
        # S\{x} is already 3-free. We just need to check if y creates a 3AP in S\{x}.
        S_new = S - {x}
        # Check if y is the middle term
        for w in S_new:
            if w == y:
                continue
            u = 2 * y - w
            if u in S_new and u != w and u != y:
                return False
        # Check if y is the first term
        for v in S_new:
            if v == y or v <= y:
                continue
            w = 2 * v - y
            if w in S_new and w != v and w != y:
                return False
        # Check if y is the third term
        for v in S_new:
            if v == y or v >= y:
                continue
            u = 2 * v - y
            if u in S_new and u != v and u != y:
                return False
        return True
    
    # 1. Generate the initial greedy set
    S = set()
    for x in range(1, n + 1):
        if is_valid_addition(S, x):
            S.add(x)
    
    # 2. Local Search / Hill Climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # First pass: Try to add any missing element if possible
        for y in range(1, n + 1):
            if y in S:
                continue
            if is_valid_addition(S, y):
                S.add(y)
                improved = True
        
        # Second pass: Try to swap larger elements with smaller missing ones
        if time.time() > deadline:
            break
        
        # Build list of missing elements in ascending order (prefer small y)
        missing = [y for y in range(1, n + 1) if y not in S]
        # Build list of S elements in descending order (prefer removing large x)
        S_desc = sorted(S, reverse=True)
        
        for y in missing:
            if time.time() > deadline:
                break
            # Try to find an x in S such that x > y and swap is valid
            for x in S_desc:
                if x <= y:
                    break
                if is_valid_swap(S, x, y):
                    S.remove(x)
                    S.add(y)
                    improved = True
                    # Restart S_desc since S changed
                    S_desc = sorted(S, reverse=True)
                    break  # Found a valid swap for this y, move to next y
        
        # Third pass: Try removing elements that don't help much
        # If an element x is large and its removal allows adding several smaller elements, consider it
        if time.time() > deadline:
            break
        
        for x in sorted(S, reverse=True):
            if time.time() > deadline:
                break
            # Try removing x
            S_without_x = S - {x}
            # Count how many missing elements can be added
            added = 0
            for y in range(1, n + 1):
                if y in S_without_x:
                    continue
                if is_valid_addition(S_without_x, y):
                    added += 1
                    S_without_x.add(y)
            
            # Compare reciprocal sums
            # Current sum: sum(1/x for x in S)
            # New sum: sum(1/x for x in S_without_x)
            # We want new sum > current sum
            # This is a heuristic; we only do this if it seems beneficial
            # For simplicity, just check if we added at least 1 element
            if added > 0:
                # Check if the new set has higher reciprocal sum
                # Quick check: if we removed x and added elements all < x, likely better
                # But let's just do it if added >= 1 and x is large
                if x > 100:  # Only remove large elements
                    S = S_without_x
                    improved = True
                    break  # Restart the while loop
    
    return sorted(S)
