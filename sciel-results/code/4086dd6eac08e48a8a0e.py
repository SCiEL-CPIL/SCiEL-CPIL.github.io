import time
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Find the largest set of distinct integers in 1..256 such that
    subset sums of different cardinalities are disjoint.
    Uses backtracking with cardinality-indexed subset sum sets.
    """
    deadline = time.time() + 50  # 50s safety margin before 60s limit
    best_set = list(range(226, 257))  # fallback: 31 elements AP
    best_size = len(best_set)
    
    # For a given set, compute all subset sums indexed by cardinality
    def compute_cardinality_sums(s):
        """Return dict: k -> set of all k-element subset sums"""
        n = len(s)
        sums_by_k = [set() for _ in range(n + 1)]
        sums_by_k[0].add(0)
        for x in s:
            for k in range(len(sums_by_k) - 1, -1, -1):
                if sums_by_k[k]:
                    for v in sums_by_k[k]:
                        sums_by_k[k + 1].add(v + x)
        return sums_by_k
    
    def is_valid(s):
        """Check if no two cardinalities share any sum value."""
        sums_by_k = compute_cardinality_sums(s)
        seen = set()
        for k in range(len(sums_by_k)):
            for v in sums_by_k[k]:
                if v in seen:
                    return False
                seen.add(v)
        return True
    
    def is_valid_fast(current_sums, new_elem, n):
        """
        Check if adding new_elem to current set (with current_sums)
        maintains validity. current_sums[k] = set of k-element sums.
        """
        # Compute new sums after adding new_elem
        # new_sums[k] = old_sums[k] ∪ {x + new_elem : x in old_sums[k-1]}
        # Check that no new sum collides with existing sums of different cardinality
        # Also check new sums don't collide with each other across cardinalities
        old_max_k = n  # current set has n elements
        # New set will have n+1 elements
        new_sums = [set() for _ in range(n + 2)]
        for k in range(old_max_k + 1):
            new_sums[k] = set(current_sums[k])
        for k in range(old_max_k, -1, -1):
            for v in current_sums[k]:
                new_sums[k + 1].add(v + new_elem)
        
        # Check cross-cardinality disjointness
        seen = set()
        for k in range(n + 2):
            for v in new_sums[k]:
                if v in seen:
                    return False
                seen.add(v)
        return True
    
    # Backtracking: try adding elements from large to small
    # We build the set incrementally
    def backtrack(available, current, current_sums, depth):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        n = len(current)
        # Update best
        if n > best_size:
            best_size = n
            best_set = list(current)
        
        if not available:
            return
        
        # Try adding each available element (larger first for better separation)
        for i in range(len(available) - 1, -1, -1):
            if time.time() > deadline:
                return
            elem = available[i]
            new_avail = available[:i] + available[i+1:]
            
            # Quick check: can we add this?
            # Compute new cardinality sums
            new_sums = [set() for _ in range(n + 2)]
            for k in range(n + 1):
                new_sums[k] = set(current_sums[k])
            for k in range(n, -1, -1):
                for v in current_sums[k]:
                    new_sums[k + 1].add(v + elem)
            
            # Check validity
            valid = True
            seen = set()
            for k in range(n + 2):
                for v in new_sums[k]:
                    if v in seen:
                        valid = False
                        break
                    seen.add(v)
                if not valid:
                    break
            
            if valid:
                current.append(elem)
                backtrack(new_avail, current, new_sums, depth + 1)
                current.pop()
    
    # Start with empty set
    available = list(range(256, 0, -1))  # 256 down to 1
    current_sums = [set() for _ in range(1)]
    current_sums[0].add(0)
    
    backtrack(available, [], current_sums, 0)
    
    # Ensure we return Python ints in a list
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
