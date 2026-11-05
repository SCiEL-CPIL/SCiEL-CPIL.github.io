import time

def construct():
    """
    Returns a nonempty subset of [1, 256] maximizing cardinality
    such that for every a < b <= c in the set, a does not divide b + c.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, 10s safety margin for 60s total
    
    # Initial known good solution: Upper tail [171, 256]
    # Verification: For a >= 171, max sum b+c <= 256+256=512.
    # Min multiple of a greater than 2a is 3a.
    # 3 * 171 = 513 > 512.
    # Thus no b+c is a multiple of a.
    S = set(range(171, 257))
    
    def is_valid(S):
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                for k in range(j, n):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Verify initial solution (should be fast enough for 86 elements)
    if not is_valid(S):
        # Fallback to smaller tail if logic fails (should not happen)
        S = set(range(192, 257))
        if not is_valid(S):
            # Last resort
            return list(range(1, 2))

    # Try to improve by adding smaller elements
    # We iterate downwards from 170 to 1
    # We only accept an addition if it doesn't require removals,
    # or if it allows a 1-for-1 swap that we can greedily find.
    
    # Optimization: Precompute conflicts for potential additions
    # For a candidate a, conflicts are pairs (b,c) in S such that a | (b+c).
    
    for a in range(170, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if a can be added directly
        conflicts = []
        S_list = sorted(S)
        for i, b in enumerate(S_list):
            if b <= a:
                continue
            for c in S_list[i:]:
                if (b + c) % a == 0:
                    conflicts.append((b, c))
        
        if not conflicts:
            S.add(a)
            continue
            
        # If there are conflicts, try to resolve with 1 removal
        # Find an element x in S that is part of all conflicts?
        # Or simply, try removing each candidate x from the union of conflict elements
        # and check if S - {x} + {a} is valid.
        
        candidates_to_remove = set()
        for b, c in conflicts:
            candidates_to_remove.add(b)
            candidates_to_remove.add(c)
            
        # Limit checks to avoid O(N^3) blowup per swap
        if len(candidates_to_remove) > 20:
            continue
            
        found_swap = False
        for x in candidates_to_remove:
            if time.time() > deadline:
                break
            new_S = S - {x}
            new_S.add(a)
            # Quick check: does x resolve all conflicts?
            # Not necessarily, x might not be in all conflicts.
            # We must verify the new set.
            # To speed up, we only check conflicts involving a and new pairs?
            # Actually, removing x might create new issues? No, removing elements
            # only reduces sums. Adding a introduces new constraints.
            # The old set was valid. Removing x keeps it valid.
            # Adding a might violate constraints with other pairs.
            # So we just need to check if a divides any b+c in (S - {x}).
            
            valid = True
            for b, c in conflicts:
                if x == b or x == c:
                    continue
                # This conflict still exists
                # Wait, if x is not in (b,c), the conflict remains.
                # So x must be in EVERY conflict to resolve them all by removal.
                # This is a necessary condition for 1-removal fix.
                pass
            
            # Better logic: 
            # If we remove x, the set of conflicts remaining is those (b,c) 
            # such that x not in {b,c}.
            # If any such conflict remains, the set is invalid.
            # So x must be in the intersection of all conflict pairs.
            
            # Let's find the intersection of all conflict pairs
            if conflicts:
                common = set(conflicts[0])
                for b, c in conflicts[1:]:
                    common.intersection_update({b, c})
                if x not in common:
                    continue
            
            # If x is in all conflicts, removing it resolves all conflicts involving a.
            # Does removing x break anything else? No.
            # So new_S is valid.
            S = new_S
            found_swap = True
            break
            
    return list(S)

# EVOLVE-BLOCK-END
