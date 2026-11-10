import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def compute_sums_set(S):
        """Compute the set of all pairwise sums a+b with a<=b from set S."""
        sums = set()
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            sums.add(2 * a)
            for b in S_sorted[i+1:]:
                sums.add(a + b)
        return sums
    
    def is_sidon(S):
        """Check if S is a Sidon set (all pairwise sums a+b with a<=b are distinct)."""
        S_sorted = sorted(S)
        sums = set()
        for i, a in enumerate(S_sorted):
            if 2 * a in sums:
                return False
            sums.add(2 * a)
            for b in S_sorted[i+1:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def is_maximal(S):
        """Check if S is inclusion-maximal: no element from [1,N]\S can be added."""
        S_set = set(S)
        sums = compute_sums_set(S)
        for z in range(1, N + 1):
            if z in S_set:
                continue
            # Check if z can be added
            if 2 * z in sums:
                continue
            can_add = True
            for s in S:
                if z + s in sums:
                    can_add = False
                    break
            if can_add:
                return False
        return True
    
    def sidon_and_maximal(S):
        """Check both Sidon and maximality."""
        if not is_sidon(S):
            return False
        return is_maximal(S)
    
    def greedy_set_cover_construct():
        """
        Greedy set-cover approach:
        - Start with empty set
        - At each step, pick the element that appears in the most "uncovered" collision hyperedges
        - A collision hyperedge is a set of elements whose inclusion would violate Sidon property
        - We want to minimize the final set size while ensuring maximality
        
        Actually, let me reconsider. The problem is:
        - Find a Sidon set S such that S is maximal (no element can be added)
        - Minimize |S|
        
        This is equivalent to: find the smallest set S such that:
        1. S is Sidon
        2. For every z in [1,N]\S, z conflicts with S (i.e., adding z would violate Sidon)
        
        The greedy approach:
        - Start with S = {}
        - While S is not maximal, add an element that "covers" the most uncovered conflicts
        - "Covers a conflict" means: for some z that could be added to S, adding x to S makes z no longer addable
        
        Let me implement this properly.
        """
        S = []
        S_set = set()
        sums = set()
        
        def get_addable_elements(S_set, sums):
            """Get all elements that can be added to S without violating Sidon."""
            addable = []
            for z in range(1, N + 1):
                if z in S_set:
                    continue
                if 2 * z in sums:
                    continue
                can_add = True
                for s in S:
                    if z + s in sums:
                        can_add = False
                        break
                if can_add:
                    addable.append(z)
            return addable
        
        def get_blocked_by(x, S_set, sums):
            """
            Given candidate x, determine which addable elements would be blocked
            if x is added to S.
            Returns the set of elements that are currently addable but would become
            non-addable after adding x.
            """
            # First, compute what sums would be added
            new_sums = {2 * x}
            for s in S:
                new_sums.add(x + s)
            
            # Check if x itself is valid
            if any(s in sums for s in new_sums):
                return set()  # x is not valid
            
            # Now, for each currently addable element z, check if it becomes non-addable
            blocked = set()
            for z in range(1, N + 1):
                if z in S_set or z == x:
                    continue
                # Check if z is currently addable
                if 2 * z in sums:
                    continue
                z_currently_addable = True
                for s in S:
                    if z + s in sums:
                        z_currently_addable = False
                        break
                if not z_currently_addable:
                    continue
                
                # Check if z would be blocked after adding x
                # z is blocked if:
                # 1. 2z is in new_sums
                # 2. z + x is in sums (existing) or in new_sums
                # 3. z + s is in new_sums for some s in S
                #    (Note: z+s in sums was already false since z was addable)
                
                # Actually, let me just check: can z be added to S ∪ {x}?
                # Sums in S ∪ {x} are: sums ∪ new_sums
                # z can be added if:
                # - 2z not in sums ∪ new_sums
                # - z + s not in sums ∪ new_sums for all s in S ∪ {x}
                
                # Since z was addable to S, we know:
                # - 2z not in sums
                # - z + s not in sums for all s in S
                
                # So z is blocked by x if:
                # - 2z in new_sums, OR
                # - z + x in sums ∪ new_sums, OR  
                # - z + s in new_sums for some s in S (but this is z+s = x+t for some t, or z+s = 2x)
                
                # Let me just check directly
                all_sums = sums | new_sums
                if 2 * z in all_sums:
                    blocked.add(z)
                    continue
                z_blocked = False
                for s in S:
                    if z + s in all_sums:
                        z_blocked = True
                        break
                if not z_blocked and z + x in all_sums:
                    z_blocked = True
                if z_blocked:
                    blocked.add(z)
            
            return blocked
        
        # Greedy: at each step, pick x that blocks the most addable elements
        while True:
            if time.time() > deadline:
                break
            
            addable = get_addable_elements(S_set, sums)
            if not addable:
                break
            
            best_x = None
            max_blocked = -1
            best_blocked_set = set()
            
            for x in addable:
                blocked = get_blocked_by(x, S_set, sums)
                if len(blocked) > max_blocked:
                    max_blocked = len(blocked)
                    best_x = x
                    best_blocked_set = blocked
            
            if best_x is None:
                break
            
            # Add best_x to S
            S.append(best_x)
            S_set.add(best_x)
            # Update sums
            sums.add(2 * best_x)
            for s in S[:-1]:
                sums.add(best_x + s)
            
            # Check if S is now maximal
            # S is maximal if there are no addable elements
            # (which is checked at the top of the loop)
        
        return S
    
    def try_remove_and_repair(S):
        """
        Try to remove elements from S and repair maximality by adding one element.
        Returns a smaller maximal Sidon set if found, else None.
        """
        S_sorted = sorted(S)
        n = len(S_sorted)
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                x1, x2 = S_sorted[i], S_sorted[j]
                S_rem = [s for s in S_sorted if s != x1 and s != x2]
                S_rem_set = set(S_rem)
                sums_rem = compute_sums_set(S_rem)
                
                # Find all elements that are addable to S_rem
                addable = []
                for z in range(1, N + 1):
                    if z in S_rem_set:
                        continue
                    if 2 * z in sums_rem:
                        continue
                    can_add = True
                    for s in S_rem:
                        if z + s in sums_rem:
                            can_add = False
                            break
                    if can_add:
                        addable.append(z)
                
                # We need to find y such that S' = S_rem ∪ {y} is maximal
                # This means: for all z in addable \ {y}, z is blocked by y
                # And y itself must be valid with S_rem (which it is, since y in addable)
                
                for y in addable:
                    # Check if y blocks all other addable elements
                    y_valid = True
                    # y is in addable, so it's valid with S_rem
                    
                    all_blocked = True
                    for z in addable:
                        if z == y:
                            continue
                        # Check if z is blocked by y given S_rem
                        # z is blocked if z cannot be added to S_rem ∪ {y}
                        # Sums in S_rem ∪ {y}: sums_rem ∪ {2y} ∪ {y+s for s in S_rem}
                        new_sums_y = {2 * y}
                        for s in S_rem:
                            new_sums_y.add(y + s)
                        all_sums = sums_rem | new_sums_y
                        
                        # Check if z can be added
                        if 2 * z in all_sums:
                            continue  # z is blocked
                        z_blocked = False
                        for s in S_rem:
                            if z + s in all_sums:
                                z_blocked = True
                                break
                        if not z_blocked and z + y in all_sums:
                            z_blocked = True
                        if not z_blocked:
                            all_blocked = False
                            break
                    
                    if all_blocked:
                        # Found a valid replacement
                        S_new = S_rem + [y]
                        # Verify
                        if sidon_and_maximal(S_new):
                            return S_new
                        # Even if verification fails (shouldn't happen), continue
                        # to find another valid y
        
        return None
    
    # Phase 1: Greedy set-cover construction
    S = greedy_set_cover_construct()
    if not S:
        S = [1]
    
    # Phase 2: Local search - try to reduce size by 2-for-1 swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_new = try_remove_and_repair(S)
        if S_new is not None and len(S_new) < len(S):
            S = S_new
            improved = True
    
    # Phase 3: Try single-element removals (shouldn't work for maximal sets, but check)
    if time.time() < deadline:
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            if time.time() > deadline:
                break
            S_rem = [s for s in S_sorted if s != S_sorted[i]]
            if is_sidon(S_rem) and is_maximal(S_rem):
                S = S_rem
                break
    
    return S
