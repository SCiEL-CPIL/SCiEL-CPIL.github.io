import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit

    N = 192
    MAX_SIZE = 48

    def time_left():
        return deadline - time.time()

    def can_add(S, x, pair_sums):
        """
        Check if x can be added to S.
        S: set of selected integers
        x: candidate integer
        pair_sums: dict mapping a in S to set of (b+c)%a for distinct b,c in S\{a}
        """
        S_list = list(S)
        n = len(S_list)
        
        # Check 1: No 3 distinct in S sum to 0 mod x
        if n >= 3:
            if x == 1:
                return False
            # Iterate pairs (b, c) in S
            # We need to check if d = (-b-c) % x is in S and d != b, d != c
            # To avoid duplicates, we can just check all pairs.
            # Optimization: Use a set for S for O(1) lookup.
            S_set = S
            for i in range(n):
                b = S_list[i]
                for j in range(i + 1, n):
                    c = S_list[j]
                    d = (-b - c) % x
                    if d in S_set and d != b and d != c:
                        return False
        
        # Check 2: For each a in S, no 3 distinct in (S\{a}) U {x} sum to 0 mod a
        # This is equivalent to: for each a in S, does (S\{a}) contain a pair (b,c) 
        # such that (x+b+c) % a == 0? i.e., (b+c) % a == (-x) % a?
        for a in S_list:
            if a == 1:
                # If a=1, any sum is 0 mod 1. 
                # We need 3 distinct others. 
                # S\{a} has n-1 elements. Plus x makes n elements.
                # If n >= 3, there are at least 3 elements in S\{a} U {x}.
                # So if a=1 is in S and n >= 3, adding x makes n+1 >= 4 elements.
                # Wait, if a=1 is in S, and we add x, the set becomes S U {x}.
                # For a=1, we need to check if there are 3 distinct others in S U {x} \ {1}.
                # S U {x} \ {1} has size n (if 1 in S) or n+1 (if 1 not in S).
                # If 1 in S, size is n. If n >= 3, it's invalid.
                # So if 1 in S and n >= 3, we cannot add x.
                # Actually, if 1 in S and n >= 3, S is already invalid?
                # No, S is valid. If 1 in S, S must have size < 4.
                # So if 1 in S, n <= 3.
                # If n == 3, S\{1} has 2 elements. Adding x makes 3 elements.
                # Sum of 3 distinct others for a=1: the 3 others are S\{1} U {x}.
                # Their sum is divisible by 1. So invalid.
                # So if 1 in S, we can only add x if resulting size < 4?
                # No, if resulting size is 4, a=1 has 3 others. Invalid.
                # So if 1 in S, max size is 3.
                if n >= 3:
                    return False
                continue
            
            target = (-x) % a
            if target in pair_sums.get(a, set()):
                return False
        
        return True

    def add_element(S, x, pair_sums):
        """Add x to S and update pair_sums."""
        S.add(x)
        
        # Update pair_sums for existing a in S
        S_without_x = S - {x}
        for a in S_without_x:
            if a == 1:
                continue
            # Add pairs (x, b) for b in S_without_x \ {a}
            # Note: pair_sums[a] contains (b+c)%a for b,c in S\{a}
            # Now S is S_old U {x}. S\{a} is (S_old\{a}) U {x}
            # New pairs involving x: (x, b) where b in S_old\{a}
            for b in S_without_x:
                if b != a:
                    pair_sums[a].add((x + b) % a)
        
        # Initialize pair_sums for x
        if x != 1:
            pair_sums[x] = set()
            # Pairs (b, c) in S_without_x
            S_list = list(S_without_x)
            n = len(S_list)
            for i in range(n):
                b = S_list[i]
                for j in range(i + 1, n):
                    c = S_list[j]
                    pair_sums[x].add((b + c) % x)

    def remove_element(S, x, pair_sums):
        """Remove x from S and update pair_sums. 
        Note: This is tricky because we need to remove specific sums.
        It's easier to rebuild pair_sums for all a in S\{x} if needed, 
        or just rebuild the whole structure if swaps are rare.
        Given the speed of building, rebuilding is acceptable for local search steps.
        """
        S.remove(x)
        # Rebuild pair_sums for all a in S
        new_pair_sums = {}
        for a in S:
            if a == 1:
                continue
            new_pair_sums[a] = set()
            S_list = list(S - {a})
            n = len(S_list)
            for i in range(n):
                b = S_list[i]
                for j in range(i + 1, n):
                    c = S_list[j]
                    new_pair_sums[a].add((b + c) % a)
        # Replace pair_sums content
        pair_sums.clear()
        pair_sums.update(new_pair_sums)

    def greedy_build(order):
        S = set()
        pair_sums = {}
        for x in order:
            if len(S) >= MAX_SIZE:
                break
            if can_add(S, x, pair_sums):
                add_element(S, x, pair_sums)
        return S, pair_sums

    def local_search(S, pair_sums):
        S = set(S)
        pair_sums = dict(pair_sums) # Deep copy not strictly needed if we rebuild, but let's be safe
        # Rebuild pair_sums to ensure consistency
        pair_sums = {}
        for a in S:
            if a == 1:
                continue
            pair_sums[a] = set()
            S_list = list(S - {a})
            n = len(S_list)
            for i in range(n):
                b = S_list[i]
                for j in range(i + 1, n):
                    c = S_list[j]
                    pair_sums[a].add((b + c) % a)

        unselected = set(range(1, N + 1)) - S
        improved = True
        
        while improved and time_left() > 5:
            improved = False
            
            # Try to add
            for x in list(unselected):
                if len(S) >= MAX_SIZE:
                    break
                if can_add(S, x, pair_sums):
                    add_element(S, x, pair_sums)
                    unselected.discard(x)
                    improved = True
            
            # Try swaps
            if len(S) < MAX_SIZE:
                for a in list(S):
                    if time_left() < 5:
                        break
                    # Remove a
                    remove_element(S, a, pair_sums)
                    unselected.add(a)
                    
                    # Try to add as many as possible
                    added = []
                    for x in list(unselected):
                        if len(S) >= MAX_SIZE:
                            break
                        if can_add(S, x, pair_sums):
                            add_element(S, x, pair_sums)
                            unselected.discard(x)
                            added.append(x)
                    
                    if len(added) > 1:
                        improved = True
                    elif len(added) == 1:
                        # Swapped 1 for 1. Net zero.
                        # Check if we can add more?
                        # If not, revert to keep original set (or keep if we want to explore?)
                        # Usually, keeping the original is safer if no gain.
                        # But maybe the new element allows further additions?
                        # We already tried to add more in the loop above.
                        # So if only 1 added, and no more can be added, it's a neutral swap.
                        # We can revert to avoid oscillation.
                        for x in added:
                            remove_element(S, x, pair_sums)
                            unselected.add(x)
                        add_element(S, a, pair_sums)
                        unselected.discard(a)
                    else:
                        # No added. Revert.
                        add_element(S, a, pair_sums)
                        unselected.discard(a)

        return S, pair_sums

    random.seed(42)
    best_S = set()
    best_ps = {}

    # Strategy 1: Descending
    S, ps = greedy_build(range(N, 0, -1))
    if len(S) > len(best_S):
        best_S, best_ps = S, ps

    # Strategy 2: Ascending
    S, ps = greedy_build(range(1, N + 1))
    if len(S) > len(best_S):
        best_S, best_ps = S, ps

    # Strategy 3: Random restarts
    for _ in range(100):
        if time_left() < 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, ps = greedy_build(order)
        if len(S) > len(best_S):
            best_S, best_ps = S, ps

    # Local search on best
    if time_left() > 10:
        best_S, best_ps = local_search(best_S, best_ps)

    # More random restarts with local search
    for _ in range(50):
        if time_left() < 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, ps = greedy_build(order)
        S, ps = local_search(S, ps)
        if len(S) > len(best_S):
            best_S, best_ps = S, ps

    # Final validation
    def is_valid(S):
        S_list = list(S)
        n = len(S_list)
        for i, a in enumerate(S_list):
            if a == 1:
                if n >= 4:
                    return False
                continue
            others = [S_list[j] for j in range(n) if j != i]
            if len(others) < 3:
                continue
            for b, c, d in itertools.combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    import itertools # Need to import this for is_valid
    
    if not is_valid(best_S):
        # Fallback
        best_S = set(range(100, 148))
        if not is_valid(best_S):
            best_S = set()
            for x in range(N, 0, -1):
                if len(best_S) >= MAX_SIZE:
                    break
                # Simple check without pair_sums for fallback
                S_list = list(best_S)
                valid = True
                if len(S_list) >= 3:
                    for i in range(len(S_list)):
                        for j in range(i+1, len(S_list)):
                            for k in range(j+1, len(S_list)):
                                if (S_list[i]+S_list[j]+S_list[k]) % x == 0:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                if valid:
                    for a in S_list:
                        if a == 1:
                            if len(S_list) >= 3:
                                valid = False
                                break
                            continue
                        for i in range(len(S_list)):
                            if S_list[i] == a: continue
                            for j in range(i+1, len(S_list)):
                                if S_list[j] == a: continue
                                if (x + S_list[i] + S_list[j]) % a == 0:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                if valid:
                    best_S.add(x)

    return sorted(best_S)
