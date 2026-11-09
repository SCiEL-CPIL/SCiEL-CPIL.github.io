import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 80
    
    # Precompute conflict masks using 128-bit integers
    # For each element x in 1..128, precompute which other elements conflict with it
    # in the sense that there exists a third element z such that {x, y, z} forms a forbidden triple.
    # 
    # A forbidden triple is {a, b, c} where a+b, a+c, b+c are all in the set.
    # 
    # Key insight: if we have elements a and b selected, then for any c to be added,
    # we need a+c and b+c NOT both in the set (or one of them exceeds n).
    # 
    # Better approach: precompute for each pair (a,b) where a+b <= n, the set of c values
    # that would form a forbidden triple with a and b. c is forbidden if a+c <= n, b+c <= n,
    # and a+c and b+c are both in the set.
    #
    # Even better: use the structural insight that large elements are "safe" among themselves.
    # If all elements are >= 65, then any pair sum exceeds 128, so no pair sum can be in the set.
    # Thus any subset of {65, ..., 128} is valid.
    #
    # For small elements, we need careful checking.
    #
    # Let's use a different strategy:
    # 1. Start with all elements >= 65 (size 64).
    # 2. Try to add small elements one by one, checking validity efficiently.
    # 3. Use local search to improve.
    
    # Precompute: for each element x, and for each potential "partner" y in the set,
    # determine which elements c would create a forbidden triple with x and y.
    #
    # A triple {a, b, c} is forbidden iff a+b, a+c, and b+c are all in the set.
    # So if a and b are in the set, c is forbidden iff:
    #   a+b is in the set (must check)
    #   a+c is in the set
    #   b+c is in the set
    #
    # Let's precompute a 2D array: conflict[x][y] = bitmask of c values that would be
    # forbidden if both x and y are in the set.
    #
    # Actually, let's think differently. For a given set S, when we want to add element z,
    # we need to check: for all pairs (a, b) in S with a < b, is {a, b, z} a forbidden triple?
    # That is, are a+b, a+z, and b+z all in S ∪ {z}?
    # Since z is not yet in S, a+z and b+z are not in S unless they equal z (impossible for positive a,b).
    # Wait, a+z could be in S if a+z <= 128 and a+z is already selected.
    #
    # So z can be added to S iff for all pairs (a, b) in S:
    #   NOT (a+b in S AND a+z in S AND b+z in S)
    #
    # This is still O(|S|^2) per addition.
    #
    # Let's use the bit-parallel approach more cleverly:
    # For each z, precompute which elements x have x+z <= n and x+z "could be in S".
    # Actually, let's precompute for each z, the set of pairs (a,b) such that a+z and b+z are both <= n.
    # Then z conflicts with pair (a,b) iff a+b, a+z, b+z are all in S.
    #
    # New plan: use a boolean array for membership, and for each candidate z,
    # iterate over pairs in S but only those where both a+z and b+z <= n.
    # For each such pair, check if a+b, a+z, b+z are all present.
    #
    # Even better: for each z, precompute the list of x such that x+z <= n.
    # Then for candidate z, the relevant pairs (a,b) in S are those where both a and b are in this list.
    # 
    # Let me just implement a clean version with the structural insight and greedy + local search.
    
    def can_add(S_bool, S_list, z):
        """Check if z can be added to the set represented by S_bool/S_list."""
        n_local = n
        # For each pair (a, b) in S_list, check if adding z creates a forbidden triple
        # with any third element c in S_list.
        # Actually, the forbidden triple must include z, so we check pairs (a, b) in S_list
        # and see if {a, b, z} is forbidden, i.e., a+b, a+z, b+z are all in S ∪ {z}.
        # Since z not in S, a+z in S ∪ {z} means a+z in S (a+z != z for a>0).
        # Similarly b+z in S.
        # And a+b must be in S.
        
        for i in range(len(S_list)):
            a = S_list[i]
            a_plus_z = a + z
            if a_plus_z > n_local:
                continue
            if not S_bool[a_plus_z]:
                continue
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                b_plus_z = b + z
                if b_plus_z > n_local:
                    continue
                if not S_bool[b_plus_z]:
                    continue
                # Check if a+b is in S
                ab = a + b
                if ab <= n_local and S_bool[ab]:
                    return False
        return True
    
    # Initial set: all elements from 65 to 128
    S_bool = [False] * (n + 1)
    S_list = list(range(65, n + 1))
    for x in S_list:
        S_bool[x] = True
    
    best_set = S_list.copy()
    best_size = len(S_list)
    
    # Candidates: 1 to 64
    candidates = list(range(1, 65))
    
    # Greedy: try to add candidates
    for _ in range(50):  # Multiple passes
        random.shuffle(candidates)
        improved = False
        for z in candidates:
            if S_bool[z]:
                continue
            if can_add(S_bool, S_list, z):
                S_bool[z] = True
                S_list.append(z)
                S_list.sort()
                improved = True
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_set = S_list.copy()
                break
        if not improved:
            break
    
    # Local search: try swaps
    all_elements = list(range(1, n + 1))
    
    for iteration in range(10000):
        if time.time() > deadline - 5:
            break
        
        improved = False
        
        # Try 1-for-1 swaps: remove x, add y
        removed_list = S_list.copy()
        random.shuffle(removed_list)
        added_list = [x for x in all_elements if not S_bool[x]]
        random.shuffle(added_list)
        
        found_swap = False
        for x in removed_list:
            if time.time() > deadline - 5:
                break
            # Temporarily remove x
            S_bool[x] = False
            S_list.remove(x)
            
            for y in added_list:
                if can_add(S_bool, S_list, y):
                    # Add y
                    S_bool[y] = True
                    S_list.append(y)
                    S_list.sort()
                    improved = True
                    found_swap = True
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_set = S_list.copy()
                    break
            
            # Restore x
            S_bool[x] = True
            S_list.append(x)
            S_list.sort()
            
            if found_swap:
                break
        
        if not improved:
            # Try 1-for-2: remove x, add y and z
            removed_list2 = S_list.copy()
            random.shuffle(removed_list2)
            not_in_set = [x for x in all_elements if not S_bool[x]]
            random.shuffle(not_in_set)
            
            found_double = False
            for x in removed_list2:
                if time.time() > deadline - 3:
                    break
                S_bool[x] = False
                S_list.remove(x)
                
                trial_list = S_list.copy()
                trial_bool = S_bool.copy()
                
                # Try adding two elements
                for y in not_in_set:
                    if y == x:
                        continue
                    if not can_add(trial_bool, trial_list, y):
                        continue
                    trial_bool[y] = True
                    trial_list.append(y)
                    trial_list.sort()
                    
                    for z2 in not_in_set:
                        if z2 == x or z2 == y:
                            continue
                        if not can_add(trial_bool, trial_list, z2):
                            continue
                        # Accept
                        S_bool = trial_bool
                        S_list = trial_list
                        improved = True
                        found_double = True
                        if len(S_list) > best_size:
                            best_size = len(S_list)
                            best_set = S_list.copy()
                        break
                    
                    if found_double:
                        break
                
                if found_double:
                    break
                else:
                    # Restore
                    S_bool[x] = True
                    S_list.append(x)
                    S_list.sort()
            
            if found_double:
                continue
        
        if not improved:
            # Stagnation: random perturbation
            if random.random() < 0.3:
                # Remove a random element
                x = random.choice(S_list)
                S_bool[x] = False
                S_list.remove(x)
    
    # Final verification
    def is_valid(final_set):
        if len(final_set) < 3:
            return True
        present = [False] * (n + 1)
        for x in final_set:
            present[x] = True
        fs = list(final_set)
        for i in range(len(fs)):
            a = fs[i]
            for j in range(i + 1, len(fs)):
                b = fs[j]
                ab = a + b
                if ab > n or not present[ab]:
                    continue
                for k in range(len(fs)):
                    if k == i or k == j:
                        continue
                    c = fs[k]
                    ac = a + c
                    bc = b + c
                    if ac <= n and bc <= n and present[ac] and present[bc]:
                        return False
        return True
    
    if is_valid(best_set):
        return list(best_set)
    else:
        return list(range(65, n + 1))
