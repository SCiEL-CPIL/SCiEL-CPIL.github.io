def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c, all in [1, N]
    # where 1/a = 1/b + 1/c, i.e., b*c = a*(b+c)
    forbidden_pairs = {}  # (b, c) -> set of a values, and also track conflicts
    
    # For each pair (b, c) with 1 <= b < c <= N, compute a = b*c/(b+c)
    # If a is integer and a != b and a != c and 1 <= a <= N, then (a,b,c) is forbidden
    # We need to avoid having all three in our set
    
    # Build conflict structure: for each element x, which other elements conflict with it
    # Two elements x, y conflict if there exists z such that {x,y,z} is a forbidden triple
    # But actually, we need to avoid any triple, so the constraint is on triples not pairs.
    
    # Better approach: for each element x, precompute the set of pairs (b,c) from [1,N]\{x}
    # such that {x,b,c} is a forbidden triple. Then x cannot coexist with both b and c.
    
    # Let's precompute for each x, the list of "forbidden pair constraints"
    # constraint[x] = list of (b, c) such that {x, b, c} is a forbidden triple
    
    constraint = [[] for _ in range(N + 1)]
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    # (a, b, c) is a forbidden triple with a < b < c (since a = bc/(b+c) < min(b,c))
                    # Actually a could be > b or > c? Let's check: a = bc/(b+c)
                    # If b < c, then a = bc/(b+c) < bc/c = b, so a < b < c
                    # So a is always the smallest
                    for elem in [a, b, c]:
                        # This triple involves elements a, b, c
                        # For each element, record that it can't be with the other two together
                        pass
                    # Record: if we have a in our set, we can't have both b and c
                    # If we have b in our set, we can't have both a and c
                    # If we have c in our set, we can't have both a and b
                    constraint[a].append((b, c))
                    constraint[b].append((a, c))
                    constraint[c].append((a, b))
    
    # Start with the base set [91, 180]
    base_set = set(range(91, N + 1))
    current_set = set(base_set)
    best_set = set(base_set)
    best_size = len(base_set)
    
    def is_valid(s):
        """Check if set s has no forbidden triple."""
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = s_list[i], s_list[j], s_list[k]
                    # Check if 1/a = 1/b + 1/c or any permutation
                    # Since all positive, the smallest must be a
                    mn = min(a, b, c)
                    mx1, mx2 = sorted([x for x in (a, b, c) if x != mn])
                    if mn * mx1 * mx2 == mx1 * mx2 * (mx1 + mx2) // (mx1 + mx2) * (mx1 + mx2):
                        # More directly: check if mx1*mx2 = mn*(mx1+mx2)
                        if mx1 * mx2 == mn * (mx1 + mx2):
                            return False
        return True
    
    # Greedily try to add elements from [1, 90]
    # Sort candidates by how many constraints they have with current set
    candidates = list(range(1, 91))
    
    # For each candidate x, count how many pairs (b,c) in constraint[x] have both b,c in current_set
    def count_conflicts(x, s):
        count = 0
        for (b, c) in constraint[x]:
            if b in s and c in s:
                count += 1
        return count
    
    candidates.sort(key=lambda x: count_conflicts(x, current_set))
    
    for x in candidates:
        if time.time() - start_time > DEADLINE:
            break
        # Try adding x
        if x not in current_set:
            # Check if adding x creates any forbidden triple
            valid = True
            for (b, c) in constraint[x]:
                if b in current_set and c in current_set:
                    valid = False
                    break
            if valid:
                current_set.add(x)
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
    
    # Local search: try swapping one element out for another
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        # Try removing each element and see if we can add more
        for remove_elem in list(current_set):
            if time.time() - start_time > DEADLINE:
                break
            trial_set = current_set - {remove_elem}
            # Try adding each candidate not in trial_set
            for add_elem in range(1, N + 1):
                if add_elem in trial_set:
                    continue
                # Check if adding add_elem to trial_set is valid
                valid = True
                for (b, c) in constraint[add_elem]:
                    if b in trial_set and c in trial_set:
                        valid = False
                        break
                if valid:
                    new_set = trial_set | {add_elem}
                    if len(new_set) > len(current_set):
                        current_set = new_set
                        improved = True
                        if len(current_set) > best_size:
                            best_set = set(current_set)
                            best_size = len(current_set)
                        break
            # Also try just removing if it allows adding multiple
            # Skip for now, single swap is enough
    
    # Final local search: try adding any remaining element
    for add_elem in range(1, N + 1):
        if time.time() - start_time > DEADLINE:
            break
        if add_elem in current_set:
            continue
        valid = True
        for (b, c) in constraint[add_elem]:
            if b in current_set and c in current_set:
                valid = False
                break
        if valid:
            current_set.add(add_elem)
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
    
    return sorted(best_set)
