"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    Uses a greedy heuristic: iteratively select the vertex (number) that
    blocks the most other candidates, thereby minimizing the final set size
    while ensuring maximality.
    """
    selected = []
    sums = set()
    available = set(range(1, N + 1))
    
    while available:
        best_v = None
        max_degree = -1
        
        # Evaluate each candidate in 'available'
        for v in available:
            # 1. Check if v is compatible with current 'selected'
            # New sums introduced by v: 2v and v+s for s in selected
            new_sums_v = {2 * v}
            for s in selected:
                new_sums_v.add(v + s)
            
            # If any new sum collides with existing sums, v is invalid
            if sums.intersection(new_sums_v):
                continue
            
            # v is a valid candidate to add.
            # 2. Calculate 'degree': how many other available elements u
            #    become incompatible if we add v?
            #    u is incompatible if (selected + [v]) + [u] is NOT Sidon.
            
            # Let S_temp = selected + [v]
            # Let sums_temp = sums | new_sums_v
            # u is blocked if:
            #   - 2u is in sums_temp
            #   - OR (u + s) is in sums_temp for any s in S_temp
            
            degree = 0
            # We iterate over available u != v
            for u in available:
                if u == v:
                    continue
                
                # Check if u is blocked by adding v
                # Condition 1: 2u in sums_temp
                if 2 * u in sums.union(new_sums_v):
                    degree += 1
                    continue
                
                # Condition 2: (u + s) in sums_temp for some s in selected + [v]
                blocked = False
                # Check against sums from selected
                for s in selected:
                    if (u + s) in sums:
                        blocked = True
                        break
                if not blocked:
                    # Check against sums from v
                    # s can be v or any s in selected
                    # u + v in sums_temp?
                    # u + s (where s in selected) in sums_temp?
                    # sums_temp contains 'sums' and 'new_sums_v'.
                    # We already checked u+s in sums above.
                    # Now check if u+s is in new_sums_v?
                    # new_sums_v = {2v} U {v+s_i}
                    # So we need to check if (u+s) == 2v or (u+s) == v+s_i
                    
                    # Simpler: just check if (u+s) is in the union set.
                    # But union creates a new set each time? No, we can just check.
                    
                    # Let's optimize the check for u compatibility with S_temp.
                    # We need to verify if any sum u+s_temp (s_temp in S_temp) is in sums_temp.
                    # S_temp = selected U {v}
                    # sums_temp = sums U new_sums_v
                    
                    # We already checked u+s (s in selected) against sums.
                    # We need to check:
                    # 1. u+s (s in selected) against new_sums_v
                    # 2. u+v against sums
                    # 3. u+v against new_sums_v (which is 2v? No, u+v vs 2v -> u=v, false. u+v vs v+s -> u=s, false)
                    
                    # Let's just do a direct check for clarity and correctness, 
                    # given the small size.
                    for s in selected:
                        if (u + s) in new_sums_v:
                            blocked = True
                            break
                    if not blocked and (u + v) in sums:
                        blocked = True
                    # u+v in new_sums_v?
                    # new_sums_v has 2v and v+s.
                    # u+v == 2v => u=v (False)
                    # u+v == v+s => u=s (False, u in available, s in selected, disjoint)
                    
                    if blocked:
                        degree += 1
            
            if degree > max_degree:
                max_degree = degree
                best_v = v
        
        if best_v is None:
            # No valid vertex, set is maximal
            break
        
        # Update state with best_v
        selected.append(best_v)
        
        # Update sums
        sums.add(2 * best_v)
        for s in selected[:-1]:
            sums.add(best_v + s)
        
        # Update available: remove best_v and any u incompatible with new selected
        new_available = set()
        for u in available:
            if u == best_v:
                continue
            
            # Check if u is compatible with new selected
            # New sums for u: 2u, u+s for s in selected
            # Collision if any in sums
            
            if 2 * u in sums:
                continue
            
            ok = True
            for s in selected:
                if (u + s) in sums:
                    ok = False
                    break
            
            if ok:
                new_available.add(u)
        
        available = new_available
        
    return selected
# EVOLVE-BLOCK-END
