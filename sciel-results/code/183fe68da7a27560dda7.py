import time

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute APs ending at each index x (1-indexed)
    aps_ending = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for d in range(1, (x - 1) // 3 + 1):
            i = x - 3 * d
            if i >= 1:
                j = i + d
                k = i + 2 * d
                aps_ending[x].append((i, j, k))
    
    def solve_with_deadline(k, dl):
        colors = [None] * (N + 1)
        colors[1] = 0
        
        # Forward checking: maintain domains for future positions
        # domain[x] is a bitmask of valid colors for position x
        # Initially, all colors are valid for all positions
        full_mask = (1 << k) - 1
        domain = [full_mask] * (N + 1)
        domain[1] = 1  # Only color 0 is valid for position 1
        
        def backtrack(x):
            if time.time() > dl:
                return None
            
            if x > N:
                return colors[1:].copy()
            
            # Try each color in the domain
            mask = domain[x]
            while mask:
                c = (mask & -mask).bit_length() - 1  # Least significant bit
                mask &= mask - 1  # Remove this bit
                
                # Check constraints for APs ending at x
                valid = True
                for (i, j, k_idx) in aps_ending[x]:
                    ci = colors[i]
                    cj = colors[j]
                    ck = colors[k_idx]
                    if len({ci, cj, ck, c}) < 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign color
                colors[x] = c
                old_domain = domain[x + 1:N + 1].copy() if x + 1 <= N else []
                
                # Forward checking: update domains for future positions
                fc_valid = True
                for y in range(x + 1, N + 1):
                    new_mask = full_mask
                    for (i, j, k_idx) in aps_ending[y]:
                        # Check if this AP is affected by the assignment at x
                        # The AP is (i, j, k_idx, y)
                        # If x is one of i, j, k_idx, then the constraint is partially satisfied
                        if x == i or x == j or x == k_idx:
                            # The other three positions must be assigned or will be assigned
                            # For now, just check if the constraint can be satisfied
                            pass
                    # Simplified: just check if the new color c conflicts with any assigned position in APs ending at y
                    for (i, j, k_idx) in aps_ending[y]:
                        assigned = []
                        if i == x:
                            assigned.append(c)
                        if j == x:
                            assigned.append(c)
                        if k_idx == x:
                            assigned.append(c)
                        # Check other assigned positions
                        if colors[i] is not None and i != x:
                            assigned.append(colors[i])
                        if colors[j] is not None and j != x:
                            assigned.append(colors[j])
                        if colors[k_idx] is not None and k_idx != x:
                            assigned.append(colors[k_idx])
                        if len(assigned) == 3:
                            # All three previous positions are assigned
                            # Check if any color for y would work
                            for cy in range(k):
                                if len({cy} | set(assigned)) >= 3:
                                    break
                            else:
                                # No valid color for y
                                fc_valid = False
                                break
                    if not fc_valid:
                        break
                
                if fc_valid:
                    res = backtrack(x + 1)
                    if res is not None:
                        return res
                
                # Undo
                colors[x] = None
                if x + 1 <= N:
                    domain[x + 1:N + 1] = old_domain
            
            return None
        
        return backtrack(2)
    
    # Try k=3
    sol = solve_with_deadline(3, start_time + 40.0)
    if sol:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]
    
    # Try k=4
    sol = solve_with_deadline(4, start_time + 80.0)
    if sol:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]
    
    # Fallback: return a valid solution with more colors
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
