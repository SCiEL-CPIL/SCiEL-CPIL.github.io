def construct():
    n = 128
    
    def solve_periodic(P, num_colors):
        # Precompute constraints
        constraints = []
        # Map variable index to list of constraint indices
        var_to_cons = {i: [] for i in range(P)}
        
        for a in range(1, n+1):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                i1 = (a - 1) % P
                i2 = (a + d - 1) % P
                i3 = (a + 2*d - 1) % P
                i4 = (a + 3*d - 1) % P
                cid = len(constraints)
                constraints.append((i1, i2, i3, i4))
                for idx in [i1, i2, i3, i4]:
                    if cid not in var_to_cons[idx]:
                        var_to_cons[idx].append(cid)
                        
        x = [-1] * P
        
        def is_valid(idx, val):
            x[idx] = val
            # Check constraints involving idx that are now fully assigned
            for cid in var_to_cons[idx]:
                i1, i2, i3, i4 = constraints[cid]
                if x[i1] != -1 and x[i2] != -1 and x[i3] != -1 and x[i4] != -1:
                    if len({x[i1], x[i2], x[i3], x[i4]}) < 3:
                        return False
            return True
            
        def backtrack(pos):
            if pos == P:
                return True
            # Try values
            for val in range(num_colors):
                if is_valid(pos, val):
                    if backtrack(pos + 1):
                        return True
            x[pos] = -1
            return False
            
        if backtrack(0):
            return [x[i % P] for i in range(n)]
        return None

    # Try periods > 42
    for P in [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]:
        for nc in [3, 4]:
            res = solve_periodic(P, nc)
            if res is not None:
                # Verify
                valid = True
                for a in range(1, n+1):
                    for d in range(1, (n-a)//3 + 1):
                        c = [res[a-1], res[a+d-1], res[a+2*d-1], res[a+3*d-1]]
                        if len(set(c)) < 3:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    return res
    # Fallback
    return [i % 3 for i in range(n)]
