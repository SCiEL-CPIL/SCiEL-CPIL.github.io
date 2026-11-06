def construct():
    import itertools
    import time
    
    # Try cyclic construction for n=34 down to n=30
    for n in [34, 33, 32, 31, 30]:
        # Variables: v[i][j] for 0 < i < j < n
        # Map (i,j) to a flat index
        vars_map = {}
        var_list = []
        idx = 0
        for i in range(1, n):
            for j in range(i+1, n):
                vars_map[(i,j)] = idx
                var_list.append((i,j))
                idx += 1
        
        num_vars = len(var_list)
        # colors[i] is the color of the base triple {0, i, j}
        # We need to map any triple {a,b,c} to a base variable
        def get_base_var(a, b, c):
            # Shift so min is 0
            min_v = min(a,b,c)
            a2 = (a - min_v) % n
            b2 = (b - min_v) % n
            c2 = (c - min_v) % n
            # The set is {0, u, v}
            # u, v are non-zero
            # Sort u, v
            if a2 == 0:
                u, v = b2, c2
            elif b2 == 0:
                u, v = a2, c2
            else:
                u, v = a2, b2
            
            if u > v:
                u, v = v, u
            
            # u, v are in 1..n-1
            return vars_map[(u,v)]
        
        # Precompute for each variable, which quads and quints it belongs to?
        # This is hard because one variable defines many actual triples.
        # Instead, let's just check the whole graph after each assignment.
        # Optimization: Only check quads/quints that contain at least one "newly assigned" base triple?
        # No, a single base triple assignment affects many actual triples.
        
        # Let's use a simpler check:
        # We maintain an array `base_colors` of length num_vars, -1 if unassigned.
        
        def is_valid_partial():
            # Check for violations in the determined part of the graph
            # A quad is a violation if all 4 triples are assigned and all are 0 (red)
            # A quint is a violation if all 10 triples are assigned and all are 1 (blue)
            
            # To make this fast, we can precompute for each quad the 4 base variables it depends on.
            # If any of the 4 base variables is unassigned, the quad is not yet determined.
            # If all 4 are assigned, check if all are 0.
            
            pass # Placeholder
            
        # This approach is getting complex. Let's try a different tactic.
        # Use the seed program's local search but with a better initialization.
        # The seed program uses random initialization.
        # A better initialization for Ramsey problems is often a "random" one, 
        # but we can try to initialize with a pattern.
        
        # Let's stick to the Cyclic Backtracking but implement it carefully.
        # We will precompute the list of quads and quints and for each, the 4 or 10 base variables.
        
        quads = list(itertools.combinations(range(n), 4))
        quints = list(itertools.combinations(range(n), 5))
        
        quad_base_vars = []
        for q in quads:
            a,b,c,d = q
            # The 4 triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            vars_in_quad = set()
            for t in [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]:
                vars_in_quad.add(get_base_var(*t))
            quad_base_vars.append(list(vars_in_quad))
            
        quint_base_vars = []
        for q in quints:
            vars_in_quint = set()
            for t in itertools.combinations(q, 3):
                vars_in_quint.add(get_base_var(*t))
            quint_base_vars.append(list(vars_in_quint))
            
        # Now, backtracking
        # We need to order the variables.
        # Let's use a simple order: by index in var_list.
        
        # To speed up, we can use a time limit.
        start_time = time.time()
        time_limit = 10 # seconds
        
        base_colors = [-1] * num_vars
        
        # Forward checking:
        # For each quad, we can track the number of red edges assigned.
        # If it reaches 4, violation.
        # For each quint, track number of blue edges assigned.
        # If it reaches 10, violation.
        
        red_counts = [0] * len(quads)
        blue_counts = [0] * len(quints)
        
        # When we assign a variable, we update the counts for all quads/quints it belongs to.
        # Precompute variable_to_quads and variable_to_quints
        var_to_quads = [[] for _ in range(num_vars)]
        var_to_quints = [[] for _ in range(num_vars)]
        
        for qi, vars_in_quad in enumerate(quad_base_vars):
            for v in vars_in_quad:
                var_to_quads[v].append(qi)
                
        for qi, vars_in_quint in enumerate(quint_base_vars):
            for v in vars_in_quint:
                var_to_quints[v].append(qi)
                
        # Note: A variable might appear multiple times in a quad's variable set?
        # No, we used a set. But different triples in the quad might map to the SAME base variable.
        # If so, the count logic is tricky.
        # Example: Quad {0, 1, 2, 3}.
        # Triples: {0,1,2} -> base(1,2). {0,1,3} -> base(1,3). {0,2,3} -> base(2,3). {1,2,3} -> base(1,2) (shifted by 1? No, min is 1, so {0,1,2} relative to 1 is {0,1,2} -> base(1,2)).
        # So {0,1,2} and {1,2,3} map to the same base variable!
        # This means the color of {0,1,2} and {1,2,3} is the SAME.
        # So if base(1,2) is Red, both are Red.
        # So in the quad {0,1,2,3}, the number of red edges depends on the value of base(1,2).
        # If base(1,2)=0, then 2 edges are red.
        # If base(1,2)=1, then 0 edges are red (from this variable).
        
        # So we cannot just increment a count. We need to know the multiplicity.
        # Let's store for each quad, a list of (var_id, multiplicity).
        
        quad_var_mults = []
        for q in quads:
            a,b,c,d = q
            counts = {}
            for t in [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]:
                v = get_base_var(*t)
                counts[v] = counts.get(v, 0) + 1
            quad_var_mults.append(counts)
            
        quint_var_mults = []
        for q in quints:
            counts = {}
            for t in itertools.combinations(q, 3):
                v = get_base_var(*t)
                counts[v] = counts.get(v, 0) + 1
            quint_var_mults.append(counts)
            
        # Now, when we assign var v to color c:
        # For each quad containing v:
        #   If c == 0 (Red): red_count[quad] += multiplicity
        #   If c == 1 (Blue): red_count[quad] += 0 (but blue_count increases)
        # Wait, we need to track both.
        # Actually, we just need to check if red_count == 4 or blue_count == 10.
        
        # Let's restart the counts logic.
        # red_count[quad] = sum of multiplicities of red variables in the quad.
        # blue_count[quad] = sum of multiplicities of blue variables in the quad.
        # Note: red_count + blue_count = total edges in quad (4) only if all vars are assigned.
        # If some vars are unassigned, red_count + blue_count < 4.
        
        # Violation if red_count == 4 OR blue_count == 10 (for quints).
        
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            
            if idx == num_vars:
                return True
                
            v_id = idx
            # Try 0 then 1
            for c in [0, 1]:
                base_colors[v_id] = c
                
                # Update counts
                valid = True
                # Update quads
                for qi in var_to_quads[v_id]:
                    mult = quad_var_mults[qi][v_id]
                    if c == 0:
                        red_counts[qi] += mult
                        if red_counts[qi] == 4:
                            valid = False
                            break
                    else:
                        # blue
                        # We don't track blue_count for quads, only red.
                        # But we need to check blue for quints.
                        pass
                        
                if not valid:
                    # Rollback
                    for qi in var_to_quads[v_id]:
                        mult = quad_var_mults[qi][v_id]
                        if c == 0:
                            red_counts[qi] -= mult
                    continue # Try next color
                
                # Update quints
                for qi in var_to_quints[v_id]:
                    mult = quint_var_mults[qi][v_id]
                    if c == 1:
                        blue_counts[qi] += mult
                        if blue_counts[qi] == 10:
                            valid = False
                            break
                    else:
                        pass
                        
                if not valid:
                    # Rollback quints
                    for qi in var_to_quints[v_id]:
                        mult = quint_var_mults[qi][v_id]
                        if c == 1:
                            blue_counts[qi] -= mult
                    # Rollback quads (if we broke out of quint loop, quad loop finished)
                    # Wait, if valid was False in quad loop, we continued.
                    # If valid was False in quint loop, we need to rollback quads too?
                    # No, the quad rollback was done in the 'if not valid' block for quads.
                    # But if we passed quads and failed quints, we need to rollback quads?
                    # No, the quads were updated and not rolled back in the quint failure path.
                    # We need a clean rollback mechanism.
                    
                    # Let's restructure.
                    
                    # Rollback quads
                    for qi in var_to_quads[v_id]:
                        mult = quad_var_mults[qi][v_id]
                        if c == 0:
                            red_counts[qi] -= mult
                    continue # Try next color
                
                # If both passed, recurse
                if backtrack(idx + 1):
                    return True
                
                # Rollback quads
                for qi in var_to_quads[v_id]:
                    mult = quad_var_mults[qi][v_id]
                    if c == 0:
                        red_counts[qi] -= mult
                # Rollback quints
                for qi in var_to_quints[v_id]:
                    mult = quint_var_mults[qi][v_id]
                    if c == 1:
                        blue_counts[qi] -= mult
                        
            return False
            
        if backtrack(0):
            # Construct the full coloring
            colors = []
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        v_id = get_base_var(a,b,c)
                        colors.append(base_colors[v_id])
            return {'n': n, 'colors': colors}
            
    # Fallback: return a small valid instance
    # For n=9, use the pattern from seed
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
