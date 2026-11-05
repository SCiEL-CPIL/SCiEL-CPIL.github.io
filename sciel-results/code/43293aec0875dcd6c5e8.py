"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def solve_sum_free(vals, seed=42, time_limit=30.0):
        n = len(vals)
        if n == 0:
            return []
        
        val_to_idx = {v: i for i, v in enumerate(vals)}
        constraints = []
        seen_pairs = set()
        # Pre-compute constraints
        for i in range(n):
            for j in range(i, n):
                s = vals[i] + vals[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Ensure distinct indices for the triple logic? 
                    # The constraint is "not (x_i and x_j and x_k)".
                    # If i==j, it's 2*a=b. If i==k, a+b=a => b=0, but 0 is not in set.
                    # So i, j, k are likely distinct unless 0 is involved, but 0 is excluded.
                    # However, if a+b=c and a=c, then b=0. Excluded.
                    # If a=b, 2a=c. Distinct values.
                    key = tuple(sorted((i, j, k)))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        constraints.append((i, j, k))
        
        num_constraints = len(constraints)
        
        # Pre-compute variable to constraints map
        var_to_cons = [[] for _ in range(n)]
        for idx, (i, j, k) in enumerate(constraints):
            var_to_cons[i].append(idx)
            var_to_cons[j].append(idx)
            var_to_cons[k].append(idx)
            
        # Initial solution: All 1s (maximize size, then repair)
        sol = [1] * n
        
        # State tracking
        # cons_viol[c] = 1 if constraint c is violated (all 3 vars are 1), else 0
        cons_viol = [0] * num_constraints
        # var_viol_count[v] = number of violated constraints involving v
        var_viol_count = [0] * n
        active_violations = 0
        current_size = n
        
        # Initialize state
        for c_idx, (i, j, k) in enumerate(constraints):
            if sol[i] and sol[j] and sol[k]:
                cons_viol[c_idx] = 1
                active_violations += 1
                var_viol_count[i] += 1
                var_viol_count[j] += 1
                var_viol_count[k] += 1
        
        best_sol = sol[:]
        best_size = n
        
        rng = random.Random(seed)
        start_time = time.time()
        
        # Heuristic for flipping: 
        # If we flip a var v in a violated constraint c:
        # - If v is 1 and we set to 0: c becomes satisfied. 
        #   Other constraints involving v might become satisfied (if they were violated and v was 1) 
        #   or stay violated. They cannot become violated because we are removing a 1.
        #   Wait, if we remove a 1, a constraint can only go from Violated to Satisfied.
        #   So flipping 1->0 always decreases active_violations by at least 1 (for the chosen constraint)
        #   and possibly more (for other constraints involving v).
        # - If v is 0 and we set to 1: c remains violated (or becomes violated if it wasn't, but we pick violated ones).
        #   Wait, WalkSAT usually picks a violated clause and flips a literal.
        #   Here, a "clause" is NOT (x_i AND x_j AND x_k). It is violated if x_i=x_j=x_k=1.
        #   To satisfy it, we must set at least one to 0.
        #   So we should only flip 1->0 when we have violations.
        #   When we have NO violations, we want to flip 0->1 to increase size.
        
        # Strategy:
        # 1. If active_violations > 0:
        #    Pick a random violated constraint c.
        #    Pick a variable v in c that is 1.
        #    Flip v to 0.
        #    Update state.
        # 2. If active_violations == 0:
        #    Pick a random variable v that is 0.
        #    Check if flipping v to 1 creates any violations.
        #    If not, flip to 1.
        #    If yes, we might need to restart or do a more complex move.
        #    To avoid getting stuck, if we can't add, we can randomly flip a 1->0 to "unlock" space, 
        #    but this reduces size. We should only do this if we've been stuck for a while.
        #    Alternatively, use a "Simulated Annealing" like acceptance or just restart.
        #    Given the time limit, let's try a simple strategy:
        #    If no add possible, flip a random 1->0 to break a potential future blockage? 
        #    Actually, if no add is possible, it means every 0-variable is part of a "would-be" violation 
        #    with two existing 1s. 
        #    Let's just try to add. If we can't, we pick a random 1 to remove (sacrifice size to allow growth later).
        
        stuck_count = 0
        
        for _ in range(1000000): # Safety cap on iterations
            if time.time() - start_time > time_limit:
                break
                
            if active_violations > 0:
                # Find a violated constraint
                # We can pick random index and check, or maintain a list. 
                # Picking random index is O(1) average if many violations.
                # Let's just pick a random constraint index and check.
                # To be efficient, we can maintain a list of violated constraint indices.
                # But updating that list is complex. 
                # For n=240, num_constraints is small (< 5000). 
                # Let's just iterate to find one? No, that's O(C).
                # Let's maintain `violated_cons` list? 
                # Or just pick random c_idx until we find a violated one.
                
                c_idx = rng.randrange(num_constraints)
                # If we pick a non-violated one, we wasted time. 
                # Better: Keep a list of violated constraint indices.
                # Let's refactor to maintain a set/list of violated indices for O(1) selection.
                
                # Since we can't easily refactor the loop structure without rewriting, 
                # let's just assume num_constraints is small enough that finding one is fast, 
                # or we can just pick a random variable from the whole set that has var_viol_count > 0.
                
                # Pick a variable with var_viol_count > 0
                # We can build a list of such variables occasionally, or just pick random var.
                # Let's pick a random variable v. If var_viol_count[v] > 0, flip it.
                # This might be slow if few variables are involved.
                
                # Better: Maintain a list `active_vars` of indices where var_viol_count > 0.
                # Updating this list is tricky.
                
                # Let's stick to: Pick random c_idx. If violated, process. Else retry.
                # If we have many violations, this is fast. If few, it's slow.
                # Given we start with all 1s, there are MANY violations initially.
                
                if cons_viol[c_idx]:
                    i, j, k = constraints[c_idx]
                    # Choose which one to flip. 
                    # Heuristic: Flip the one with the highest var_viol_count? 
                    # Or random? Random is WalkSAT.
                    # Let's try: Flip the one that is "least useful" (lowest degree in graph of constraints? No, highest violation count means it's causing more trouble).
                    # Actually, removing a variable with high var_viol_count fixes more constraints.
                    # So we should prefer flipping the variable with the HIGHEST var_viol_count among i, j, k.
                    
                    candidates = [i, j, k]
                    # Filter to those that are 1
                    candidates = [v for v in candidates if sol[v] == 1]
                    if not candidates:
                        continue # Should not happen if cons_viol is 1
                    
                    # Pick the one with max var_viol_count
                    v_to_flip = max(candidates, key=lambda x: var_viol_count[x])
                    
                    # Perform Flip 1 -> 0
                    sol[v_to_flip] = 0
                    current_size -= 1
                    stuck_count = 0
                    
                    # Update state
                    for c in var_to_cons[v_to_flip]:
                        ci, cj, ck = constraints[c]
                        # Was it violated?
                        was_violated = cons_viol[c]
                        # Now one of its vars is 0, so it CANNOT be violated.
                        if was_violated:
                            cons_viol[c] = 0
                            active_violations -= 1
                            # Update var_viol_count for the other two vars
                            for v in (ci, cj, ck):
                                if v != v_to_flip:
                                    var_viol_count[v] -= 1
                        # If it wasn't violated, it stays not violated.
                    
                    # Update var_viol_count for v_to_flip
                    # It was contributing to `var_viol_count[v_to_flip]` violations.
                    # Now it's 0, so it contributes to 0 violations.
                    # We already decremented the others. We need to reset v_to_flip's count.
                    # Wait, the loop above decremented the *other* vars.
                    # We need to set var_viol_count[v_to_flip] to 0?
                    # No, var_viol_count[v] is the number of violated constraints *involving* v.
                    # If v is 0, it can't be part of a violated constraint (which requires 1,1,1).
                    # So var_viol_count[v_to_flip] should become 0.
                    # But we didn't decrement it in the loop above (we skipped v != v_to_flip).
                    # So we must explicitly set it.
                    var_viol_count[v_to_flip] = 0
                    
            else:
                # No violations. Try to add.
                # Find a variable v with sol[v]==0 and var_viol_count[v]==0.
                # If such v exists, adding it creates NO new violations.
                # Why? A new violation involving v would require v=1 and two other vars u, w=1 such that u+w=v or u+v=w or v+w=u.
                # If var_viol_count[v] was 0, it means there are NO currently violated constraints involving v.
                # But wait. var_viol_count tracks *currently* violated constraints.
                # If we add v, we might create a violation with two *other* 1s.
                # Example: sol[u]=1, sol[w]=1, u+w=v. Constraint (u,w,v).
                # Currently, if sol[v]=0, this constraint is NOT violated. So it doesn't count in var_viol_count[u], [w], or [v].
                # So var_viol_count[v]==0 does NOT guarantee we can add v.
                
                # We need to check if adding v creates a violation.
                # A violation is created if there exists a constraint (a,b,c) involving v
                # such that the other two are 1.
                
                # To do this fast, we can just pick a random 0-variable and check.
                # If we find one that is safe, add it.
                # If we check many and none are safe, we are stuck.
                
                # Let's pick a random index v where sol[v]==0.
                # If none, break.
                zeros = [i for i in range(n) if sol[i] == 0]
                if not zeros:
                    break # All 1s and no violations? Perfect.
                
                v = rng.choice(zeros)
                
                # Check if adding v is safe
                safe = True
                for c in var_to_cons[v]:
                    ci, cj, ck = constraints[c]
                    # Check the other two
                    others = [x for x in (ci, cj, ck) if x != v]
                    if sol[others[0]] == 1 and sol[others[1]] == 1:
                        safe = False
                        break
                
                if safe:
                    sol[v] = 1
                    current_size += 1
                    stuck_count = 0
                    # Update state: 
                    # Adding v (0->1) can only create violations, not resolve them.
                    # So we check which constraints become violated.
                    for c in var_to_cons[v]:
                        ci, cj, ck = constraints[c]
                        if cons_viol[c] == 0: # It wasn't violated before
                            others = [x for x in (ci, cj, ck) if x != v]
                            if sol[others[0]] == 1 and sol[others[1]] == 1:
                                # Now it IS violated
                                cons_viol[c] = 1
                                active_violations += 1
                                # Update var_viol_count for all 3 vars
                                for x in (ci, cj, ck):
                                    var_viol_count[x] += 1
                else:
                    # Cannot add v. 
                    # Increment stuck counter.
                    stuck_count += 1
                    if stuck_count > 1000:
                        # Force a move: flip a random 1 to 0 to potentially allow growth
                        ones = [i for i in range(n) if sol[i] == 1]
                        if ones:
                            v = rng.choice(ones)
                            sol[v] = 0
                            current_size -= 1
                            # Update state for 1->0
                            for c in var_to_cons[v]:
                                ci, cj, ck = constraints[c]
                                if cons_viol[c]:
                                    cons_viol[c] = 0
                                    active_violations -= 1
                                    for x in (ci, cj, ck):
                                        if x != v:
                                            var_viol_count[x] -= 1
                            var_viol_count[v] = 0
                            stuck_count = 0
                            # This is a "kick" to escape local optimum
                            
            # Update best
            if current_size > best_size:
                best_size = current_size
                best_sol = sol[:]
                
        return [vals[i] for i in range(n) if best_sol[i]]
    
    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        subset = solve_sum_free(vals, seed=seed)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
