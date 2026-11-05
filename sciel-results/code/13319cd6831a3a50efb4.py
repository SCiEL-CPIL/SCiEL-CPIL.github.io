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
        
        # Build constraints
        # cons_vars[c] = list of variable indices that cannot all be 1
        cons_vars = []
        var_cons = [[] for _ in range(n)]
        seen_constraints = set()
        
        for i in range(n):
            for j in range(i, n):
                s = vals[i] + vals[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Determine unique variable indices
                    if i == j:
                        vars_involved = tuple(sorted((i, k)))
                    else:
                        vars_involved = tuple(sorted((i, j, k)))
                    
                    if vars_involved not in seen_constraints:
                        seen_constraints.add(vars_involved)
                        c_idx = len(cons_vars)
                        cons_vars.append(list(vars_involved))
                        for v in vars_involved:
                            var_cons[v].append(c_idx)
        
        num_cons = len(cons_vars)
        
        def is_violated(c, state):
            return all(state[v] == 1 for v in cons_vars[c])
        
        def get_violations(state):
            violated = set()
            for c in range(num_cons):
                if is_violated(c, state):
                    violated.add(c)
            return violated
        
        # Initial solutions
        rng = random.Random(seed)
        
        def init_odds():
            return [1 if v % 2 != 0 else 0 for v in vals]
        
        def init_top():
            # Heuristic: large positive numbers
            max_v = max(vals)
            if max_v > 0:
                threshold = max_v / 2.0
                return [1 if v > threshold else 0 for v in vals]
            else:
                return [0] * n

        def init_bottom():
            min_v = min(vals)
            if min_v < 0:
                threshold = min_v / 2.0
                return [1 if v < threshold else 0 for v in vals]
            else:
                return [0] * n

        def init_random():
            return [rng.randint(0, 1) for _ in range(n)]
        
        initializers = [init_odds, init_top, init_bottom, init_random]
        
        best_sol = [0] * n
        best_size = 0
        
        start_time = time.time()
        
        # Try each initializer with a portion of the time, or run one long search with restarts
        # Let's run a single search loop that can restart.
        
        current_init = 0
        sol = initializers[current_init]()
        
        # Calculate initial violations
        violated = get_violations(sol)
        
        # Track best
        cur_size = sum(sol)
        if cur_size > best_size:
            best_size = cur_size
            best_sol = sol[:]
        
        restart_threshold = 1000 # iterations without improvement
        iter_count = 0
        last_improve_iter = 0
        max_iter = 10000000 # safety
        
        while time.time() - start_time < time_limit and iter_count < max_iter:
            iter_count += 1
            
            if cur_size > best_size:
                best_size = cur_size
                best_sol = sol[:]
                last_improve_iter = iter_count
                
            if len(violated) == 0:
                # Try to add a variable
                # Find a variable v with sol[v]==0 that doesn't cause violation
                # To be fast, we can just pick a random 0 and check.
                # Or maintain a list of "safe" zeros? Too complex.
                # Just pick a random zero.
                zeros = [i for i in range(n) if sol[i] == 0]
                if zeros:
                    v = rng.choice(zeros)
                    # Check if setting v=1 violates any constraint
                    safe = True
                    for c in var_cons[v]:
                        # Check if other vars in c are 1
                        others = [u for u in cons_vars[c] if u != v]
                        if all(sol[u] == 1 for u in others):
                            safe = False
                            break
                            
                    if safe:
                        sol[v] = 1
                        cur_size += 1
                        # Update violations
                        # v was 0, now 1. Constraints involving v might become violated.
                        for c in var_cons[v]:
                            if c not in violated:
                                if is_violated(c, sol):
                                    violated.add(c)
                        continue
                else:
                    # No zeros, can't add
                    pass
                
                # If we couldn't add, or no zeros, we are stuck.
                # Kick: flip a random 1 to 0
                ones = [i for i in range(n) if sol[i] == 1]
                if ones:
                    v = rng.choice(ones)
                    sol[v] = 0
                    cur_size -= 1
                    for c in var_cons[v]:
                        if c in violated:
                            if not is_violated(c, sol):
                                violated.discard(c)
                        # If c was not violated, it remains not violated (since we turned a 1 to 0)
                    continue
                else:
                    # No ones, empty set. Break?
                    break

            else:
                # Resolve violation
                c = rng.choice(list(violated))
                # Pick a random variable in this constraint to flip
                v = rng.choice(cons_vars[c])
                # Flip v
                if sol[v] == 1:
                    sol[v] = 0
                    cur_size -= 1
                    for cc in var_cons[v]:
                        if cc in violated:
                            if not is_violated(cc, sol):
                                violated.discard(cc)
                else:
                    # Should not happen if we only pick from violated constraints (all vars 1)
                    # But if logic error, handle it
                    sol[v] = 1
                    cur_size += 1
                    for cc in var_cons[v]:
                        if cc not in violated:
                            if is_violated(cc, sol):
                                violated.add(cc)
                
                # Restart logic
                if iter_count - last_improve_iter > restart_threshold:
                    # Restart with a new initializer
                    current_init = (current_init + 1) % len(initializers)
                    sol = initializers[current_init]()
                    violated = get_violations(sol)
                    cur_size = sum(sol)
                    if cur_size > best_size:
                        best_size = cur_size
                        best_sol = sol[:]
                    last_improve_iter = iter_count
                    restart_threshold = rng.randint(500, 2000)
        
        return [vals[i] for i in range(n) if best_sol[i]]
    
    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        # Allocate time based on size?
        # 60: 10s, 120: 20s, 240: 40s? Total 70s.
        # Let's use 20s, 30s, 40s = 90s.
        time_limit = 20.0 if size == 60 else (30.0 if size == 120 else 40.0)
        subset = solve_sum_free(vals, seed=seed, time_limit=time_limit)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
