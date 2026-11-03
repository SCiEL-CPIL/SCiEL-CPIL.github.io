# ... (imports)

def construct(n_rows, k):
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used |= (1 << d)
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    def solve_scope(scope, seed, max_nodes=1000000):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        nodes = [0]
        
        def build_row(idx):
            nonlocal used_bits, nodes
            if nodes[0] > max_nodes:
                return False
            nodes[0] += 1
            
            if idx == n_rows:
                return True
            
            row = [0]
            row_diff_bits = 0
            
            if idx == 0:
                min_start = 1
            else:
                min_start = rows[idx - 1][1] + 1
            
            def rec(row, row_diff_bits, min_x):
                nonlocal used_bits, nodes
                if nodes[0] > max_nodes:
                    return False
                nodes[0] += 1
                
                if len(row) == k + 1:
                    if used_bits & row_diff_bits:
                        return False
                    
                    used_bits |= row_diff_bits
                    rows.append(row)
                    if build_row(idx + 1):
                        return True
                    else:
                        rows.pop()
                        used_bits ^= row_diff_bits
                        return False

                remaining = (k + 1) - len(row)
                max_x = scope - (remaining - 1)
                
                if max_x < min_x:
                    return False

                blocked = used_bits | row_diff_bits
                candidates = []
                
                for x in range(min_x, max_x + 1):
                    ok = True
                    for a in row:
                        d = x - a
                        if (blocked >> d) & 1:
                            ok = False
                            break
                    if ok:
                        candidates.append(x)
                
                if not candidates:
                    return False
                
                # Heuristic: Sort candidates by "constraint tightness"
                # We want to pick x such that it doesn't block too many future x's?
                # Or just fail-first: pick the one with fewest options for the next step?
                # That's expensive.
                # Let's try picking the smallest x (greedy) or largest x (to spread out)?
                # Usually smallest x is better for minimizing scope.
                
                # Let's add a check: if we are on the last row, we must complete it.
                # If idx == n_rows - 1, we have less flexibility.
                
                # Improved Heuristic:
                # If few candidates, try all in order (deterministic).
                # If many, pick a subset.
                
                if len(candidates) <= 5:
                    for x in candidates:
                        new_diffs = 0
                        for a in row:
                            d = x - a
                            new_diffs |= (1 << d)
                        if rec(row + [x], row_diff_bits | new_diffs, x + 1):
                            return True
                    return False
                else:
                    # Pick a few candidates to try
                    # 1. Smallest
                    # 2. Middle
                    # 3. Largest
                    # 4. Random
                    choices = []
                    if len(candidates) > 0: choices.append(candidates[0])
                    if len(candidates) > 1: choices.append(candidates[len(candidates)//2])
                    if len(candidates) > 2: choices.append(candidates[-1])
                    # Add random
                    if len(candidates) > 3: choices.append(rng.choice(candidates))
                    
                    # Deduplicate while preserving order
                    seen = set()
                    unique_choices = []
                    for c in choices:
                        if c not in seen:
                            seen.add(c)
                            unique_choices.append(c)
                            
                    for x in unique_choices:
                        new_diffs = 0
                        for a in row:
                            d = x - a
                            new_diffs |= (1 << d)
                        if rec(row + [x], row_diff_bits | new_diffs, x + 1):
                            return True
                    return False
            
            return rec(row, row_diff_bits, min_start)
        
        success = build_row(0)
        if success:
            return rows
        return None

    t0 = time.time()
    time_limit = 115
    best_found = best_rows
    best_scope_found = best_scope
    
    scopes_to_check = list(range(best_scope - 1, 111, -1))
    
    for scope in scopes_to_check:
        if time.time() - t0 > time_limit:
            break
        
        scope_start_time = time.time()
        remaining_total = time_limit - (time.time() - t0)
        slice_time = min(4.0, remaining_total)
        
        if slice_time < 0.5:
            break
            
        found = None
        seed = 0
        while time.time() - scope_start_time < slice_time:
            budget = 5000000
            res = solve_scope(scope, seed, max_nodes=budget)
            if res:
                found = res
                break
            seed += 1
            
        if found:
            best_found = found
            best_scope_found = scope
            
    return best_found
