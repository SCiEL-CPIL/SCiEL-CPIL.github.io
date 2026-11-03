import time
import random
import math
import sys

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 145.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    counting_bound = math.ceil(math.sqrt(2 * total_diffs))

    # Initial greedy to get a valid baseline
    def greedy_solve(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(k):
                found = False
                for attempt in range(2000):
                    # Bias towards smaller gaps
                    hi = row[-1] + 1 + int(80 * (rng.random() ** 0.3))
                    if hi < row[-1] + 1:
                        hi = row[-1] + 1
                    x = rng.randint(row[-1] + 1, hi)
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                        break
                if not found:
                    # Sequential fallback
                    x = row[-1]
                    while True:
                        x += 1
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            break
            rows.append(row)
        return rows

    best_rows = greedy_solve(42)
    best_scope = max(max(r) for r in best_rows)

    # Simulated Annealing on the difference vectors
    # State: list of n_rows lists, each of length k (the gaps)
    # Row values are prefix sums of gaps.
    # Cost: scope (max value) + penalty for duplicate differences
    
    def get_row_from_gaps(gaps):
        row = [0]
        s = 0
        for g in gaps:
            s += g
            row.append(s)
        return row

    def get_diffs_from_row(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    # Initialize state with random valid-ish gaps
    def init_state(seed):
        rng = random.Random(seed)
        state = []
        for _ in range(n_rows):
            gaps = [rng.randint(1, 10) for _ in range(k)]
            state.append(gaps)
        return state

    def cost(state):
        # Calculate scope and number of duplicate differences
        rows = [get_row_from_gaps(gaps) for gaps in state]
        scope = max(max(r) for r in rows)
        
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs_from_row(r))
        
        # Count duplicates
        seen = set()
        dup_count = 0
        for d in all_diffs:
            if d in seen:
                dup_count += 1
            else:
                seen.add(d)
        
        # We want to minimize scope, but duplicates are a hard constraint.
        # If we treat duplicates as a huge penalty, SA might find valid solutions.
        # Penalty should be large enough to force validity before optimizing scope.
        # Let's use a dynamic penalty or a very large one.
        # Since scope is ~100-200, and total diffs ~112, max dup count is 112.
        # Let's say penalty per duplicate is 1000.
        return scope + 1000 * dup_count, dup_count

    def anneal(seed, time_budget):
        rng = random.Random(seed)
        start_time = time.time()
        state = init_state(seed)
        current_cost, current_dups = cost(state)
        best_state = [g[:] for g in state]
        best_cost = current_cost
        best_dups = current_dups
        
        T = 1000.0
        cooling_rate = 0.999
        
        while time.time() - start_time < time_budget:
            if T < 1.0:
                break
                
            # Pick a random row and a random gap index
            r_idx = rng.randint(0, n_rows - 1)
            g_idx = rng.randint(0, k - 1)
            
            old_gap = state[r_idx][g_idx]
            # Propose a new gap
            # Bias towards small gaps to keep scope low, but allow larger ones to escape
            # new_gap = max(1, old_gap + rng.randint(-5, 5))
            # Or random restart of the gap
            if rng.random() < 0.1:
                new_gap = rng.randint(1, 50)
            else:
                new_gap = max(1, old_gap + rng.randint(-3, 3))
            
            if new_gap == old_gap:
                continue
                
            state[r_idx][g_idx] = new_gap
            
            new_cost, new_dups = cost(state)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_cost = new_cost
                current_dups = new_dups
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_dups = new_dups
                    best_state = [g[:] for g in state]
            else:
                state[r_idx][g_idx] = old_gap
                
            T *= cooling_rate
            
        # Return the best state found
        if best_dups == 0:
            rows = [get_row_from_gaps(gaps) for gaps in best_state]
            return rows, max(max(r) for r in rows)
        return None, float('inf')

    # Run SA for a while
    # We can try multiple seeds or restarts
    current_best_scope = best_scope
    current_best_rows = best_rows
    
    # Try to improve
    t_start = time.time()
    num_restarts = 0
    while time.time() - t_start < 130:
        num_restarts += 1
        seed = 100 + num_restarts
        time_left = deadline - time.time() - 5
        if time_left < 2:
            break
        # Allocate time to this restart
        t_budget = min(10.0, time_left / 2) 
        rows, scope = anneal(seed, t_budget)
        if rows is not None and scope < current_best_scope:
            current_best_scope = scope
            current_best_rows = rows
            # If we hit the counting bound, we can stop
            if current_best_scope <= counting_bound:
                break

    # If SA didn't find a valid solution, return the greedy one
    # But SA usually finds valid ones quickly because the penalty is high.
    # However, if it fails, we must return a valid solution.
    # The greedy one is guaranteed valid.
    
    # One more check: if SA found a valid one with scope < greedy, use it.
    # If SA found valid but scope > greedy, keep greedy.
    # If SA found no valid, keep greedy.
    
    # Wait, the cost function returns scope + penalty. 
    # If best_dups == 0, it's valid.
    # The scope returned is the actual scope.
    
    # We should also try a local search on the best SA result if it's close?
    # No, let's stick to the SA result if it's valid and better.
    
    return current_best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
