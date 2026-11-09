import random
import time
import sys
import copy

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    
    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)

    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_solution(seed=0):
        random.seed(seed)
        used = set()
        rows = []
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                candidates = []
                x = current_max + 1
                limit = x + 1000 + row_idx * 50
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 50:
                            break
                    x += 1
                if not candidates:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1
                if len(candidates) <= 3:
                    chosen = random.choice(candidates)
                else:
                    alpha = 0.005
                    weights = [1.0 / (1.0 + c * alpha) for c in candidates[:20]]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates[:20], weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen = c
                            break
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            rows.append(row)
        return rows

    def rebuild_row(rows, row_idx, seed, max_time=2.0):
        start = time.time()
        other_rows = [r for i, r in enumerate(rows) if i != row_idx]
        used_by_others = set()
        for r in other_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_by_others.add(r[j] - r[i])
        random.seed(seed)
        best_new_row = None
        best_new_max = max(rows[row_idx])
        for attempt in range(10):
            if time.time() - start > max_time:
                break
            row = [0]
            current_max = 0
            valid = True
            new_row_used = set()
            while len(row) < k_plus_1:
                x = current_max + 1
                found = False
                limit = x + 500
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_by_others) and not (new_diffs & new_row_used):
                        if len(new_diffs) == len(row):
                            row.append(x)
                            new_row_used.update(new_diffs)
                            current_max = x
                            found = True
                            break
                    x += 1
                if not found:
                    valid = False
                    break
            if valid:
                row_max = max(row)
                if row_max < best_new_max:
                    best_new_max = row_max
                    best_new_row = row
        if best_new_row is not None:
            new_rows = [list(r) for r in rows]
            new_rows[row_idx] = best_new_row
            return new_rows
        return None

    def local_search(rows, max_time=5.0):
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        start = time.time()
        used = compute_used(rows)
        improved = True
        stall_count = 0
        seed_counter = 0
        while improved and time.time() - start < max_time:
            improved = False
            stall_count += 1
            elements = []
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if i > 0:
                        elements.append((rows[r][i], r, i))
            elements.sort(reverse=True)
            found_improvement = False
            for val, max_r, max_i in elements[:5]:
                if time.time() - start > max_time - 0.1:
                    break
                row = rows[max_r]
                current_val = row[max_i]
                if current_val <= 0:
                    continue
                min_val = row[max_i - 1] + 1
                max_allowed = current_val - 1
                if max_i < len(row) - 1:
                    max_allowed = min(max_allowed, row[max_i + 1] - 1)
                if max_allowed < min_val:
                    continue
                old_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - current_val)
                        old_diffs.add(d)
                used -= old_diffs
                new_val_found = None
                candidates_to_try = list(range(min_val, min(current_val, min_val + 100)))
                for v in candidates_to_try:
                    valid = True
                    new_diffs = set()
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                    if valid and len(new_diffs) == len(row) - 1:
                        new_val_found = v
                        break
                if new_val_found is not None:
                    row[max_i] = new_val_found
                    used |= new_diffs
                    found_improvement = True
                    score = get_score(rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in rows]
                    break
                else:
                    used |= old_diffs
            if found_improvement:
                improved = True
                continue
            max_val = -1
            max_r = -1
            for r in range(n_rows):
                if max(rows[r]) > max_val:
                    max_val = max(rows[r])
                    max_r = r
            if max_r != -1 and time.time() - start < max_time - 0.5:
                seed_counter += 1
                new_rows = rebuild_row(rows, max_r, seed_counter, max_time=1.0)
                if new_rows:
                    score = get_score(new_rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in new_rows]
                        rows = new_rows
                        used = compute_used(rows)
                        improved = True
                        continue
            if stall_count < 3 and time.time() - start < max_time - 0.5:
                swap_done = False
                for r1 in range(n_rows):
                    for r2 in range(r1 + 1, n_rows):
                        if time.time() - start > max_time - 0.2:
                            break
                        for i1 in range(1, len(rows[r1])):
                            for i2 in range(1, len(rows[r2])):
                                v1 = rows[r1][i1]
                                v2 = rows[r2][i2]
                                if v1 == v2:
                                    continue
                                if i1 > 0 and v2 <= rows[r1][i1-1]:
                                    continue
                                if i1 < len(rows[r1])-1 and v2 >= rows[r1][i1+1]:
                                    continue
                                if i2 > 0 and v1 <= rows[r2][i2-1]:
                                    continue
                                if i2 < len(rows[r2])-1 and v1 >= rows[r2][i2+1]:
                                    continue
                                old_diffs1 = set()
                                for a in rows[r1]:
                                    if a != v1:
                                        old_diffs1.add(abs(a - v1))
                                old_diffs2 = set()
                                for a in rows[r2]:
                                    if a != v2:
                                        old_diffs2.add(abs(a - v2))
                                used -= (old_diffs1 | old_diffs2)
                                new_diffs1 = set()
                                valid = True
                                for a in rows[r1]:
                                    if a != v1:
                                        d = abs(a - v2)
                                        if d in used:
                                            valid = False
                                            break
                                        new_diffs1.add(d)
                                if valid:
                                    new_diffs2 = set()
                                    for a in rows[r2]:
                                        if a != v2:
                                            d = abs(a - v1)
                                            if d in used:
                                                valid = False
                                                break
                                            new_diffs2.add(d)
                                if valid:
                                    rows[r1][i1] = v2
                                    rows[r2][i2] = v1
                                    used |= (new_diffs1 | new_diffs2)
                                    swap_done = True
                                    found_improvement = True
                                    score = get_score(rows)
                                    if score < best_score:
                                        best_score = score
                                        best_rows = [list(r) for r in rows]
                                    break
                                else:
                                    used |= (old_diffs1 | old_diffs2)
                            if swap_done:
                                break
                        if swap_done:
                            break
                    if swap_done:
                        break
                if swap_done:
                    improved = True
                    continue
            if not found_improvement:
                improved = False
                stall_count += 1
                if stall_count > 5:
                    break
        return best_rows

    def pareto_dominated(a, b):
        """
        Check if solution a dominates solution b.
        We minimize the max entry (score).
        Diversity is approximated by the number of distinct small differences used.
        """
        score_a = get_score(a)
        score_b = get_score(b)
        if score_a < score_b:
            return True
        if score_a > score_b:
            return False
        # Tie-breaking: prefer solution with more "compact" differences (lower sum of diffs)
        sum_a = sum(compute_used(a))
        sum_b = sum(compute_used(b))
        return sum_a < sum_b

    def is_dominated(sol, pareto_front):
        for other in pareto_front:
            if pareto_dominated(other, sol):
                return True
        return False

    def update_pareto_front(pareto_front, new_sol):
        # Remove dominated solutions
        new_front = [s for s in pareto_front if not pareto_dominated(new_sol, s)]
        # Check if new solution is dominated by remaining
        if not is_dominated(new_sol, new_front):
            new_front.append(new_sol)
        return new_front

    # Main search loop with Pareto Front
    best_rows = None
    best_score = float('inf')
    pareto_front = []
    
    num_seeds = 150
    for seed in range(num_seeds):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 10:
            break
        
        # Time allocation
        if seed < 30:
            ls_time = min(2.0, remaining * 0.1)
        elif seed < 80:
            ls_time = min(4.0, remaining * 0.15)
        else:
            ls_time = min(6.0, remaining * 0.2)
            
        if ls_time < 0.5:
            break
        
        # 1. Generate initial solution
        rows = greedy_solution(seed)
        
        # 2. Local Search
        rows = local_search(rows, max_time=ls_time)
        
        # 3. Update Best and Pareto Front
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            if score <= 112:
                break
        
        # Check Pareto dominance before adding
        if not is_dominated(rows, pareto_front):
            pareto_front = update_pareto_front(pareto_front, [list(r) for r in rows])
            # Cap Pareto front size to prevent memory issues
            if len(pareto_front) > 50:
                # Keep best 50 by score
                pareto_front.sort(key=lambda r: get_score(r))
                pareto_front = pareto_front[:50]
        
        # 4. Crossover/Mutation from Pareto Front (if time permits and front has size > 1)
        if len(pareto_front) > 1 and remaining > 15:
            # Pick two diverse solutions
            parent1 = random.choice(pareto_front)
            parent2 = random.choice(pareto_front)
            if parent1 is not parent2:
                # Hybrid: Take rows from parent1, but try to improve them using local search
                # Seeded with parent2's characteristics (not directly used, but randomization helps)
                # Simple mutation: Take parent1, apply local search with different seed
                mutated = [list(r) for r in parent1]
                # Slight random perturbation before LS
                if random.random() < 0.5:
                    # Pick a random row and rebuild it
                    r_idx = random.randint(0, n_rows - 1)
                    new_row_res = rebuild_row(mutated, r_idx, seed + 1000, max_time=1.0)
                    if new_row_res:
                        mutated = new_row_res
                
                ls_time_mut = min(3.0, remaining * 0.2)
                if ls_time_mut > 1.0:
                    mutated = local_search(mutated, max_time=ls_time_mut)
                    mut_score = get_score(mutated)
                    if mut_score < best_score:
                        best_score = mut_score
                        best_rows = [list(r) for r in mutated]
                    if not is_dominated(mutated, pareto_front):
                        pareto_front = update_pareto_front(pareto_front, [list(r) for r in mutated])
                        if len(pareto_front) > 50:
                            pareto_front.sort(key=lambda r: get_score(r))
                            pareto_front = pareto_front[:50]

    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Final polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        final_rows = local_search(best_rows, max_time=min(remaining * 0.8, 10.0))
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            
    return best_rows
