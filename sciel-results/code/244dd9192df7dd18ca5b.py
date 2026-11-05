import random
import time

def construct(n_rows, k):
    start_time = time.time()
    # The evaluator kills at 150s. We need a safety margin.
    # The task description says 120s limit for the function, but the prompt says 150s wall clock kill.
    # We will use 115s as the internal limit to be safe.
    time_limit = 115.0

    best_rows = None
    best_score = float('inf')

    expected_diffs = (k + 1) * k // 2

    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid_row(row):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = row_diffs(row)
        return len(diffs) == expected_diffs

    def get_valid_next(row, used, cap):
        valids = []
        last = row[-1]
        for x in range(last + 1, cap + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and len(new_diffs) == len(row):
                valids.append(x)
        return valids

    def fragmentation_score(used, new_diffs):
        """
        Heuristic: Prefer moves that keep the 'used' set compact.
        We calculate the number of 'gaps' in the used set.
        A gap is a maximal interval of unused integers.
        We want to minimize the number of gaps, or the size of the largest gap?
        Actually, 'contiguous resource carving' suggests we want to fill the space densely.
        So, we prefer moves that result in a used set with fewer, smaller gaps.
        """
        all_diffs = used | new_diffs
        if not all_diffs:
            return 0
        
        # To count gaps efficiently, we can look at the sorted diffs
        # But for small sets, we can just check connectivity or count transitions.
        # A simpler proxy: The sum of (diff - previous_diff - 1) for sorted diffs?
        # Let's use the number of isolated points or gaps.
        # For speed, we'll use a simpler heuristic: 
        # Score = max(all_diffs) - len(all_diffs) (sparsity)
        # Or better: How many integers in [1, max(all_diffs)] are NOT in all_diffs?
        # This is O(max_diff).
        
        # Let's stick to a fast proxy:
        # We prefer x such that the new diffs are close to existing diffs (clustering)
        # or close to 0 (compactness).
        
        # The "fragmentation" idea:
        # Count how many new "holes" are created.
        # A hole is created if x-a is not in used, and (x-a-1) is in used, and (x-a+1) is in used?
        # No, that's a "filled" hole.
        
        # Let's try: Minimize the number of connected components in the graph of used differences?
        # Too slow.
        
        # Let's use: Sum of distances to nearest used diff?
        # If we pick x, new_diffs are d1, d2...
        # Cost = sum(min(|d - u| for u in used) for d in new_diffs)
        # This encourages clustering.
        
        cost = 0
        for d in new_diffs:
            min_dist = float('inf')
            for u in used:
                dist = abs(d - u)
                if dist < min_dist:
                    min_dist = dist
            cost += min_dist
        return cost

    def randomized_greedy(max_scope_hint=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                cap = max_scope_hint if max_scope_hint else 200
                if best_score < float('inf'):
                    cap = min(cap, best_score)

                valids = get_valid_next(row, used, cap)
                if not valids:
                    return None

                # Score valids
                # We want to favor smaller x, but also "good" fragmentation/clustering
                scored_valids = []
                for v in valids[:50]: # Limit to first 50 for speed
                    new_diffs = {v - a for a in row}
                    # Heuristic 1: Prefer smaller v
                    score_v = v
                    # Heuristic 2: Prefer clustering (lower fragmentation cost)
                    cost = fragmentation_score(used, new_diffs)
                    # Combine: We want low v and low cost.
                    # Normalize cost? Cost is usually small.
                    total_score = score_v + cost * 0.1
                    scored_valids.append((total_score, v))
                
                scored_valids.sort()
                
                # Weighted random choice from the top candidates
                # Take top 10
                top_candidates = scored_valids[:10]
                if len(top_candidates) > 1:
                    # Uniform random among top 10? Or weighted by inverse score?
                    # Let's use weighted by 1/score to favor best
                    weights = [1.0 / (s + 1) for s, v in top_candidates]
                    total_w = sum(weights)
                    r = random.uniform(0, total_w)
                    acc = 0
                    chosen = top_candidates[0][1]
                    for (s, v), w in zip(top_candidates, weights):
                        acc += w
                        if r <= acc:
                            chosen = v
                            break
                else:
                    chosen = top_candidates[0][1]

                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows

    def local_search(rows):
        current_max = max(max(r) for r in rows)
        row_diff_sets = [row_diffs(r) for r in rows]
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            row_order = list(range(n_rows))
            random.shuffle(row_order)

            for i in row_order:
                if time.time() - start_time >= time_limit:
                    break
                row = rows[i]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() - start_time >= time_limit:
                        break
                    current_val = row[pos]
                    min_val = row[pos - 1] + 1
                    max_val = current_val - 1

                    if max_val < min_val:
                        continue

                    other_diffs = set()
                    for r_idx in range(n_rows):
                        if r_idx != i:
                            other_diffs.update(row_diff_sets[r_idx])

                    found = False
                    for v in range(min_val, max_val + 1):
                        if v <= row[pos - 1]:
                            continue
                        if pos < len(row) - 1 and v >= row[pos + 1]:
                            continue

                        candidate_row = list(row)
                        candidate_row[pos] = v
                        cand_diffs = row_diffs(candidate_row)

                        if len(cand_diffs) != expected_diffs:
                            continue
                        if cand_diffs & other_diffs:
                            continue

                        rows[i] = candidate_row
                        row_diff_sets[i] = cand_diffs
                        new_max = max(max(r) for r in rows)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                        found = True
                        break
                    if found:
                        break
                if improved:
                    break
        return rows, current_max

    def simulated_annealing(rows, max_iters=20000):
        current_rows = [list(r) for r in rows]
        current_max = max(max(r) for r in current_rows)
        row_diff_sets = [row_diffs(r) for r in current_rows]
        
        best_rows_sa = [list(r) for r in current_rows]
        best_max_sa = current_max
        
        T = 20.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() - start_time >= time_limit - 2:
                break
            if T < 0.01:
                break
                
            i = random.randint(0, n_rows - 1)
            pos = random.randint(1, len(current_rows[i]) - 1)
            
            other_diffs = set()
            for r_idx in range(n_rows):
                if r_idx != i:
                    other_diffs.update(row_diff_sets[r_idx])
            
            current_val = current_rows[i][pos]
            min_val = current_rows[i][pos - 1] + 1
            # Allow increasing values to escape local optima
            max_val = min(current_val + 10, current_max + 10)
            
            if max_val <= min_val:
                continue
                
            v = random.randint(min_val, max_val)
            if v == current_val:
                continue
                
            if v <= current_rows[i][pos - 1]:
                continue
            if pos < len(current_rows[i]) - 1 and v >= current_rows[i][pos + 1]:
                continue
            
            candidate_row = list(current_rows[i])
            candidate_row[pos] = v
            cand_diffs = row_diffs(candidate_row)
            
            if len(cand_diffs) != expected_diffs:
                continue
            if cand_diffs & other_diffs:
                continue
            
            old_row = current_rows[i]
            current_rows[i] = candidate_row
            row_diff_sets[i] = cand_diffs
            actual_new_max = max(max(r) for r in current_rows)
            
            delta = actual_new_max - current_max
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_max = actual_new_max
                if actual_new_max < best_max_sa:
                    best_max_sa = actual_new_max
                    best_rows_sa = [list(r) for r in current_rows]
            else:
                current_rows[i] = old_row
                row_diff_sets[i] = row_diffs(old_row)
            
            T *= cooling
        
        return best_rows_sa, best_max_sa

    def validate_solution(rows):
        all_diffs = set()
        for r in rows:
            if not is_valid_row(r):
                return False
            d_set = row_diffs(r)
            if all_diffs & d_set:
                return False
            all_diffs.update(d_set)
        return True

    # Main Loop
    while time.time() - start_time < time_limit:
        rows = randomized_greedy()
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                
                # Local Search
                if time.time() - start_time < time_limit - 5:
                    improved_rows, improved_score = local_search(best_rows)
                    if validate_solution(improved_rows):
                        if improved_score < best_score:
                            best_score = improved_score
                            best_rows = improved_rows

                # Simulated Annealing
                if time.time() - start_time < time_limit - 10:
                    sa_rows, sa_score = simulated_annealing(best_rows)
                    if validate_solution(sa_rows):
                        if sa_score < best_score:
                            best_score = sa_score
                            best_rows = sa_rows

                if best_score <= 105:
                    break

    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
