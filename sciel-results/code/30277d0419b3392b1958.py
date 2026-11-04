import time
import random
import itertools

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    # Get an initial upper bound using a fast greedy method
    init_rows = _simple_greedy(n_rows, k)
    if not init_rows:
        return []
    current_best_scope = max(max(r) for r in init_rows)
    current_best_rows = init_rows

    # Try to improve with a better heuristic
    improved_rows = _greedy_degree(n_rows, k, deadline - 30.0)
    if improved_rows:
        improved_scope = max(max(r) for r in improved_rows)
        if improved_scope < current_best_scope:
            current_best_scope = improved_scope
            current_best_rows = improved_rows

    best_found_scope = current_best_scope
    best_found_rows = current_best_rows

    # Try local search polish on the best found
    if time.time() < deadline - 10.0:
        polished_rows, polished_scope = _polish(best_found_rows, k, deadline - 5.0)
        if polished_scope < best_found_scope:
            best_found_scope = polished_scope
            best_found_rows = polished_rows

    # Try a few more greedy restarts with different strategies
    if time.time() < deadline - 10.0:
        for attempt in range(3):
            if time.time() > deadline - 10.0:
                break
            rows = _greedy_random(n_rows, k, random.Random(42 + attempt))
            if rows:
                scope = max(max(r) for r in rows)
                if scope < best_found_scope:
                    best_found_scope = scope
                    best_found_rows = rows
                    # Polish this new best
                    if time.time() < deadline - 5.0:
                        polished_rows, polished_scope = _polish(best_found_rows, k, deadline - 5.0)
                        if polished_scope < best_found_scope:
                            best_found_scope = polished_scope
                            best_found_rows = polished_rows

    return best_found_rows


def _simple_greedy(n_rows, k):
    """Deterministic smallest-first greedy."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows


def _greedy_degree(n_rows, k, deadline):
    """
    Greedy with degree-based heuristic: prefer candidates that leave more
    options for future elements (i.e., use differences that are less "popular"
    in the sense of being harder to place).
    """
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if time.time() > deadline:
                return None
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # Score: prefer differences that are larger (less constrained)
                # or that don't block many future options
                # Simple heuristic: prefer smaller x (tighter scope)
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    
    return rows


def _greedy_random(n_rows, k, rng):
    """Randomized greedy: at each step, pick from a random subset of valid candidates."""
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # With some probability, accept this candidate
                # Otherwise, try a few more
                if rng.random() < 0.3 or x == 0:
                    row.append(x)
                    used.update(new_diffs)
                else:
                    # Try a few more candidates before deciding
                    best_x = x
                    best_score = float('inf')
                    for trial in range(5):
                        trial_x = x + rng.randint(0, 10)
                        trial_diffs = {trial_x - a for a in row}
                        if not (trial_diffs & used):
                            # Score: prefer smaller x
                            score = trial_x
                            if score < best_score:
                                best_score = score
                                best_x = trial_x
                    row.append(best_x)
                    used.update({best_x - a for a in row})
                    x = best_x
            x += 1
        rows.append(row)
    
    # Verify validity
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)
    
    return rows


def _polish(rows, k, deadline):
    """
    Local search: try to move any element (not just the last) to a smaller value
    that keeps the whole set valid, accepting only moves that reduce the global max.
    """
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)

    def _compute_all_diffs(rows):
        all_d = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    all_d.add(d)
        return all_d

    def _is_valid(rows):
        all_d = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in all_d:
                        return False
                    all_d.add(d)
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to reduce the scope by moving elements in the row with the max value
        max_row_idx = max(range(len(best)), key=lambda i: max(best[i]))
        row = best[max_row_idx]
        
        # Try moving each element (except the first, which is 0) to a smaller value
        for pos in range(len(row) - 1, 0, -1):
            if time.time() > deadline:
                return best, best_scope
            old_val = row[pos]
            
            # Try values from row[pos-1]+1 to old_val-1
            for new_val in range(row[pos - 1] + 1, old_val):
                row[pos] = new_val
                
                # Check validity
                if _is_valid(best):
                    new_scope = max(max(r) for r in best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        break
                    else:
                        # Revert
                        row[pos] = old_val
                else:
                    row[pos] = old_val
            
            if improved:
                break
        
        # Also try to reduce other rows if they have large values
        for row_idx in range(len(best)):
            if time.time() > deadline:
                return best, best_scope
            if row_idx == max_row_idx:
                continue
            row = best[row_idx]
            row_max = max(row)
            if row_max >= best_scope:
                # Try to reduce this row's max
                for pos in range(len(row) - 1, 0, -1):
                    old_val = row[pos]
                    for new_val in range(row[pos - 1] + 1, old_val):
                        row[pos] = new_val
                        if _is_valid(best):
                            new_scope = max(max(r) for r in best)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                improved = True
                                break
                            else:
                                row[pos] = old_val
                        else:
                            row[pos] = old_val
                    if improved:
                        break
            if improved:
                break

    return best, best_scope
