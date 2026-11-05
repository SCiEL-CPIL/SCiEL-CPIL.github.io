import time
import random
import math
import bisect
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solutions with column-wise greedy + randomized variants
    seed = 0
    while time.time() < deadline - 80:  # Reserve 80s for SA + polish
        rows = _column_wise_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope <= lower_bound + 1:
                    break
        seed += 1
    
    # Also try traditional greedy for diversity
    if time.time() < deadline - 75:
        rows = _greedy_construct(n_rows, k, 0)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        if best_rows is None:
            # Fallback: simple sequential
            best_rows = _simple_construct(n_rows, k)
        best_scope = _scope(best_rows)
    
    # Phase 2: Local search with maintained diff-count (fast validity checks)
    if time.time() < deadline - 15:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Element-wise local search for final polish
    if time.time() < deadline - 5 and best_rows is not None:
        polished = _element_wise_polish(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    # Phase 4: Final constrained max-row reduction
    if time.time() < deadline - 1 and best_rows is not None:
        final = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if final is not None:
            fin_scope = _scope(final)
            if fin_scope < best_scope:
                best_scope = fin_scope
                best_rows = final
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs_set(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _simple_construct(n_rows, k):
    """Simple sequential construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
                    break
                x += 1
        rows.append(row)
    return rows

def _column_wise_construct(n_rows, k, seed):
    """
    Column-wise list-scheduling construction.
    Build all rows simultaneously, processing positions left-to-right.
    Uses randomized row ordering and value selection for diversity.
    """
    rng = random.Random(seed * 7919 + 13)
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    for pos in range(1, k + 1):
        # Randomize row processing order
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for idx in row_order:
            row = rows[idx]
            prev_val = row[-1]
            
            # Try values starting from prev_val + 1 with some randomness
            x = prev_val + 1
            found = False
            max_x = prev_val + 100
            
            while x <= max_x:
                new_diffs = set()
                conflict = False
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if not conflict:
                    # With probability 0.8, take the first valid value; otherwise explore further
                    if rng.random() < 0.8 or x == max_x:
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                    # Otherwise skip and try next
                x += 1
            
            if not found:
                # Try harder with larger range
                for x in range(prev_val + 1, prev_val + 500):
                    new_diffs = set()
                    conflict = False
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if not conflict:
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
            
            if not found:
                return None
    
    return rows

def _greedy_construct(n_rows, k, seed):
    """Best-fit greedy construction."""
    rng = random.Random(seed)
    used = set()
    used_sorted = []
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            scored_xs = []
            last_x = row[-1]
            limit = 400
            
            x = last_x + 1
            while len(valid_xs) < 30 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 30:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        score = _packing_score(new_diffs, used_sorted, x)
                        scored_xs.append((score, x))
                    x += 1
            
            if not valid_xs:
                return None
            
            scored_xs.sort(key=lambda t: t[0])
            top_n = min(4, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(used_sorted, d)
        
        rows.append(row)
    
    return rows

def _packing_score(new_diffs, used, x):
    """Score a candidate x by how well its differences pack into used space."""
    if not used:
        return x * 1.0
    
    used_sorted = used if isinstance(used, list) else sorted(used)
    
    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist
    
    return total_dist / max(len(new_diffs), 1) + 0.1 * x

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """
    Local search with maintained diff-count for O(1) validity checks.
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute difference sets for each row
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    # Maintain a global count of how many rows use each difference value.
    diff_count = {}
    for diffs in row_diffs:
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def _row_is_valid(new_diffs, exclude_idx):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[exclude_idx]:
                cnt -= 1
            if cnt > 0:
                return False
        return True
    
    def _commit_row(row_idx, new_row, new_diffs):
        nonlocal rows, row_diffs, diff_count
        for d in row_diffs[row_idx]:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        rows[row_idx] = new_row
        row_diffs[row_idx] = new_diffs
    
    # Phase A: General local search
    T_start = 20.0
    T_end = 0.3
    total_steps = 50000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    T = T_start
    steps_since_improve = 0
    max_no_improve = 5000
    
    for step in range(total_steps):
        if time.time() > deadline - 20:
            break
        
        # 45% target max row, 30% random, 25% second-max row
        r = rng.random()
        if r < 0.45:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        elif r < 0.75:
            row_idx = rng.randint(0, n_rows - 1)
        else:
            sorted_idx = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
            row_idx = sorted_idx[1] if n_rows > 1 else 0
        
        new_row = _generate_candidate_row(rows, row_idx, k, rng)
        if new_row is None:
            continue
        
        n = len(new_row)
        new_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                new_diffs.add(new_row[j] - new_row[i])
        
        if not _row_is_valid(new_diffs, row_idx):
            continue
        
        # Compute new scope efficiently
        max_others = max(rows[i][-1] for i in range(n_rows) if i != row_idx)
        new_scope = max(max_others, new_row[-1])
        
        delta = new_scope - current_scope
        if delta <= 0:
            _commit_row(row_idx, new_row, new_diffs)
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                _commit_row(row_idx, new_row, new_diffs)
                current_scope = new_scope
            steps_since_improve += 1
        
        T *= cool_rate
        
        # Perturbation if stuck
        if steps_since_improve > max_no_improve:
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row(rows, pert_idx, k, rng)
            if pert_row is not None:
                n = len(pert_row)
                pert_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        pert_diffs.add(pert_row[j] - pert_row[i])
                if _row_is_valid(pert_diffs, pert_idx):
                    _commit_row(pert_idx, pert_row, pert_diffs)
                    current_scope = _scope(rows)
            steps_since_improve = 0
    
    # Phase B: Dedicated shrink-the-max-row loop
    for attempt in range(60):
        if time.time() > deadline - 5:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        target_max = current_max - 1
        found = False
        for _ in range(15):
            if time.time() > deadline - 5:
                break
            new_row = _build_constrained_row(rows, max_idx, k, rng, target_max)
            if new_row is None:
                target_max += 1
                continue
            n = len(new_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - new_row[i])
            if _row_is_valid(new_diffs, max_idx):
                _commit_row(max_idx, new_row, new_diffs)
                current_scope = _scope(rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
                found = True
                break
            target_max += 1
        
        if not found:
            for _ in range(8):
                if time.time() > deadline - 5:
                    break
                new_row = _generate_candidate_row(rows, max_idx, k, rng)
                if new_row is None:
                    continue
                n = len(new_row)
                new_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        new_diffs.add(new_row[j] - new_row[i])
                if not _row_is_valid(new_diffs, max_idx):
                    continue
                if new_row[-1] < current_max:
                    _commit_row(max_idx, new_row, new_diffs)
                    current_scope = _scope(rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                    break
    
    return best_rows

def _build_constrained_row(rows, exclude_idx, k, rng, target_max):
    """Build a row with all elements <= target_max using best-fit packing."""
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        
        x = last_x + 1
        while x <= target_max:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(3, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)
    
    return row

def _generate_candidate_row(rows, exclude_idx, k, rng):
    """Generate a new candidate row using best-fit packing heuristic."""
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    used_sorted = sorted(used_diffs)
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 400
        
        x = last_x + 1
        while len(valid_xs) < 25 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 25:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(4, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)
    
    return row

def _element_wise_polish(rows, n_rows, k, start_time, deadline):
    """
    Element-wise local search: try to reduce individual elements in each row.
    This is finer-grained than row-level replacement.
    """
    rng = random.Random(999)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute row diffs and global diff count
    row_diffs = []
    diff_count = {}
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def _valid_diffs(new_diffs, excl):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[excl]:
                cnt -= 1
            if cnt > 0:
                return False
        return True
    
    def _commit_element(row_idx, pos, new_val):
        nonlocal rows, row_diffs, diff_count
        row = rows[row_idx]
        old_val = row[pos]
        
        # Compute old diffs involving position pos
        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
        
        # Compute new diffs involving position pos
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)
        
        # Check validity: new_diffs shouldn't conflict with other rows
        # Remove old_diffs from count, check new_diffs, then commit
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        
        valid = True
        for d in new_diffs:
            if diff_count.get(d, 0) > 0:
                valid = False
                break
        
        if valid:
            for d in new_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
            row[pos] = new_val
            # Update row_diffs
            new_row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    new_row_diffs.add(row[j] - row[i])
            row_diffs[row_idx] = new_row_diffs
            return True
        else:
            # Rollback
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
            return False
    
    # Try to reduce elements, starting from the largest values
    for iteration in range(200):
        if time.time() > deadline - 3:
            break
        
        # Sort rows by max element, process largest first
        row_order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
        
        for row_idx in row_order:
            if time.time() > deadline - 3:
                break
            
            row = rows[row_idx]
            # Try to reduce elements from right to left
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline - 3:
                    break
                
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                
                # Try reducing by small amounts first
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val <= lower_bound:
                        break
                    
                    # Quick check: compute new diffs
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    
                    # Check against global count (excluding this row's contribution)
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    
                    # Check if new_diffs conflict with other rows
                    valid = True
                    for d in new_diffs:
                        cnt = diff_count.get(d, 0)
                        if d in row_diffs[row_idx]:
                            cnt -= 1
                        if cnt > 0:
                            valid = False
                            break
                    
                    if valid:
                        if _commit_element(row_idx, pos, new_val):
                            new_scope = _scope(rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [list(r) for r in rows]
                            break  # Move to next element
    
    return best_rows

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: repeatedly try to reduce the maximum element using
    maintained diff-count for fast validity checks.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    row_diffs = []
    diff_count = {}
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def _valid(new_diffs, excl):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[excl]:
                cnt -= 1
            if cnt > 0:
                return False
        return True
    
    def _commit(idx, new_row, new_diffs):
        nonlocal rows, row_diffs, diff_count
        for d in row_diffs[idx]:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        rows[idx] = new_row
        row_diffs[idx] = new_diffs
    
    for iteration in range(80):
        if time.time() > deadline - 1:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        target = current_max - 1
        improved = False
        for _ in range(8):
            if time.time() > deadline - 1:
                break
            new_row = _build_constrained_row(rows, max_idx, k, rng, target)
            if new_row is None:
                target += 1
                continue
            n = len(new_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - new_row[i])
            if _valid(new_diffs, max_idx):
                _commit(max_idx, new_row, new_diffs)
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                improved = True
                break
            target += 1
        
        if not improved:
            new_row = _generate_candidate_row(rows, max_idx, k, rng)
            if new_row is not None and new_row[-1] < current_max:
                n = len(new_row)
                new_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        new_diffs.add(new_row[j] - new_row[i])
                if _valid(new_diffs, max_idx):
                    _commit(max_idx, new_row, new_diffs)
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
    
    return best_rows

def _is_valid(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == n_rows * (k + 1) * k // 2
