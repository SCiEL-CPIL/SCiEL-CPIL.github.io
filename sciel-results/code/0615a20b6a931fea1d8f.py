import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin from 150s kill

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2  # pairwise differences per row of length k+1

    def get_diff_mask(row):
        """Calculate the bitmask of all pairwise differences for a row."""
        mask = 0
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                mask |= (1 << d)
        return mask

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used_mask = 0
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            m = get_diff_mask(r)
            if m & used_mask:
                return False
            if bin(m).count("1") != needed:
                return False
            used_mask |= m
        return True

    # ---- Multiple Greedy Initializations ----
    def greedy_deterministic():
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    new_diffs_mask |= bit
                if not conflict:
                    row.append(x)
                    used_mask |= new_diffs_mask
            rows.append(row)
        return rows

    def greedy_randomized(seed):
        rng = random.Random(seed)
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    new_diffs_mask |= bit
                if not conflict:
                    # Randomly decide to take this or skip? 
                    # Simple randomized greedy: take the first valid one, but maybe add some noise?
                    # Actually, for this problem, the first valid one is usually best for minimizing scope.
                    # To get diversity, we can randomly skip some valid candidates.
                    if rng.random() < 0.2: # 20% chance to skip a valid candidate
                        continue
                    row.append(x)
                    used_mask |= new_diffs_mask
            rows.append(row)
        return rows

    # ---- Efficient Backtracking with Forward Checking ----
    def backtrack_construct(target_max, rng, n_rows, k, time_budget=3.0):
        """
        Attempts to build a set of rows with max entry <= target_max.
        Uses forward checking and randomized restarts.
        """
        used_mask = 0
        rows = []
        t_start = time.time()

        def build_row(used_mask, target_max, row_prefix, row_mask, depth, time_budget):
            """
            Builds a single row of length k+1 starting with 0,
            avoiding diffs in used_mask, with max element <= target_max.
            Returns None if not possible.
            """
            if depth == k:
                return row_prefix[:]

            last = row_prefix[-1]
            
            # Collect valid candidates with forward checking
            candidates = []
            for x in range(last + 1, target_max + 1):
                if time.time() - t_start > time_budget:
                    return None
                
                new_diffs_mask = 0
                conflict = False
                for a in row_prefix:
                    d = x - a
                    bit = 1 << d
                    if (used_mask & bit) or (row_mask & bit):
                        conflict = True
                        break
                    new_diffs_mask |= bit
                
                if not conflict:
                    # Forward check: ensure we can still complete the row
                    remaining_positions = k - depth - 1
                    if x > target_max - remaining_positions:
                        continue  # not enough room for remaining elements
                    candidates.append((x, new_diffs_mask))
            
            if not candidates:
                return None
            
            # Randomize order for diversity
            rng.shuffle(candidates)
            
            # Try candidates
            for x, new_diffs_mask in candidates:
                res = build_row(used_mask, target_max, row_prefix + [x], row_mask | new_diffs_mask, depth + 1, time_budget)
                if res is not None:
                    return res
            
            return None

        for _ in range(n_rows):
            if time.time() - t_start > time_budget:
                return None
            
            row = build_row(used_mask, target_max, [0], 0, 0, time_budget)
            if row is None:
                return None
            rows.append(row)
            used_mask |= get_diff_mask(row)
        
        return rows

    # ---- Optimized Simulated Annealing with Incremental Mask Updates ----
    def local_search(start, rng, max_time=8.0, T0=5.0):
        cur = [list(r) for r in start]
        row_masks = [get_diff_mask(r) for r in cur]
        cur_used_mask = 0
        for m in row_masks:
            cur_used_mask |= m

        row_maxes = [max(r) for r in cur]
        cur_scope = max(row_maxes)
        cur_sum = sum(sum(r) for r in cur)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0
        t_start = time.time()

        def try_rebuild_row(other_mask, limit, rng):
            """Rebuild a single row avoiding other_mask, with scope limit."""
            def _bt(cur_row, cur_mask, depth):
                if depth == k:
                    return cur_row[:]
                last = cur_row[-1]
                cands = []
                for x in range(last + 1, limit + 1):
                    nm = 0
                    ok = True
                    for a in cur_row:
                        d = x - a
                        bit = 1 << d
                        if (other_mask & bit) or (cur_mask & bit):
                            ok = False
                            break
                        nm |= bit
                    if ok:
                        cands.append((x, nm))
                if not cands:
                    return None
                rng.shuffle(cands)
                for x, nm in cands:
                    res = _bt(cur_row + [x], cur_mask | nm, depth + 1)
                    if res:
                        return res
                return None
            return _bt([0], 0, 0)

        while time.time() - t_start < max_time and remaining() > 5:
            # Select row: bias towards rows with large max
            if rng.random() < 0.6:
                r = row_maxes.index(cur_scope)
            else:
                r = rng.randrange(n_rows)
            
            row = cur[r]
            other_used_mask = cur_used_mask & ~(row_masks[r])
            
            move_type = rng.random()
            new_row = None
            new_mask = None
            
            if move_type < 0.4:
                # MOVE 1: Targeted Compression of the Max Element
                p = len(row) - 1
                current_max = row[p]
                prev_val = row[p - 1]
                
                for v in range(current_max - 1, prev_val, -1):
                    test_row = row[:]
                    test_row[p] = v
                    # Incremental mask update
                    # Diffs involving old row[p]
                    old_diffs_mask = 0
                    for a in row[:p]:
                        old_diffs_mask |= (1 << (row[p] - a))
                    # Diffs involving new v
                    new_diffs_mask = 0
                    ok = True
                    for a in row[:p]:
                        d = v - a
                        bit = 1 << d
                        if (other_used_mask & bit) or (row_masks[r] & ~old_diffs_mask & bit) or (new_diffs_mask & bit):
                            ok = False
                            break
                        new_diffs_mask |= bit
                    if ok:
                        # Check popcount
                        # new_mask = (row_masks[r] & ~old_diffs_mask) | new_diffs_mask
                        # But we need to ensure no conflicts within new_diffs_mask itself (handled above)
                        # And no conflict with other_used_mask (handled above)
                        # And no conflict with row_masks[r] & ~old_diffs_mask (handled above? 
                        # Wait, row_masks[r] & ~old_diffs_mask contains diffs NOT involving row[p].
                        # new_diffs_mask contains diffs involving v.
                        # Are there conflicts between new_diffs_mask and (row_masks[r] & ~old_diffs_mask)?
                        # Yes, we checked `(row_masks[r] & ~old_diffs_mask & bit)`.
                        # So it seems ok.
                        candidate_mask = (row_masks[r] & ~old_diffs_mask) | new_diffs_mask
                        if bin(candidate_mask).count("1") == needed:
                            new_row = test_row
                            new_mask = candidate_mask
                            break

            elif move_type < 0.75:
                # MOVE 2: Random Interior Mutation
                p = rng.randrange(1, len(row))
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 15)
                
                if lo > hi:
                    continue
                
                span = hi - lo + 1
                v = lo + int(span * (rng.random() ** 1.3))
                
                if v == row[p]:
                    continue
                
                test_row = row[:]
                test_row[p] = v
                
                # Incremental mask update
                old_diffs_mask = 0
                for a in row[:p]:
                    old_diffs_mask |= (1 << (row[p] - a))
                for a in row[p+1:]:
                    old_diffs_mask |= (1 << (a - row[p]))
                    
                new_diffs_mask = 0
                ok = True
                for a in row[:p]:
                    d = v - a
                    bit = 1 << d
                    if (other_used_mask & bit) or (row_masks[r] & ~old_diffs_mask & bit) or (new_diffs_mask & bit):
                        ok = False
                        break
                    new_diffs_mask |= bit
                if ok:
                    for a in row[p+1:]:
                        d = a - v
                        bit = 1 << d
                        if (other_used_mask & bit) or (row_masks[r] & ~old_diffs_mask & bit) or (new_diffs_mask & bit):
                            ok = False
                            break
                        new_diffs_mask |= bit
                
                if ok:
                    candidate_mask = (row_masks[r] & ~old_diffs_mask) | new_diffs_mask
                    if bin(candidate_mask).count("1") == needed:
                        new_row = test_row
                        new_mask = candidate_mask

            else:
                # MOVE 3: Row Reconstruction
                new_row = try_rebuild_row(other_used_mask, cur_scope, rng)
                if new_row is None:
                    new_row = try_rebuild_row(other_used_mask, cur_scope + 8, rng)
                if new_row is None:
                    continue
                new_mask = get_diff_mask(new_row)
                if bin(new_mask).count("1") != needed or (new_mask & other_used_mask):
                    continue

            if new_row is None:
                continue

            new_max_row = max(new_row)
            new_sum_row = sum(new_row)
            
            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)
            
            new_sum = cur_sum - sum(row) + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj
            
            if d <= 0 or (T > 0.01 and rng.random() < math.exp(-d / T)):
                cur[r] = new_row
                row_masks[r] = new_mask
                cur_used_mask = other_used_mask | new_mask
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]
            
            T = max(0.05, T * 0.999)

        return best

    # ---- Main Search Loop ----
    # 1. Initial Greedy (multiple variants)
    best_rows = greedy_deterministic()
    best_scope = scope_of(best_rows)
    
    # Try a few randomized greedy for better initial bound
    for seed in range(20):
        if remaining() < 5:
            break
        gr = greedy_randomized(seed)
        gs = scope_of(gr)
        if gs < best_scope:
            best_scope = gs
            best_rows = gr

    rng = random.Random(42)

    # 2. Main search loop: alternate between backtracking and SA
    while remaining() > 8:
        # Try to find a better solution via Backtracking with target = best_scope - 1
        target = best_scope - 1
        found = False
        
        # Try a few backtracking restarts with time budget
        bt_time_budget = min(2.5, remaining() * 0.3)
        res = backtrack_construct(target, rng, n_rows, k, time_budget=bt_time_budget)
        if res is not None:
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res
                found = True
        
        # Run SA on the current best to polish
        if remaining() > 5:
            sa_time = min(8.0, remaining() * 0.5)
            res = local_search(best_rows, rng, max_time=sa_time)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    # Final safety check
    if not is_valid(best_rows):
        best_rows = greedy_deterministic()

    return best_rows
