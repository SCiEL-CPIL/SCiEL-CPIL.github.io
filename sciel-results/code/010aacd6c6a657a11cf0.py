import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Helper to get all differences in a row
    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ai)
        return diffs

    def is_valid_global(rows):
        all_diffs = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row)):
                ai = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ai
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
            # Check strictly increasing
            for a, b in zip(row, row[1:]):
                if b <= a:
                    return False
        return True

    # 1. Row Generation: Greedy with Randomization
    def gen_row(used_diffs, force_greedy=False, scope=None):
        """
        Generates a row of k+1 integers starting at 0.
        Avoids conflicts with used_diffs.
        If scope is given, the max element must be <= scope.
        """
        row = [0]
        local_diffs = set()

        for _ in range(k):
            base = row[-1]
            valid_candidates = []

            # Scan for valid candidates
            # We look ahead a bit to find options
            limit = base + 200
            if scope is not None:
                limit = min(limit, scope)

            for v in range(base + 1, limit + 1):
                new_diffs = {v - a for a in row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & local_diffs:
                    continue
                valid_candidates.append(v)
                if len(valid_candidates) >= 10:
                    break

            if not valid_candidates:
                # Fallback greedy: find the very next valid one
                v = base + 1
                while True:
                    if scope is not None and v > scope:
                        return None
                    new_diffs = {v - a for a in row}
                    if not (new_diffs & used_diffs) and not (new_diffs & local_diffs):
                        break
                    v += 1
                valid_candidates = [v]

            # Selection strategy
            if force_greedy:
                chosen_val = valid_candidates[0]
            else:
                # Bias towards smaller values, but allow some randomness
                # Pick from the first few valid candidates
                pick_idx = random.randint(0, min(4, len(valid_candidates) - 1))
                chosen_val = valid_candidates[pick_idx]

            row.append(chosen_val)
            # Update local_diffs
            for a in row[:-1]:
                local_diffs.add(chosen_val - a)

        return row

    # 2. Backtracking Row Builder with Scope
    def backtrack_row(used, scope):
        """
        Build one row of size k+1 with max <= scope, all diffs disjoint from used.
        Returns the row or None.
        """
        row = [0]
        cur_used = set(used)

        def dfs(row, cur_used):
            if len(row) == k + 1:
                return row[:]
            last = row[-1]
            rem = k + 1 - len(row)
            # Prune: if even tightest packing exceeds scope
            if last + rem > scope:
                return None

            for x in range(last + 1, scope + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & cur_used:
                    continue
                new_row = row + [x]
                new_cur = cur_used | new_diffs
                res = dfs(new_row, new_cur)
                if res is not None:
                    return res
            return None

        return dfs(row, cur_used)

    # 3. Compression: Try to reduce max value of a row
    def try_compress_row(row, forbidden_diffs, target_max=None):
        """
        Attempts to reduce the maximum value of `row` by greedily replacing elements
        from the end to the start with smaller valid values.
        Returns a new row if successful in reducing max, or the original row if not.
        """
        new_row = row[:]
        improved = False

        for i in range(k, 0, -1):
            old_val = new_row[i]
            prefix = new_row[:i]
            prefix_diffs = set()
            for a_idx in range(len(prefix)):
                for b_idx in range(a_idx + 1, len(prefix)):
                    prefix_diffs.add(prefix[b_idx] - prefix[a_idx])

            # Determine upper bound for v
            upper_bound = old_val
            if target_max is not None:
                upper_bound = min(upper_bound, target_max)
            else:
                upper_bound = old_val

            best_v = None
            # Scan from prefix[-1] + 1 up to upper_bound - 1
            for v in range(prefix[-1] + 1, upper_bound):
                # Quick check: diffs with prefix
                new_diffs_prefix = {v - a for a in prefix}
                if new_diffs_prefix & forbidden_diffs:
                    continue
                if new_diffs_prefix & prefix_diffs:
                    continue

                # Check diffs with suffix (elements i+1 to k)
                valid = True
                # We need to ensure all diffs in the candidate row are unique and not in forbidden
                # Candidate row: prefix + [v] + suffix
                suffix = new_row[i + 1:]
                
                # Diffs involving v and suffix: suffix[j] - v
                # Diffs involving prefix and suffix: already in row, but we changed i, so no change there?
                # Wait, changing i changes diffs (i, j) for j>i. It does NOT change diffs (a, b) where a,b != i.
                # So we only need to check:
                # 1. Diffs (v, prefix) -> new_diffs_prefix (checked)
                # 2. Diffs (v, suffix) -> {s - v for s in suffix}
                # 3. Diffs (prefix, suffix) -> unchanged
                # 4. Uniqueness among new_diffs_prefix, new_diffs_suffix, and existing diffs.
                
                new_diffs_suffix = set()
                for s in suffix:
                    d = s - v
                    new_diffs_suffix.add(d)
                    if d in forbidden_diffs:
                        valid = False
                        break
                    if d in new_diffs_prefix:
                        valid = False
                        break
                    if d in prefix_diffs:
                        valid = False
                        break
                if not valid:
                    continue

                # Check uniqueness between new_diffs_prefix and new_diffs_suffix (done above)
                # Check uniqueness against each other? 
                # We added to sets, so we just need to ensure new_diffs_prefix and new_diffs_suffix don't overlap with each other or forbidden/prefix_diffs.
                # We checked new_diffs_suffix against new_diffs_prefix.
                # We checked new_diffs_prefix against prefix_diffs and forbidden.
                # We checked new_diffs_suffix against prefix_diffs and forbidden.
                # Do we need to check new_diffs_suffix against suffix internal diffs?
                # The suffix internal diffs are part of the original row and are valid against forbidden.
                # However, we must ensure new_diffs_suffix doesn't collide with suffix internal diffs.
                # Let's compute suffix internal diffs.
                suffix_diffs = set()
                for a_idx in range(len(suffix)):
                    for b_idx in range(a_idx + 1, len(suffix)):
                        suffix_diffs.add(suffix[b_idx] - suffix[a_idx])
                
                if new_diffs_suffix & suffix_diffs:
                    continue
                if new_diffs_prefix & suffix_diffs:
                    continue

                # If we reach here, v is a valid candidate
                best_v = v
                break

            if best_v is not None:
                new_row[i] = best_v
                improved = True

        if improved:
            return new_row
        return row

    # Initial Construction
    # Use a mix of greedy and randomized
    used = set()
    rows = []
    for i in range(n_rows):
        if i == 0:
            row = gen_row(used, force_greedy=True)
        else:
            row = gen_row(used, force_greedy=False)
        
        if row is None:
            # Fallback: pure greedy without scope limit
            row = gen_row(used, force_greedy=True)
            
        row_diffs = get_row_diffs(row)
        used |= row_diffs
        rows.append(row)

    best_rows = [r[:] for r in rows]
    best_score = max(max(r) for r in best_rows)
    best_diffs = set(used)

    # Local Search with Row Replacement and Compression
    current_rows = [r[:] for r in best_rows]
    current_max = best_score

    # Precompute diffs for each row
    current_row_diffs = [get_row_diffs(r) for r in current_rows]

    t = 10.0
    cooling = 0.999
    no_improve_count = 0
    max_no_improve = 500

    while time.time() < deadline:
        if t < 0.1:
            t = 5.0
            no_improve_count = 0

        # Pick a row to modify
        idx = random.randrange(n_rows)

        # Calculate the set of "forbidden" diffs: all diffs from other rows
        forbidden = set()
        for i in range(n_rows):
            if i != idx:
                forbidden |= current_row_diffs[i]

        # Strategy: 
        # 1. Try to compress the existing row.
        # 2. If that fails or doesn't improve, generate a new random row.
        
        new_row = current_rows[idx][:]
        new_row_diffs = get_row_diffs(new_row)

        # Attempt compression
        compressed_row = try_compress_row(new_row, forbidden)
        if compressed_row != new_row:
            new_row = compressed_row
            new_row_diffs = get_row_diffs(new_row)
            # Verify validity (compression should preserve it, but double check)
            if not (new_row_diffs & forbidden):
                # Calculate new global max
                temp_rows = current_rows[:]
                temp_rows[idx] = new_row
                new_max = max(max(r) for r in temp_rows)
                
                if new_max < current_max:
                    current_rows[idx] = new_row
                    current_row_diffs[idx] = new_row_diffs
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_rows = [r[:] for r in current_rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    t *= cooling
                    continue
                elif new_max == current_max:
                    if random.random() < 0.5: # Accept neutral compression
                        current_rows[idx] = new_row
                        current_row_diffs[idx] = new_row_diffs
                        no_improve_count += 1
                        t *= cooling
                        continue

        # If compression didn't help enough, try generating a new row
        # Use greedy with some randomness
        gen_type = random.random()
        if gen_type < 0.3:
            new_row = gen_row(forbidden, force_greedy=True)
        else:
            new_row = gen_row(forbidden, force_greedy=False)
            
        if new_row is None:
            t *= cooling
            continue

        new_row_diffs = get_row_diffs(new_row)
        
        if new_row_diffs & forbidden:
            t *= cooling
            continue
            
        temp_rows = current_rows[:]
        temp_rows[idx] = new_row
        new_max = max(max(r) for r in temp_rows)
        
        if new_max < current_max:
            current_rows[idx] = new_row
            current_row_diffs[idx] = new_row_diffs
            current_max = new_max
            if current_max < best_score:
                best_score = current_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif new_max == current_max:
            if random.random() < 0.2:
                current_rows[idx] = new_row
                current_row_diffs[idx] = new_row_diffs
                no_improve_count += 1
        else:
            delta = new_max - current_max
            prob = math.exp(-delta / t)
            if random.random() < prob:
                current_rows[idx] = new_row
                current_row_diffs[idx] = new_row_diffs
                current_max = new_max
                no_improve_count += 1
        
        t *= cooling
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            t = 10.0
            no_improve_count = 0

    # Final Polish: Try backtracking to replace rows with smaller max
    # This is a targeted search for better solutions
    if time.time() < deadline - 10:
        # Sort rows by max descending
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in order:
                if time.time() > deadline - 5:
                    break
                cur_max_row = best_rows[i][-1]
                # Try to find a replacement with max < cur_max_row
                # Use backtracking with scope cur_max_row - 1
                target = cur_max_row - 1
                if target < k: # Minimum possible max is k
                    continue
                
                # Get forbidden diffs
                forbidden = set()
                for j in range(n_rows):
                    if j != i:
                        forbidden |= get_row_diffs(best_rows[j])
                
                new_row = backtrack_row(forbidden, target)
                if new_row is not None:
                    # Validate
                    new_rows = [best_rows[j] if j != i else new_row for j in range(n_rows)]
                    if is_valid_global(new_rows):
                        new_max = max(max(r) for r in new_rows)
                        if new_max < best_score:
                            best_rows = new_rows
                            best_score = new_max
                            improved = True
                            # Re-sort order
                            order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)

    return best_rows
# EVOLVE-BLOCK-END
