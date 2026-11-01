"""DLX exact-cover + greedy fallback for difference-triangle sets."""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Return set of all positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid(rows, k):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0


def _greedy_construct(n_rows, k, deadline):
    """Fast greedy: build rows one by one, picking smallest available values."""
    used = set()
    rows = []
    for _ in range(n_rows):
        if time.time() > deadline:
            break
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_d = {x - a for a in row}
            if not (new_d & used):
                row.append(x)
                used |= new_d
        rows.append(row)
    return rows


def _dlx_solve(n_rows, k, m, deadline):
    """
    DLX exact-cover: select n_rows disjoint Sidon sets (each of size k+1, starting at 0)
    such that their positive differences exactly partition {1, 2, ..., m}.
    Returns list of rows or None.
    """
    total_diffs = n_rows * k * (k + 1) // 2
    if m != total_diffs:
        return None  # exact cover requires m == total_diffs

    # Generate all candidate rows: subsets of {0, 1, ..., m} of size k+1 containing 0,
    # that are Sidon (all internal diffs distinct) and whose diff set is a subset of {1..m}.
    candidates = []
    cand_diffs = []
    max_val = m
    # We need to enumerate all (k+1)-subsets of {0..m} containing 0.
    # For m up to ~105 and k=5 (size 6), this is C(105,5) ~ 96M — too many.
    # Instead, use a smarter generation: build rows greedily with pruning.

    # Alternative: use a recursive generator with pruning.
    def gen_rows(current, used_in_row_diffs, start):
        if len(current) == k + 1:
            diffs = set()
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    diffs.add(current[j] - current[i])
            if len(diffs) == len(current) * (len(current) - 1) // 2:  # all distinct
                candidates.append(current[:])
                cand_diffs.append(diffs)
            return
        # Pruning: remaining positions
        remaining = (k + 1) - len(current)
        # The largest possible value is max_val
        # We need at least 'remaining' more values greater than current[-1]
        if current:
            last = current[-1]
        else:
            last = 0
        # Try values from last+1 to max_val
        # But we can prune: the smallest possible next value is last+1
        # We need to be able to pick 'remaining' values from {last+1, ..., max_val}
        if max_val - last < remaining:
            return
        for x in range(last + 1, max_val + 1):
            # Check that new diffs don't collide within this row
            new_d = {x - a for a in current}
            if new_d & used_in_row_diffs:
                continue
            current.append(x)
            gen_rows(current, used_in_row_diffs | new_d, x + 1)
            current.pop()
            if time.time() > deadline:
                return

    gen_rows([0], set(), 1)

    if not candidates:
        return None

    n_cand = len(candidates)
    if n_cand == 0:
        return None

    # Build exact cover: columns = [n_rows row-slots] + [m diff-slots]
    # Each candidate covers 1 row-slot and |diffs| diff-slots.
    # We need to pick exactly n_rows candidates such that every column is covered exactly once.

    # DLX implementation
    # Columns: 0..n_rows-1 are row slots, n_rows..n_rows+m-1 are diff slots
    n_cols = n_rows + m

    # For each candidate, store which columns it covers
    cand_cols = []
    for i in range(n_cand):
        cols = [i % n_rows]  # row slot — but we want any row slot, not fixed
        # Actually, row slots should be "any of the n_rows rows", so each candidate covers all row slots? No.
        # In exact cover for this problem: we need to assign candidates to rows.
        # Simpler: since rows are interchangeable, we can just pick n_rows candidates
        # whose diff sets partition {1..m}. The row assignment is trivial.
        # So: columns = m diff-slots. We pick n_rows candidates such that their diff sets
        # are disjoint and union = {1..m}.
        # This is a set-packing / exact-cover on the diff universe.

    # Redo: columns = m diff-slots only. We need to pick n_rows candidates
    # whose diff sets are pairwise disjoint and whose union is exactly {1..m}.
    # Each candidate covers |diffs| columns.

    # DLX with m columns
    # Head = 0, columns 1..m, candidates as rows in the matrix
    # L, R, U, D arrays

    # Let's use a simpler backtracking with bitsets for speed
    # m is at most ~105, so we can use Python ints as bitsets

    # diff_bitset[c] = bitmask of diffs covered by candidate c
    diff_bitsets = []
    for diffs in cand_diffs:
        mask = 0
        for d in diffs:
            if 1 <= d <= m:
                mask |= (1 << (d - 1))
            else:
                mask = None
                break
        if mask is None:
            diff_bitsets.append(None)
        else:
            diff_bitsets.append(mask)

    # Filter out candidates with None
    valid_cands = []
    valid_masks = []
    for i in range(n_cand):
        if diff_bitsets[i] is not None:
            valid_cands.append(i)
            valid_masks.append(diff_bitsets[i])

    n_vc = len(valid_cands)
    if n_vc < n_rows:
        return None

    # Sort candidates by number of bits set (fewer first) for better pruning
    order = sorted(range(n_vc), key=lambda i: bin(valid_masks[i]).count('1'))
    ordered_cands = [valid_cands[i] for i in order]
    ordered_masks = [valid_masks[i] for i in order]

    full_mask = (1 << m) - 1

    # Backtracking: pick n_rows candidates with disjoint masks covering full_mask
    # Use recursive search with pruning
    best = [None]

    def backtrack(idx, chosen, covered_mask):
        if time.time() > deadline:
            return
        if len(chosen) == n_rows:
            if covered_mask == full_mask:
                best[0] = [candidates[c] for c in chosen]
                return
            return
        # Pruning: even if we pick all remaining candidates, can we cover full_mask?
        remaining_needed = n_rows - len(chosen)
        if remaining_needed == 0:
            return
        # Try candidates from idx onward
        for i in range(idx, n_vc):
            if time.time() > deadline:
                return
            mask = ordered_masks[i]
            if mask & covered_mask:
                continue  # overlap
            new_mask = covered_mask | mask
            # Pruning: check if new_mask can still reach full_mask
            # (always true since we only add bits)
            chosen.append(ordered_cands[i])
            backtrack(i + 1, chosen, new_mask)
            chosen.pop()
            if best[0] is not None:
                return

    backtrack(0, [], 0)
    return best[0]


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (the scope).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2

    # Simple counting bound: m(m+1)/2 >= total_diffs
    lower_bound = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))

    best_rows = None
    best_scope = float('inf')

    # First, get a quick greedy solution as fallback
    greedy_rows = _greedy_construct(n_rows, k, deadline)
    if greedy_rows:
        best_rows = greedy_rows
        best_scope = _max_entry(greedy_rows)

    # Try DLX for exact cover at m = total_diffs (optimal if it works)
    # m = total_diffs means the differences exactly partition {1..m}
    m_optimal = total_diffs
    if m_optimal >= lower_bound:
        dlx_result = _dlx_solve(n_rows, k, m_optimal, deadline)
        if dlx_result is not None:
            scope = _max_entry(dlx_result)
            if scope < best_scope:
                best_scope = scope
                best_rows = dlx_result

    # If DLX didn't find optimal, try a few m values above total_diffs
    # (in case the optimal doesn't use exactly {1..m} but a subset)
    # Actually, for difference triangles, the optimal might not have consecutive diffs.
    # But let's try m = total_diffs + small offset
    for m_try in range(total_diffs + 1, total_diffs + 10):
        if time.time() > deadline:
            break
        if m_try > best_scope:
            break
        dlx_result = _dlx_solve(n_rows, k, m_try, deadline)
        if dlx_result is not None:
            scope = _max_entry(dlx_result)
            if scope < best_scope:
                best_scope = scope
                best_rows = dlx_result

    # If we still have time, try local search to improve greedy
    if time.time() < deadline - 5:
        rng = random.Random(42)
        rows = best_rows[:]
        if rows:
            used = set()
            for row in rows:
                used |= _row_diffs_set(row)

            current_scope = _max_entry(rows)
            best_local = [r[:] for r in rows]
            best_local_scope = current_scope

            T = 5.0
            cooling = 0.99
            iters = 0
            max_iters = 20000

            while iters < max_iters and time.time() < deadline - 2:
                iters += 1
                row_idx = rng.randint(0, n_rows - 1)
                old_row = rows[row_idx][:]
                old_diffs = _row_diffs_set(old_row)

                for d in old_diffs:
                    used.discard(d)

                # Perturb: randomly adjust one position
                new_row = old_row[:]
                pos = rng.randint(1, len(new_row) - 1)
                delta = rng.randint(-8, 8)
                new_val = new_row[pos] + delta
                valid_move = True
                if new_val <= new_row[pos - 1]:
                    valid_move = False
                if pos < len(new_row) - 1 and new_val >= new_row[pos + 1]:
                    valid_move = False
                if new_val <= 0:
                    valid_move = False
                if valid_move:
                    new_row[pos] = new_val

                # Check validity
                new_diffs = set()
                valid = True
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d in new_diffs or d in used:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        break

                if valid:
                    rows[row_idx] = new_row
                    for d in new_diffs:
                        used.add(d)
                    new_scope = _max_entry(rows)

                    if new_scope <= current_scope:
                        current_scope = new_scope
                        if new_scope < best_local_scope:
                            best_local_scope = new_scope
                            best_local = [r[:] for r in rows]
                    else:
                        if rng.random() < math.exp(-(new_scope - current_scope) / T):
                            current_scope = new_scope
                        else:
                            rows[row_idx] = old_row
                            for d in new_diffs:
                                used.discard(d)
                            for d in old_diffs:
                                used.add(d)
                else:
                    for d in old_diffs:
                        used.add(d)

                T *= cooling

            if best_local_scope < best_scope:
                best_scope = best_local_scope
                best_rows = best_local

    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, deadline)

    return best_rows

# EVOLVE-BLOCK-END
