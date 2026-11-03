"""Difference triangle set (4,7) — targeted scope reduction via bitset backtracking + compaction."""
import time

# EVOLVE-BLOCK-START
def _greedy(n_rows, k):
    """Greedy rulers: smallest next mark whose differences are all unused."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows


def _used_diffs(rows):
    s = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(r[j] - r[i])
    return s


def _valid(rows):
    seen = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows)


def _search_scope(n_rows, k, M):
    """Try to build n_rows valid rows with max entry <= M. Returns rows or None."""
    total = n_rows * k  # number of non-zero marks
    used = [False] * (M + 1)

    def place(row, pos, val, used_count):
        # add val to row at pos; check all diffs to earlier marks
        for i in range(pos):
            d = val - row[i]
            if d > M or used[d]:
                return False
        for i in range(pos):
            used[val - row[i]] = True
        row[pos] = val
        used_count += pos
        return True

    def unplace(row, pos, used_count):
        for i in range(pos):
            used[val - row[i]] = False  # val captured
        row[pos] = 0
        used_count -= pos
        return used_count

    # iterative DFS with stack for speed
    # state: (row_idx, pos, val, used_count)
    # We'll use recursion with a deadline check.
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    deadline = _deadline

    def dfs(row_idx, pos):
        if time.time() > deadline:
            return False
        if row_idx == n_rows:
            return True
        if pos == k + 1:
            # row complete, move to next
            return dfs(row_idx + 1, 0)
        row = rows[row_idx]
        lo = (row[pos - 1] + 1) if pos > 0 else 1
        # upper bound for this mark: M
        for val in range(lo, M + 1):
            # check feasibility
            feasible = True
            for i in range(pos):
                d = val - row[i]
                if d > M or used[d]:
                    feasible = False
                    break
            if not feasible:
                continue
            # place
            for i in range(pos):
                used[val - row[i]] = True
            row[pos] = val
            if dfs(row_idx, pos + 1):
                return True
            # unplace
            for i in range(pos):
                used[val - row[i]] = False
            row[pos] = 0
        return False

    if dfs(0, 0):
        return [list(r) for r in rows]
    return None


def _compact(rows, M):
    """Try to reduce scope by shifting marks down. Returns possibly-improved rows."""
    # Greedy left-to-right compaction: for each mark, try smallest value that keeps validity
    # and keeps max <= M (we'll try to reduce M iteratively outside).
    for _ in range(50):  # a few passes
        changed = False
        for ri in range(len(rows)):
            row = rows[ri]
            for pos in range(1, k + 1):
                old = row[pos]
                lo = (row[pos - 1] + 1) if pos > 0 else 1
                for new in range(lo, old):
                    # tentatively set
                    row[pos] = new
                    if _valid(rows):
                        changed = True
                        break
                    row[pos] = old
                # if not changed, keep old
        if not changed:
            break
    return rows


_deadline = 0.0

def construct(n_rows, k):
    global _deadline
    t0 = time.time()
    _deadline = t0 + 130.0  # safety margin before 150s kill

    # 1. Greedy baseline
    rows = _greedy(n_rows, k)
    best = rows
    best_scope = _scope(rows)

    # 2. Try to reduce scope via targeted search, from best_scope-1 downward
    #    But first, try compaction on the greedy result.
    compacted = _compact(rows, best_scope)
    cs = _scope(compacted)
    if cs < best_scope:
        best = compacted
        best_scope = cs

    # 3. Targeted scope search: try M from best_scope-1 down to a lower bound
    #    Lower bound: need n_rows * k distinct positive diffs, min possible max diff
    #    Counting bound: roughly sqrt(2 * total) but we use a practical lower bound
    total = n_rows * k
    # simple lower bound: each row of k+1 marks gives k diffs, total n_rows*k diffs
    # all in [1, M], so M >= total
    lower = max(total, int((2 * total) ** 0.5))

    for M in range(best_scope - 1, lower - 1, -1):
        if time.time() > _deadline:
            break
        result = _search_scope(n_rows, k, M)
        if result is not None:
            # try to compact further
            compacted = _compact(result, M)
            cs = _scope(compacted)
            if cs < best_scope:
                best = compacted
                best_scope = cs
            # after finding one at M, we could try lower M, but break to save time
            # Actually continue to try lower M for better score
            # But each search is expensive; break after first success to be safe
            break

    # 4. Final compaction pass on the best
    if time.time() < _deadline:
        final = _compact(best, best_scope)
        if _scope(final) < best_scope:
            best = final

    return best
# EVOLVE-BLOCK-END
