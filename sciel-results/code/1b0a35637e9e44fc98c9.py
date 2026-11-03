import time
import random
import math
from collections import Counter

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            s.add(row[j] - row[i])
    return s

def _build_diff_counter(rows):
    c = Counter()
    for r in rows:
        for d in _diffs_for_row(r):
            c[d] += 1
    return c

def _collision_count_from_counter(counter):
    return sum(max(0, v - 1) for v in counter.values())

# ---- Hopcroft-Karp bipartite matching ----
def _hopcroft_karp(adj, n_left, n_right):
    """
    adj: list of lists, adj[left] = list of right nodes
    Returns matching dict: left -> right, and number of matched lefts.
    """
    import collections
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left
    INF = 10**9

    def bfs():
        queue = collections.deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching, pair_u

def _matching_feasible(rows, M, rng):
    """
    Given a set of rows (each length k+1, starting at 0, strictly increasing),
    check if the differences can be assigned distinct values in 1..M such that
    each row's differences are consistent with the row values.
    Actually, the differences ARE determined by the row values. So this function
    checks: are all differences distinct and in 1..M?
    """
    used = set()
    for r in rows:
        if r[0] != 0 or len(r) != len(rows[0]):
            return False
        for d in _diffs_for_row(r):
            if d <= 0 or d > M or d in used:
                return False
            used.add(d)
    return True

def _construct_rows_matching(n_rows, k, M, deadline, rng):
    """
    Construct n_rows of length k+1 starting at 0, strictly increasing,
    with all differences distinct and in 1..M.
    Uses backtracking with the matching feasibility as a forward-checking
    constraint: at each step, verify that the remaining difference slots
    can still be filled with distinct values in 1..M.
    """
    used_diffs = set()
    rows = []
    total_slots = n_rows * k * (k + 1) // 2

    def _forward_check(used_count, remaining_rows, pos_in_row, row_len, current_row):
        """
        Check if it's still possible to fill all remaining slots.
        used_count: number of diffs already used
        remaining_rows: number of rows still to construct (including current)
        pos_in_row: current position in the row (0-indexed, we're about to place pos_in_row)
        row_len: k+1
        current_row: the current row so far
        """
        # Total diffs needed: for the current row, (row_len - pos_in_row - 1) more elements
        # each new element at position p adds p diffs (to all previous elements)
        # For rows after current: each full row adds k*(k+1)/2 diffs
        current_row_remaining_diffs = 0
        for p in range(pos_in_row + 1, row_len):
            current_row_remaining_diffs += p  # element at pos p has p diffs to previous
        other_rows_diffs = (remaining_rows - 1) * k * (k + 1) // 2
        total_remaining_diffs = current_row_remaining_diffs + other_rows_diffs

        # We need total_remaining_diffs distinct values from 1..M not in used_diffs
        available = M - len(used_diffs)
        return available >= total_remaining_diffs

    def _backtrack_row(pos, current_row, current_diffs, max_val, deadline):
        if time.time() > deadline:
            return None
        n = k + 1
        if pos == n:
            return list(current_row)

        prev = current_row[-1]
        min_next = prev + 1
        max_next = min(max_val, prev + k - pos + 1)  # need at least k-pos+1 more steps of 1 each

        if min_next > max_next:
            return None

        # Forward check
        if not _forward_check(len(used_diffs), n_rows - len(rows), pos, n, current_row):
            return None

        # Generate candidates, smaller first with some randomness
        candidates = list(range(min_next, max_next + 1))
        if len(candidates) > 8:
            mid = len(candidates) // 2
            rng.shuffle(candidates[mid:])

        for val in candidates:
            if time.time() > deadline:
                return None

            # Check conflicts
            conflict = False
            new_diffs = []
            for idx in range(pos):
                d = val - current_row[idx]
                if d in current_diffs or d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)

            if conflict:
                continue

            current_row.append(val)
            current_diffs.update(new_diffs)

            result = _backtrack_row(pos + 1, current_row, current_diffs, max_val, deadline)

            current_row.pop()
            for d in new_diffs:
                current_diffs.discard(d)

            if result is not None:
                return result

        return None

    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None

        best_row = None
        best_max_val = float('inf')

        for attempt in range(40):
            if time.time() > deadline:
                break

            current_row = [0]
            current_diffs = set()

            result = _backtrack_row(1, current_row, current_diffs, M, deadline)

            if result is not None:
                row_max = result[-1]
                if row_max < best_max_val:
                    best_max_val = row_max
                    best_row = result
                if best_max_val <= M:
                    break

        if best_row is None:
            return None

        rows.append(best_row)
        for d in _diffs_for_row(best_row):
            used_diffs.add(d)

    return rows

def _sa_optimize(rows, k, deadline, rng, max_iters=None):
    """
    Simulated Annealing to minimize scope with collision penalty.
    Uses efficient incremental diff counting.
    """
    n_rows = len(rows)
    if max_iters is None:
        max_iters = 2000000

    current = [list(r) for r in rows]
    current_scope = _scope(current)

    # Build diff counter
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count_from_counter(diff_counter)

    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions

    collision_weight = 100000

    def cost(scope, coll):
        return coll * collision_weight + scope

    T = 50.0
    cooling = 0.9999
    min_T = 0.01

    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1

        # Choose random row and position
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)

        old_val = current[i][j]
        prev_val = current[i][j - 1]
        next_val = current[i][j + 1] if j < k else old_val + 10000

        # Move type
        move_type = rng.random()

        if move_type < 0.5:
            # Try to reduce (especially if near max)
            upper = old_val - 1 if old_val > prev_val + 1 else prev_val
            if upper <= prev_val:
                continue
            new_val = rng.randint(prev_val + 1, upper)
        elif move_type < 0.8:
            # Small perturbation
            delta = rng.randint(-8, 8)
            new_val = old_val + delta
        else:
            # Larger move
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 30))

        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        if new_val == old_val:
            continue

        # Compute old diffs for this row
        old_row = current[i]
        old_diffs = _diffs_for_row(old_row)

        # Apply change
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)

        # Check internal validity (all diffs positive and distinct within row)
        if len(new_diffs) != len(set(new_diffs)):
            current[i][j] = old_val
            continue

        # Update diff counter
        for d in old_diffs:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]

        for d in new_diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1

        new_collisions = _collision_count_from_counter(diff_counter)

        # New scope
        new_scope = current_scope
        if new_val > current_scope:
            new_scope = new_val
        elif old_val == current_scope and new_val < old_val:
            # Need to recompute
            new_scope = _scope(current)

        old_c = cost(current_scope, collisions)
        new_c = cost(new_scope, new_collisions)

        if new_c <= old_c:
            accept = True
        else:
            delta_c = new_c - old_c
            prob = math.exp(-delta_c / max(T, min_T))
            accept = rng.random() < prob

        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in old_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1

        T = max(T * cooling, min_T)

    return best_scope, best

def _scope_squeeze(rows, k, deadline, rng):
    """
    Try to reduce the scope by 1 iteratively using local search.
    """
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)

    # Precompute other rows' diff sets for faster checking
    other_diff_sets = []
    for i, r in enumerate(rows):
        s = _row_diffs_set(r)
        other_diff_sets.append(s)

    for _ in range(200):
        if time.time() > deadline:
            break

        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break

        i = rng.choice(max_rows)
        row = rows[i]
        old_max = row[-1]
        prev = row[k - 1]

        found = False
        for new_val in range(old_max - 1, prev, -1):
            row[k] = new_val
            new_diffs = _diffs_for_row(row)

            if len(new_diffs) != len(set(new_diffs)):
                continue

            # Check cross-row conflicts
            conflict = False
            new_diffs_set = set(new_diffs)
            for other_idx, other_set in enumerate(other_diff_sets):
                if other_idx == i:
                    continue
                for d in new_diffs_set:
                    if d in other_set:
                        conflict = True
                        break
                if conflict:
                    break

            if not conflict:
                current_scope = new_val
                found = True
                # Update the diff set for this row
                other_diff_sets[i] = new_diffs_set
                break

        if not found:
            row[k] = old_max
            continue

    return rows

def _binary_search_scope(n_rows, k, deadline, rng):
    """
    Binary search on the scope M to find the minimal feasible M.
    Uses _construct_rows_matching to test feasibility.
    """
    # Lower bound: counting argument
    total_diffs = n_rows * k * (k + 1) // 2
    low = max(1, total_diffs)  # each diff must be distinct, so at least total_diffs values
    # Actually, diffs are from 1..M, and we need total_diffs distinct values, so M >= total_diffs
    # But that's a very loose bound. The simple counting bound is 105 for (7,5).
    # Let's use a tighter lower bound: M >= total_diffs, but also M must be large enough
    # for the row structure. Each row has max diff = row[-1], and we need k+1 elements.
    # A row with k+1 elements starting at 0 has max diff = row[-1].
    # The smallest possible max diff for a row is k (if elements are 0,1,2,...,k).
    # But with distinct differences globally, we need more.
    # For now, use low = total_diffs as a safe lower bound, and high = 200.
    low = max(1, total_diffs)
    high = 200  # known best is 112, so 200 is safe

    # Quick check: if low >= high, just try low
    if low >= high:
        rows = _construct_rows_matching(n_rows, k, low, deadline, rng)
        return rows, low if rows else high

    best_rows = None
    best_M = high

    # Binary search
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        rows = _construct_rows_matching(n_rows, k, mid, deadline, rng)
        if rows is not None:
            best_rows = rows
            best_M = mid
            high = mid - 1
        else:
            low = mid + 1

    if best_rows is None:
        # Fallback: try high
        best_rows = _construct_rows_matching(n_rows, k, high, deadline, rng)
        if best_rows is not None:
            best_M = high

    return best_rows, best_M

def _fast_greedy(n_rows, k, rng, scope_limit=None):
    """
    Fast greedy construction. For each row, greedily pick the smallest
    value that doesn't create a duplicate difference.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin
    rng = random.Random(12345)

    best_result = None
    best_score = float('inf')

    # Phase 1: Binary search on scope using matching-feasible construction
    if time.time() < deadline - 30:
        rows, M = _binary_search_scope(n_rows, k, deadline - 20, rng)
        if rows is not None:
            s = _scope(rows)
            if s < best_score:
                best_score = s
                best_result = rows

    # Phase 2: Fallback to greedy if matching construction failed
    if best_result is None:
        rows = _fast_greedy(n_rows, k, rng)
        if rows is not None:
            best_result = rows
            best_score = _scope(rows)
        else:
            # Last resort: simple sequential
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                rows.append(row)
            best_result = rows
            best_score = _scope(rows)

    # Phase 3: Simulated Annealing optimization
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining * 0.75
        max_iters = int(remaining * 50000)
        s, rows = _sa_optimize(best_result, k, sa_deadline, rng, max_iters=max_iters)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows

    # Phase 4: Scope squeeze
    if time.time() < deadline - 3:
        remaining = deadline - time.time()
        squeeze_deadline = time.time() + remaining * 0.8
        squeezed = _scope_squeeze(best_result, k, squeeze_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed

    # Final validation
    if best_result is not None and not _valid(best_result, k):
        # Shouldn't happen, but fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_result = rows

    return best_result
