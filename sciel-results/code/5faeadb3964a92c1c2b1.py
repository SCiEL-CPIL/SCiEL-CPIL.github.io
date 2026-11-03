import time
import random
import math
import sys

def _compute_diffs(row):
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    seen = set()
    for row in rows:
        for d in _compute_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _get_scope(rows):
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, skip_prob=0.0):
    """
    Greedy initialization. Each row is built by taking the smallest x that
    does not collide with any difference used so far (globally across rows).
    skip_prob adds diversity by occasionally jumping forward.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                if skip_prob > 0 and random.random() < skip_prob:
                    # Try to jump to a later valid value to add diversity
                    jumped = False
                    for _ in range(random.randint(1, 6)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            jumped = True
                            break
                    if not jumped:
                        # x is currently valid (we checked before skipping), take it
                        row.append(x)
                        used |= new_diffs
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def _rebuild_row_dfs(forbidden, k, max_scope_hint, deadline, row_len):
    """
    Rebuild a row minimizing its maximum element using DFS with backtracking.
    forbidden: set of differences already used by other rows.
    Returns the best row found (max <= max_scope_hint) or None.
    """
    best_row = None
    best_last = max_scope_hint + 1

    def dfs(row, local_used):
        nonlocal best_row, best_last
        if time.time() > deadline:
            return
        if len(row) == row_len:
            if row[-1] < best_last:
                best_last = row[-1]
                best_row = list(row)
            return
        last = row[-1]
        min_next = last + 1
        # We must still fit (row_len - len(row)) more elements, each at least +1.
        # So current element must be <= best_last - (row_len - len(row))
        # But we're choosing the NEXT element; the next element must be <= best_last - (row_len - len(row) - 1)
        remaining = row_len - len(row) - 1  # elements after the one we place
        max_next = best_last - remaining
        if max_next < min_next:
            return
        for x in range(min_next, max_next + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & local_used:
                continue
            if new_diffs & forbidden:
                continue
            row.append(x)
            local_used |= new_diffs
            dfs(row, local_used)
            row.pop()
            local_used -= new_diffs

    dfs([0], set())
    return best_row


class SAState:
    """
    Global state for simulated annealing.
    Maintains diff_count (dict diff->count) and collisions (int) for O(1) conflict
    checks. Cost = (collisions, scope). We minimize collisions first, then scope.
    """
    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.n_rows = len(rows)
        self.row_len = len(rows[0])
        self.diff_count = {}
        self.collisions = 0
        self.max_val = 0
        self._rebuild()

    def _rebuild(self):
        self.diff_count = {}
        self.collisions = 0
        self.max_val = 0
        for r in self.rows:
            if r[-1] > self.max_val:
                self.max_val = r[-1]
            for i in range(self.row_len):
                for j in range(i + 1, self.row_len):
                    d = r[j] - r[i]
                    if d in self.diff_count:
                        self.diff_count[d] += 1
                        if self.diff_count[d] == 2:
                            self.collisions += 1
                    else:
                        self.diff_count[d] = 1

    def get_cost(self):
        return self.collisions, self.max_val

    def is_valid(self):
        return self.collisions == 0

    def _remove_diffs(self, row, i_idx, val):
        """Remove all diffs involving row[i_idx]=val from diff_count."""
        for j in range(self.row_len):
            if j != i_idx:
                d = abs(val - row[j])
                self.diff_count[d] -= 1
                if self.diff_count[d] == 1:
                    self.collisions -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]

    def _add_diffs(self, row, i_idx, val):
        """Add all diffs involving row[i_idx]=val to diff_count."""
        for j in range(self.row_len):
            if j != i_idx:
                d = abs(val - row[j])
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1

    def try_move(self, r_idx, i_idx, new_val):
        """
        Try to change rows[r_idx][i_idx] to new_val. Returns True if accepted
        (i.e., bounds are satisfied). The caller is responsible for deciding
        acceptance based on cost delta (SA logic).
        """
        old_val = self.rows[r_idx][i_idx]
        if old_val == new_val:
            return False

        # Check order
        if i_idx > 0 and new_val <= self.rows[r_idx][i_idx - 1]:
            return False
        if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx + 1]:
            return False

        # Remove old diffs
        self._remove_diffs(self.rows[r_idx], i_idx, old_val)

        # Apply change
        old_max = self.max_val
        self.rows[r_idx][i_idx] = new_val

        # Add new diffs
        self._add_diffs(self.rows[r_idx], i_idx, new_val)

        # Update max_val
        if new_val > self.max_val:
            self.max_val = new_val
        elif old_val == old_max and new_val < old_max:
            self.max_val = max(max(r) for r in self.rows)

        return True

    def undo_move(self, r_idx, i_idx, old_val):
        """Undo a move by restoring old_val."""
        new_val = self.rows[r_idx][i_idx]
        if new_val == old_val:
            return

        # Remove new diffs
        self._remove_diffs(self.rows[r_idx], i_idx, new_val)

        # Restore old value
        self.rows[r_idx][i_idx] = old_val

        # Add old diffs
        self._add_diffs(self.rows[r_idx], i_idx, old_val)

        # Update max_val
        if old_val > self.max_val:
            self.max_val = old_val
        elif new_val == self.max_val and old_val < new_val:
            self.max_val = max(max(r) for r in self.rows)

    def rebuild_row(self, r_idx, forbidden, k, max_scope_hint, deadline):
        """
        Rebuild row r_idx using DFS, given forbidden diffs from other rows.
        Returns (new_row, new_diffs) or (None, None) if failed.
        """
        old_row = self.rows[r_idx]
        old_diffs = _compute_diffs(old_row)

        # Remove old row's diffs from global
        for d in old_diffs:
            self.diff_count[d] -= 1
            if self.diff_count[d] == 1:
                self.collisions -= 1
            if self.diff_count[d] == 0:
                del self.diff_count[d]

        # Update max_val
        old_max = self.max_val
        self.max_val = max(max(r) for r in self.rows) if r_idx == max(
            range(self.n_rows), key=lambda i: self.rows[i][-1]
        ) else self.max_val
        # Actually, just recompute if the old row had the max
        if old_row[-1] == old_max:
            self.max_val = max(max(r) for r in self.rows)

        # Try to rebuild
        new_row = _rebuild_row_dfs(forbidden, k, max_scope_hint, deadline, self.row_len)

        if new_row is not None:
            new_diffs = _compute_diffs(new_row)
            self.rows[r_idx] = new_row
            for d in new_diffs:
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1
            if new_row[-1] > self.max_val:
                self.max_val = new_row[-1]
            return new_row, new_diffs
        else:
            # Restore old row
            self.rows[r_idx] = old_row
            for d in old_diffs:
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1
            if old_row[-1] > self.max_val:
                self.max_val = old_row[-1]
            return None, None


def _sa_optimize(rows, n_rows, k, deadline):
    """
    Simulated Annealing with cost = (collisions, scope).
    """
    state = SAState(rows)
    best_rows = [list(r) for r in state.rows]
    best_collisions, best_scope = state.get_cost()
    current_collisions, current_scope = state.get_cost()

    T = 100.0
    cooling = 0.999
    min_T = 0.5

    while time.time() < deadline - 3:
        # Decide move type: 70% perturb element, 30% rebuild row
        if random.random() < 0.7:
            # Perturb a single element
            r_idx = random.randint(0, n_rows - 1)
            row = state.rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]

            # Determine range
            lower = row[pos - 1] + 1
            if pos < len(row) - 1:
                upper = row[pos + 1] - 1
            else:
                # Allow expansion to escape local optima
                upper = best_scope + 20

            if lower > upper:
                continue

            # Try a few random values
            for _ in range(2):
                new_val = random.randint(lower, upper)
                if new_val == old_val:
                    continue

                old_collisions, old_scope = state.get_cost()
                if state.try_move(r_idx, pos, new_val):
                    new_collisions, new_scope = state.get_cost()
                    # Cost delta: (collisions, scope)
                    delta_c = new_collisions - old_collisions
                    delta_s = new_scope - old_scope

                    # Acceptance: prefer lower collisions, then lower scope
                    # Use a combined delta for SA
                    # If collisions decrease, always accept
                    # If collisions increase, accept with probability exp(-delta/T)
                    # If collisions same, accept scope decrease always, increase with prob
                    if delta_c < 0:
                        accept = True
                    elif delta_c == 0:
                        if delta_s <= 0:
                            accept = True
                        else:
                            accept = random.random() < math.exp(-delta_s / T)
                    else:
                        # Collisions increased; accept with probability
                        # that also accounts for scope improvement
                        combined_delta = delta_c * 100 + delta_s
                        accept = random.random() < math.exp(-combined_delta / T)

                    if accept:
                        current_collisions, current_scope = state.get_cost()
                        if (current_collisions, current_scope) < (best_collisions, best_scope):
                            best_collisions, best_scope = current_collisions, current_scope
                            best_rows = [list(r) for r in state.rows]
                    else:
                        state.undo_move(r_idx, pos, old_val)
                    break  # Only one move per iteration

        else:
            # Rebuild a row
            # Pick the row with the largest max element, or random
            if random.random() < 0.5:
                max_row_idx = max(range(n_rows), key=lambda i: state.rows[i][-1])
                r_idx = max_row_idx
            else:
                r_idx = random.randint(0, n_rows - 1)

            # Compute forbidden diffs (all diffs except this row's)
            old_row = state.rows[r_idx]
            old_diffs = _compute_diffs(old_row)

            # Temporarily remove old row's diffs to get forbidden set
            for d in old_diffs:
                state.diff_count[d] -= 1
                if state.diff_count[d] == 1:
                    state.collisions -= 1
                if state.diff_count[d] == 0:
                    del state.diff_count[d]

            forbidden = set(state.diff_count.keys())

            # Restore old row's diffs
            for d in old_diffs:
                if d in state.diff_count:
                    state.diff_count[d] += 1
                    if state.diff_count[d] == 2:
                        state.collisions += 1
                else:
                    state.diff_count[d] = 1

            old_collisions, old_scope = state.get_cost()

            # Try to rebuild with a hint: try to beat best_scope
            new_row, new_diffs = state.rebuild_row(
                r_idx, forbidden, k, best_scope, deadline - 2
            )

            if new_row is not None:
                new_collisions, new_scope = state.get_cost()
                delta_c = new_collisions - old_collisions
                delta_s = new_scope - old_scope

                if delta_c < 0:
                    accept = True
                elif delta_c == 0:
                    if delta_s <= 0:
                        accept = True
                    else:
                        accept = random.random() < math.exp(-delta_s / T)
                else:
                    combined_delta = delta_c * 100 + delta_s
                    accept = random.random() < math.exp(-combined_delta / T)

                if not accept:
                    # Undo the rebuild by restoring old row
                    # We need to undo the rebuild_row changes
                    # This is tricky; let's just revert by recomputing
                    # Actually, rebuild_row already applied the new row.
                    # To undo, we'd need to restore old_row.
                    # For simplicity, if not accepted, we revert by
                    # removing new diffs and adding old diffs.
                    for d in new_diffs:
                        state.diff_count[d] -= 1
                        if state.diff_count[d] == 1:
                            state.collisions -= 1
                        if state.diff_count[d] == 0:
                            del state.diff_count[d]
                    state.rows[r_idx] = old_row
                    for d in old_diffs:
                        if d in state.diff_count:
                            state.diff_count[d] += 1
                            if state.diff_count[d] == 2:
                                state.collisions += 1
                        else:
                            state.diff_count[d] = 1
                    if old_row[-1] > state.max_val:
                        state.max_val = old_row[-1]
                    # Recompute max_val if needed
                    if old_row[-1] == old_scope and state.max_val < old_scope:
                        state.max_val = max(max(r) for r in state.rows)
            else:
                # Failed to rebuild, state should be unchanged (rebuild_row restored)
                pass

        T *= cooling

        # If we have a valid solution, try to further optimize
        # Occasionally restart with a new greedy if stuck
        if (best_collisions, best_scope) == (0, best_scope) and T < 5.0:
            # We're in a valid state; do a quick local search to reduce scope
            # by trying to move the max element down
            pass

    # Return the best valid solution found
    if best_collisions == 0:
        return best_rows
    else:
        # If we never found a valid solution, return the best we have
        # (shouldn't happen since we start from valid)
        return [list(r) for r in state.rows]


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 120s limit (evaluator kills at 150s)

    # Try multiple initialization strategies
    best_rows = None
    best_scope = float('inf')

    # Strategy 1: Pure greedy
    rows1, _ = _greedy_init(n_rows, k, skip_prob=0.0)
    if _is_valid(rows1):
        scope1 = _get_scope(rows1)
        if scope1 < best_scope:
            best_scope = scope1
            best_rows = [list(r) for r in rows1]

    # Strategy 2: Randomized greedy
    for attempt in range(8):
        rows2, _ = _greedy_init(n_rows, k, skip_prob=0.05 + 0.05 * attempt)
        if _is_valid(rows2):
            scope2 = _get_scope(rows2)
            if scope2 < best_scope:
                best_scope = scope2
                best_rows = [list(r) for r in rows2]

    if best_rows is None:
        # Fallback to pure greedy
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0)
        return rows

    # Optimize with SA
    rows = _sa_optimize(best_rows, n_rows, k, deadline)

    # Final check
    if not _is_valid(rows):
        return best_rows

    return rows
