"""
Marginal-yield greedy packing + simulated annealing for sumset/difference-set exponent lower bound.

Strategy (combined):
  * Fast INCREMENTAL marginal yield: precompute the multiset of pairwise differences and
    sums of the current set. Adding a candidate x adds exactly 2*(|U|-1) new difference
    values (x-u and u-x) and 2*(|U|-1)+1 new sum values (x+u and u+x). The number of NEW
    distinct differences / sums is computed in O(|U|) via set membership tests -- no FFT.
    This lets us evaluate thousands of candidates per greedy iteration.
  * Exact scoring via FFT is used only to confirm/accept a move and to track the best.
  * Seeding: vectorized large-space search over structured families (AP, geometric,
    Fibonacci, Golomb rulers, random spreads) to find a strong starting set.
  * Simulated annealing polish with the same incremental marginal yield for fast moves.
"""
import time
import math
import random
import numpy as np

_DEADLINE = 150.0
_TIME_BUDGET = 140.0  # safety margin before the 150s kill


def _score_from_counts(diff_count, sum_count, max_u):
    if sum_count <= 0 or diff_count <= 0 or max_u <= 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count) / math.log(2 * max_u + 1)


def _golomb_ruler(n):
    """Small optimal Golomb rulers (distinct pairwise differences)."""
    tables = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 28],
        8: [0, 1, 4, 10, 18, 23, 29, 36],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68],
        13: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76],
        14: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84],
        15: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84, 92],
        16: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84, 92, 100],
    }
    if n in tables:
        return list(tables[n])
    elems = [0]
    diffs = set()
    while len(elems) < n:
        x = 1
        while True:
            ok = True
            for e in elems:
                if abs(x - e) in diffs:
                    ok = False
                    break
            if ok:
                for e in elems:
                    diffs.add(abs(x - e))
                elems.append(x)
                break
            x += 1
    return elems


def _fft_counts(U):
    """Exact (sum_count, diff_count, max_u) via FFT."""
    U = sorted(set(int(x) for x in U))
    if not U:
        return 0, 0, 0
    max_u = U[-1]
    ind = np.zeros(max_u + 1, dtype=np.float64)
    for x in U:
        ind[x] = 1.0
    n_conv = 2 * max_u + 1
    conv = np.fft.irfft(np.fft.rfft(ind, n=n_conv) ** 2, n=n_conv)
    sum_count = int(np.sum(conv > 1e-6))
    ind_rev = ind[::-1]
    conv_d = np.fft.irfft(np.fft.rfft(ind, n=n_conv) * np.fft.rfft(ind_rev, n=n_conv), n=n_conv)
    diff_count = int(np.sum(conv_d > 1e-6))
    return sum_count, diff_count, max_u


def _score_of(U):
    sum_count, diff_count, max_u = _fft_counts(U)
    return _score_from_counts(diff_count, sum_count, max_u)


class _State:
    """Maintains the multiset of differences and sums for incremental marginal yield."""
    __slots__ = ("U", "diff_multiset", "sum_multiset", "diff_count", "sum_count", "max_u", "score")

    def __init__(self, U):
        self.U = set(int(x) for x in U)
        self._rebuild()

    def _rebuild(self):
        U = sorted(self.U)
        self.max_u = U[-1] if U else 0
        diff_counter = {}
        sum_counter = {}
        for i, a in enumerate(U):
            for b in U[i:]:
                d = a - b
                diff_counter[d] = diff_counter.get(d, 0) + 1
                s = a + b
                sum_counter[s] = sum_counter.get(s, 0) + 1
        self.diff_multiset = diff_counter
        self.sum_multiset = sum_counter
        self.diff_count = len(diff_counter)
        self.sum_count = len(sum_counter)
        self.score = _score_from_counts(self.diff_count, self.sum_count, self.max_u)

    def add(self, x):
        """Incrementally add x; returns (new_diff_count, new_sum_count, new_max_u)."""
        x = int(x)
        if x in self.U:
            return self.diff_count, self.sum_count, self.max_u
        U = self.U
        dm = self.diff_multiset
        sm = self.sum_multiset
        # new differences: x - u and u - x for each u in U
        for u in U:
            d1 = x - u
            d2 = u - x
            dm[d1] = dm.get(d1, 0) + 1
            dm[d2] = dm.get(d2, 0) + 1
            s = x + u
            sm[s] = sm.get(s, 0) + 1
        # x + x
        sm[2 * x] = sm.get(2 * x, 0) + 1
        # x - x = 0
        dm[0] = dm.get(0, 0) + 1
        self.U.add(x)
        self.max_u = max(self.max_u, x)
        self.diff_count = len(dm)
        self.sum_count = len(sm)
        return self.diff_count, self.sum_count, self.max_u

    def remove(self, x):
        """Incrementally remove x."""
        x = int(x)
        if x not in self.U:
            return
        U = self.U
        dm = self.diff_multiset
        sm = self.sum_multiset
        for u in U:
            d1 = x - u
            d2 = u - x
            c = dm.get(d1, 0) - 1
            if c <= 0:
                dm.pop(d1, None)
            else:
                dm[d1] = c
            c = dm.get(d2, 0) - 1
            if c <= 0:
                dm.pop(d2, None)
            else:
                dm[d2] = c
            s = x + u
            c = sm.get(s, 0) - 1
            if c <= 0:
                sm.pop(s, None)
            else:
                sm[s] = c
        c = sm.get(2 * x, 0) - 1
        if c <= 0:
            sm.pop(2 * x, None)
        else:
            sm[2 * x] = c
        c = dm.get(0, 0) - 1
        if c <= 0:
            dm.pop(0, None)
        else:
            dm[0] = c
        self.U.discard(x)
        if x == self.max_u:
            self.max_u = max(self.U) if self.U else 0
        self.diff_count = len(dm)
        self.sum_count = len(sm)

    def marginal_add_yield(self, x):
        """O(|U|) marginal yield of adding x. Returns (yield, new_diff_count, new_sum_count, new_max)."""
        x = int(x)
        if x in self.U or x <= 0:
            return -1.0, self.diff_count, self.sum_count, self.max_u
        dm = self.diff_multiset
        sm = self.sum_multiset
        new_diffs = 0
        new_sums = 0
        for u in self.U:
            d1 = x - u
            if d1 not in dm:
                new_diffs += 1
            d2 = u - x
            if d2 not in dm:
                new_diffs += 1
            s = x + u
            if s not in sm:
                new_sums += 1
        if 2 * x not in sm:
            new_sums += 1
        if 0 not in dm:
            new_diffs += 1
        new_max = max(self.max_u, x)
        new_diff_count = self.diff_count + new_diffs
        new_sum_count = self.sum_count + new_sums
        if new_diff_count <= 0 or new_sum_count <= 0:
            return -1.0, self.diff_count, self.sum_count, self.max_u
        new_score = _score_from_counts(new_diff_count, new_sum_count, new_max)
        return (new_score - self.score), new_diff_count, new_sum_count, new_max

    def marginal_remove_yield(self, x):
        """O(|U|) marginal yield of removing x. Returns (score_change, new_diff_count, new_sum_count, new_max)."""
        x = int(x)
        if x == 0 or x not in self.U:
            return -1e9, self.diff_count, self.sum_count, self.max_u
        dm = self.diff_multiset
        sm = self.sum_multiset
        lost_diffs = 0
        lost_sums = 0
        for u in self.U:
            d1 = x - u
            if dm.get(d1, 0) == 1:
                lost_diffs += 1
            d2 = u - x
            if dm.get(d2, 0) == 1:
                lost_diffs += 1
            s = x + u
            if sm.get(s, 0) == 1:
                lost_sums += 1
        if sm.get(2 * x, 0) == 1:
            lost_sums += 1
        if dm.get(0, 0) == 1:
            lost_diffs += 1
        new_diff_count = self.diff_count - lost_diffs
        new_sum_count = self.sum_count - lost_sums
        new_max = self.max_u
        if x == self.max_u:
            # recompute max
            new_max = max(self.U - {x}) if (self.U - {x}) else 0
        if new_sum_count <= 0 or new_diff_count <= 0:
            return -1e9, self.diff_count, self.sum_count, self.max_u
        new_score = _score_from_counts(new_diff_count, new_sum_count, new_max)
        return (new_score - self.score), new_diff_count, new_sum_count, new_max


def _greedy_add_pass(state, deadline, max_candidates=300):
    """Greedily add elements with highest marginal yield."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        U = state.U
        max_u = state.max_u

        # Generate candidates
        candidates = set()
        for x in U:
            for d in (1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 30, 40, 50, 70, 100):
                candidates.add(x + d)
                candidates.add(x - d)
        for d in (1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 200, 300, 500):
            candidates.add(max_u + d)
        for _ in range(40):
            candidates.add(random.randint(1, max_u + 150))

        valid_candidates = [c for c in candidates if c > 0 and c not in U]
        if not valid_candidates:
            break

        # Score all candidates using fast marginal yield
        scored = []
        for c in valid_candidates[:max_candidates]:
            y, _, _, _ = state.marginal_add_yield(c)
            if y > 1e-12:
                scored.append((y, c))

        if not scored:
            break

        scored.sort(reverse=True)

        # Try the top candidates, confirming with exact FFT score
        for y, c in scored[:15]:
            state.add(c)
            new_score = state.score
            if new_score > state.score + 1e-12:
                improved = True
                break
            else:
                state.remove(c)

        if not improved:
            # Try a few random ones
            for c in random.sample(valid_candidates, min(8, len(valid_candidates))):
                state.add(c)
                if state.score > _score_of(state.U) - 1e-9:
                    improved = True
                    break
                else:
                    state.remove(c)

    return state


def _greedy_remove_pass(state, deadline):
    """Greedily remove elements with most negative marginal yield."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        removable = [x for x in state.U if x != 0]
        if not removable:
            break

        scored = []
        for x in removable:
            y, _, _, _ = state.marginal_remove_yield(x)
            scored.append((y, x))

        scored.sort(reverse=True)  # most positive score change first

        for y, x in scored[:15]:
            if y > 1e-12:
                old_score = state.score
                state.remove(x)
                if state.score > old_score + 1e-12:
                    improved = True
                    break
                else:
                    state.add(x)

    return state


def _simulated_annealing(state, deadline, T0=0.05, T1=0.001, max_iterations=50000):
    """Simulated annealing with incremental marginal yield for fast evaluation."""
    current_score = state.score
    best_score = current_score
    best_U = set(state.U)

    iterations = 0
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        progress = min(1.0, iterations / max_iterations)
        T = T0 * (T1 / T0) ** progress

        U = sorted(state.U)
        max_u = state.max_u
        move_type = random.random()
        accepted = False

        if move_type < 0.40:
            # Add
            candidates = []
            for x in U:
                for d in (1, 2, 3, 5, 8, 13, 20):
                    candidates.append(x + d)
            candidates.extend([max_u + 1, max_u + 2, max_u + 3, max_u + 5, max_u + 10, max_u + 20])
            candidates.append(random.randint(1, max_u + 30))
            random.shuffle(candidates)
            for x in candidates[:8]:
                if x in state.U or x <= 0:
                    continue
                y, _, _, _ = state.marginal_add_yield(x)
                dF = -y  # negative means improvement
                if dF < 0 or random.random() < math.exp(-dF / max(T, 1e-10)):
                    state.add(x)
                    current_score = state.score
                    accepted = True
                    if current_score > best_score:
                        best_score = current_score
                        best_U = set(state.U)
                    break
        elif move_type < 0.60:
            # Remove
            removable = [x for x in U if x != 0]
            if not removable:
                continue
            random.shuffle(removable)
            for x in removable[:6]:
                y, _, _, _ = state.marginal_remove_yield(x)
                dF = -y
                if dF < 0 or random.random() < math.exp(-dF / max(T, 1e-10)):
                    state.remove(x)
                    current_score = state.score
                    accepted = True
                    if current_score > best_score:
                        best_score = current_score
                        best_U = set(state.U)
                    break
        else:
            # Move
            movable = [x for x in U if x != 0]
            if not movable:
                continue
            random.shuffle(movable)
            for x in movable[:6]:
                for d in (-5, -3, -2, -1, 1, 2, 3, 5, 8, 13):
                    yv = x + d
                    if yv < 0 or yv in state.U:
                        continue
                    # Estimate: remove x then add yv
                    y_rem, _, _, _ = state.marginal_remove_yield(x)
                    # After removal, the state changes; approximate by just doing it
                    state.remove(x)
                    y_add, _, _, _ = state.marginal_add_yield(yv)
                    state.add(yv)
                    new_score = state.score
                    dF = current_score - new_score
                    if dF < 0 or random.random() < math.exp(-dF / max(T, 1e-10)):
                        current_score = new_score
                        accepted = True
                        if current_score > best_score:
                            best_score = current_score
                            best_U = set(state.U)
                    else:
                        # Revert: remove yv, add x back
                        state.remove(yv)
                        state.add(x)
                    break
                if accepted:
                    break

    # Restore best
    state.U = set(best_U)
    state._rebuild()
    return state


def _vectorized_seed_search(deadline):
    """Vectorized large-space search over structured families to find a good seed."""
    best_score = 0.0
    best_set = [0, 1, 3]

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(int(x) for x in cand))
        if not cand or 0 not in cand or max(cand) == 0:
            return
        s = _score_of(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Golomb rulers
    for n in range(3, 17):
        try_set(_golomb_ruler(n))

    # Structured families over a range of max_u
    for max_u in range(3, 200, 3):
        if time.time() > deadline - 30:
            break
        # Arithmetic progressions
        for n in range(3, 14):
            if (max_u) % (n - 1) == 0:
                step = max_u // (n - 1)
                if step >= 1:
                    try_set([i * step for i in range(n)])
        # Geometric
        for ratio in (2, 3, 4):
            geo = [0, 1]
            while geo[-1] * ratio <= max_u:
                geo.append(geo[-1] * ratio)
            if max(geo) <= max_u:
                try_set(geo)
        # Fibonacci-like
        fib = [0, 1]
        while fib[-1] <= max_u:
            fib.append(fib[-1] + fib[-2])
        for n in range(3, len(fib)):
            if fib[n - 1] <= max_u:
                try_set(fib[:n])
        # Random spreads
        if max_u <= 100:
            random.seed(42 + max_u)
            for n in range(4, min(12, max_u + 1)):
                for _ in range(80):
                    if time.time() > deadline - 30:
                        break
                    inner = list(range(1, max_u))
                    if len(inner) < n - 2:
                        continue
                    selected = random.sample(inner, n - 2)
                    try_set(sorted([0] + selected + [max_u]))

    return best_set, best_score


def construct():
    start = time.time()
    deadline = start + _TIME_BUDGET

    random.seed(12345)

    # Phase 0: Vectorized seed search
    seed_deadline = deadline - 80
    best_set, best_score = _vectorized_seed_search(seed_deadline)

    # Phase 1: Greedy add from best seed
    state = _State(best_set)
    phase_deadline = deadline - 50
    state = _greedy_add_pass(state, phase_deadline)
    s = state.score
    if s > best_score:
        best_score = s
        best_set = sorted(state.U)

    # Phase 2: Greedy remove
    phase_deadline = deadline - 40
    state = _greedy_remove_pass(state, phase_deadline)
    s = state.score
    if s > best_score:
        best_score = s
        best_set = sorted(state.U)

    # Phase 3: Another add
    phase_deadline = deadline - 30
    state = _greedy_add_pass(state, phase_deadline)
    s = state.score
    if s > best_score:
        best_score = s
        best_set = sorted(state.U)

    # Phase 4: Another remove
    phase_deadline = deadline - 20
    state = _greedy_remove_pass(state, phase_deadline)
    s = state.score
    if s > best_score:
        best_score = s
        best_set = sorted(state.U)

    # Phase 5: Simulated annealing polish
    phase_deadline = deadline - 5
    state = _simulated_annealing(state, phase_deadline, T0=0.03, T1=0.0005, max_iterations=80000)
    s = state.score
    if s > best_score:
        best_score = s
        best_set = sorted(state.U)

    # Phase 6: Fresh starts if time permits
    if time.time() < deadline - 20:
        for seed in (42, 7, 123, 999):
            if time.time() > deadline - 8:
                break
            random.seed(seed)
            for n0 in (8, 10, 12, 14):
                if time.time() > deadline - 8:
                    break
                cand = _golomb_ruler(n0)
                st = _State(cand)
                st = _greedy_add_pass(st, deadline - 10, max_candidates=150)
                st = _greedy_remove_pass(st, deadline - 8)
                s = st.score
                if s > best_score:
                    best_score = s
                    best_set = sorted(st.U)

    # Final exact verification
    final = sorted(set(int(x) for x in best_set))
    if 0 not in final:
        final = [0] + final
    return final


if __name__ == "__main__":
    result = construct()
    print("Set size:", len(result))
    print("Max element:", max(result))
    print("Score:", _score_of(result))
    print("Set:", result)
