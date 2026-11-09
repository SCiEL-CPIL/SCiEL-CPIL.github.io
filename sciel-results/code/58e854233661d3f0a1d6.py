import time
import random
import math
import sys


def construct():
    start = time.time()
    DEADLINE = start + 145.0

    prescribed = [1, 2, 5, 11]
    pres_set = set(prescribed)
    N = 284
    random.seed(777)

    def remaining():
        return DEADLINE - time.time()

    def is_valid_set(s):
        if len(s) != len(set(s)):
            return False
        if not pres_set.issubset(set(s)):
            return False
        for v in s:
            if v < 1 or v > N:
                return False
        seen = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                v = s[i] + s[j]
                if v in seen:
                    return False
                seen.add(v)
        return True

    # ---------- Greedy baseline ----------
    def greedy_build():
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        for x in range(1, N + 1):
            if x in used:
                continue
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    def greedy_build_reversed():
        """Try greedy from high to low."""
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        for x in range(N, 0, -1):
            if x in used:
                continue
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    def greedy_build_spread():
        """Try greedy with spread: pick values that maximize minimum distance to existing."""
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        candidates = [x for x in range(1, N + 1) if x not in used]
        while candidates:
            best_x = None
            best_score = -1
            for x in candidates:
                ok = True
                for a in arr:
                    s = a + x
                    if s in sums:
                        ok = False
                        break
                if ok and (x + x) in sums:
                    ok = False
                if not ok:
                    continue
                min_dist = min(abs(x - a) for a in arr)
                if min_dist > best_score:
                    best_score = min_dist
                    best_x = x
            if best_x is None:
                break
            for a in arr:
                sums.add(a + best_x)
            sums.add(best_x + best_x)
            used.add(best_x)
            arr.append(best_x)
            candidates.remove(best_x)
        return arr

    # ---------- CSP Backtracking Solver ----------
    def csp_solve(target, time_budget):
        """
        CSP backtracking with forward checking to build a Sidon set of given target size.
        Variables: positions 0..target-1 (first 4 are prescribed).
        Domain: 1..N, with forward checking eliminating values that create duplicate sums.
        """
        end_time = start + time_budget
        used_vals = set()
        used_sums = set()
        assignment = []

        # Precompute prescribed
        for v in prescribed:
            used_vals.add(v)
            assignment.append(v)

        # Initialize sums for prescribed
        for i in range(len(prescribed)):
            for j in range(i, len(prescribed)):
                used_sums.add(prescribed[i] + prescribed[j])

        # All candidate values
        all_candidates = list(range(1, N + 1))

        # Variable ordering: try to assign remaining target - 4 positions
        # Use MRV-like heuristic: at each step, pick the value that leaves the most options
        # But for speed, we'll use a simpler approach: try all values in smart order

        def backtrack(pos):
            """Try to assign a value to position pos (0-indexed in full assignment)."""
            if time.time() > end_time:
                return False

            if pos == target:
                return True

            # Get all possible values for this position
            # Filter out used values
            candidates = [v for v in all_candidates if v not in used_vals]

            # Forward checking: for each candidate, check if it creates duplicate sums
            # Sort candidates by how many future conflicts they might cause (heuristic)
            # Simpler: just try in order, but prioritize values that minimize conflicts

            # Score candidates: fewer new sum conflicts is better
            def score_candidate(v):
                conflicts = 0
                for a in assignment:
                    s = a + v
                    if s in used_sums:
                        conflicts += 1
                if v + v in used_sums:
                    conflicts += 1
                return conflicts

            # Try candidates with lowest conflict first
            candidates.sort(key=score_candidate)

            for v in candidates:
                if time.time() > end_time:
                    return False

                # Check if v creates any duplicate sum
                valid = True
                new_sums = []
                for a in assignment:
                    s = a + v
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)
                if valid and (v + v) in used_sums:
                    valid = False

                if not valid:
                    continue

                # Try this value
                used_vals.add(v)
                for s in new_sums:
                    used_sums.add(s)
                used_sums.add(v + v)
                assignment.append(v)

                if backtrack(pos + 1):
                    return True

                # Backtrack
                assignment.pop()
                used_vals.discard(v)
                for s in new_sums:
                    used_sums.discard(s)
                used_sums.discard(v + v)

            return False

        # Try multiple random orderings of candidates for diversity
        attempts = 0
        while time.time() < end_time and attempts < 50:
            # Reset state
            used_vals = set(prescribed)
            used_sums = set()
            for i in range(len(prescribed)):
                for j in range(i, len(prescribed)):
                    used_sums.add(prescribed[i] + prescribed[j])
            assignment = list(prescribed)

            # Shuffle non-prescribed candidates for diversity
            non_pres = [v for v in range(1, N + 1) if v not in pres_set]
            random.shuffle(non_pres)

            # Override all_candidates with shuffled version
            # We need to modify the backtrack to use this order
            # Instead, let's create a fresh solver with shuffled order

            if solve_with_order(target, non_pres, time_budget - (time.time() - start)):
                return assignment

            attempts += 1

        return None

    def solve_with_order(target, candidate_order, time_budget):
        """CSP solver with a specific candidate ordering."""
        end_time = time.time() + time_budget
        used_vals = set(prescribed)
        used_sums = set()
        for i in range(len(prescribed)):
            for j in range(i, len(prescribed)):
                used_sums.add(prescribed[i] + prescribed[j])
        assignment = list(prescribed)

        # Build a dict for fast lookup of candidate order
        order_index = {v: i for i, v in enumerate(candidate_order)}

        def backtrack(pos):
            if time.time() > end_time:
                return False

            if pos == target:
                return True

            # Get candidates in the given order, filtering out used
            candidates = [v for v in candidate_order if v not in used_vals]

            # Score: fewer conflicts first
            def score_candidate(v):
                conflicts = 0
                for a in assignment:
                    s = a + v
                    if s in used_sums:
                        conflicts += 1
                if v + v in used_sums:
                    conflicts += 1
                return conflicts

            # Sort by conflicts, then by original order
            candidates.sort(key=lambda v: (score_candidate(v), order_index.get(v, 999)))

            for v in candidates:
                if time.time() > end_time:
                    return False

                valid = True
                new_sums = []
                for a in assignment:
                    s = a + v
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)
                if valid and (v + v) in used_sums:
                    valid = False

                if not valid:
                    continue

                used_vals.add(v)
                for s in new_sums:
                    used_sums.add(s)
                used_sums.add(v + v)
                assignment.append(v)

                if backtrack(pos + 1):
                    return True

                assignment.pop()
                used_vals.discard(v)
                for s in new_sums:
                    used_sums.discard(s)
                used_sums.discard(v + v)

            return False

        if backtrack(len(prescribed)):
            return assignment
        return None

    # ---------- SA fallback ----------
    class State:
        __slots__ = ('arr', 'n', 'counts', 'conflict', 'val_counts', 'fixed')

        def __init__(self, arr, fixed):
            self.arr = arr
            self.n = len(arr)
            self.fixed = fixed
            self.counts = {}
            self.conflict = 0
            self.val_counts = {}
            used = set(arr)
            for i in range(self.n):
                vi = arr[i]
                self.val_counts[vi] = self.val_counts.get(vi, 0) + 1
                for j in range(i, self.n):
                    s = vi + arr[j]
                    c = self.counts.get(s, 0)
                    if c == 1:
                        self.conflict += 1
                    self.counts[s] = c + 1

        def _inc_sum(self, s):
            c = self.counts.get(s, 0)
            if c == 1:
                self.conflict += 1
            self.counts[s] = c + 1

        def _dec_sum(self, s):
            c = self.counts[s]
            if c == 2:
                self.conflict -= 1
            if c == 1:
                del self.counts[s]
            else:
                self.counts[s] = c - 1

        def propose(self, i, new_val):
            old = self.arr[i]
            if old == new_val:
                return None
            if self.val_counts.get(new_val, 0) >= 1:
                return None
            for j in range(self.n):
                self._dec_sum(old + self.arr[j])
            self.val_counts[old] -= 1
            if self.val_counts[old] == 0:
                del self.val_counts[old]
            self.arr[i] = new_val
            for j in range(self.n):
                self._inc_sum(new_val + self.arr[j])
            self.val_counts[new_val] = self.val_counts.get(new_val, 0) + 1
            return self.conflict

        def revert(self, i, old_val, new_val):
            for j in range(self.n):
                self._dec_sum(new_val + self.arr[j])
            self.val_counts[new_val] -= 1
            if self.val_counts[new_val] == 0:
                del self.val_counts[new_val]
            self.arr[i] = old_val
            for j in range(self.n):
                self._inc_sum(old_val + self.arr[j])
            self.val_counts[old_val] = self.val_counts.get(old_val, 0) + 1

    def run_sa(target, init, budget, T_start=None, T_end=None, cool=None):
        end = min(time.time() + budget, DEADLINE - 2.0)
        fixed = [v in pres_set for v in init]
        st = State(init, fixed)
        best_obj = st.conflict
        best_arr = list(init)

        if T_start is None:
            T_start = max(0.5, best_obj * 0.3)
        if T_end is None:
            T_end = 0.005
        if cool is None:
            cool = 0.9997

        T = T_start
        n = target
        no_improve_count = 0
        improve_threshold = 300

        while time.time() < end and T > T_end:
            move_type = random.random()
            if move_type < 0.75:
                i = random.randrange(n)
                if fixed[i]:
                    continue
                old = st.arr[i]
                cur = st.conflict

                # Find good new value
                new = None
                best_delta = float('inf')
                for _ in range(50):
                    trial = random.randint(1, N)
                    if trial == old or st.val_counts.get(trial, 0) >= 1:
                        continue
                    delta = 0
                    for j in range(n):
                        s_new = trial + st.arr[j]
                        s_old = old + st.arr[j]
                        c_old = st.counts.get(s_old, 0)
                        c_new = st.counts.get(s_new, 0)
                        if c_old == 2:
                            delta -= 1
                        if c_new == 1:
                            delta += 1
                    if delta < best_delta:
                        best_delta = delta
                        new = trial
                        if delta < 0:
                            break

                if new is None:
                    new = random.randint(1, N)

                nc = st.propose(i, new)
                if nc is None:
                    continue

                if nc <= cur:
                    if nc < best_obj:
                        best_obj = nc
                        best_arr = list(st.arr)
                        no_improve_count = 0
                        if nc == 0:
                            return best_arr
                    else:
                        no_improve_count += 1
                else:
                    if random.random() < math.exp((cur - nc) / T):
                        no_improve_count += 1
                    else:
                        st.revert(i, old, new)
                        no_improve_count += 1
            else:
                i = random.randrange(n)
                j = random.randrange(n)
                if fixed[i] or fixed[j] or i == j:
                    continue
                old_i = st.arr[i]
                old_j = st.arr[j]
                cur = st.conflict

                for k in range(n):
                    st._dec_sum(old_i + st.arr[k])
                    st._dec_sum(old_j + st.arr[k])

                st.val_counts[old_i] -= 1
                st.val_counts[old_j] -= 1
                if st.val_counts[old_i] == 0:
                    del st.val_counts[old_i]
                if st.val_counts[old_j] == 0:
                    del st.val_counts[old_j]

                st.arr[i] = old_j
                st.arr[j] = old_i

                for k in range(n):
                    st._inc_sum(st.arr[i] + st.arr[k])
                    st._inc_sum(st.arr[j] + st.arr[k])

                st.val_counts[old_j] = st.val_counts.get(old_j, 0) + 1
                st.val_counts[old_i] = st.val_counts.get(old_i, 0) + 1

                nc = st.conflict

                if nc <= cur:
                    if nc < best_obj:
                        best_obj = nc
                        best_arr = list(st.arr)
                        no_improve_count = 0
                        if nc == 0:
                            return best_arr
                    else:
                        no_improve_count += 1
                else:
                    if random.random() < math.exp((cur - nc) / T):
                        no_improve_count += 1
                    else:
                        for k in range(n):
                            st._dec_sum(st.arr[i] + st.arr[k])
                            st._dec_sum(st.arr[j] + st.arr[k])

                        st.val_counts[old_j] -= 1
                        st.val_counts[old_i] -= 1
                        if st.val_counts[old_j] == 0:
                            del st.val_counts[old_j]
                        if st.val_counts[old_i] == 0:
                            del st.val_counts[old_i]

                        st.arr[i] = old_i
                        st.arr[j] = old_j

                        for k in range(n):
                            st._inc_sum(old_i + st.arr[k])
                            st._inc_sum(old_j + st.arr[k])

                        st.val_counts[old_i] = st.val_counts.get(old_i, 0) + 1
                        st.val_counts[old_j] = st.val_counts.get(old_j, 0) + 1

                        no_improve_count += 1

            T *= cool

            if no_improve_count > improve_threshold and best_obj > 0:
                break

        if best_obj == 0:
            return best_arr
        return None

    try:
        g1 = greedy_build()
        g2 = greedy_build_reversed()
        g3 = greedy_build_spread()
        best = max([g1, g2, g3], key=len)
        best_size = len(best)

        # Try CSP for larger sizes
        for target in range(best_size + 1, 21):
            if remaining() < 20:
                break

            # Give CSP a reasonable budget
            csp_budget = min(40.0, remaining() - 10)
            result = csp_solve(target, csp_budget)

            if result is not None and len(result) == target and is_valid_set(result):
                best = result
                best_size = target
                continue
            else:
                # Try SA as fallback
                sa_budget = min(remaining() - 5, 60.0)
                if sa_budget > 5:
                    # Create initial state
                    base = list(best)
                    used = set(base)
                    need = target - len(base)
                    if need > 0:
                        pool = [x for x in range(1, N + 1) if x not in used]
                        random.shuffle(pool)
                        init = base + pool[:need]
                    else:
                        init = base

                    found = run_sa(target, init, sa_budget)
                    if found is not None and is_valid_set(found):
                        best = found
                        best_size = target
                    break

        # Final validation
        if not is_valid_set(best):
            best = greedy_build()

        return best

    except Exception:
        try:
            return greedy_build()
        except Exception:
            return [1, 2, 5, 11]
