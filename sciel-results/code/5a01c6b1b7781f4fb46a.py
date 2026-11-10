import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin vs 120s kill

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))
    
    # Estimate max scope for bytearray sizing
    # Best known for (4,7) is 134. Lower bound is ~15.
    # Let's start with a reasonable buffer.
    initial_ba_size = lower_bound + 300

    def validate(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_diffs_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    used.add(row[j] - ri)
        return used

    def scope_of(rows):
        return max(max(r) for r in rows)

    # Helper to manage dynamic bytearray
    class DiffTracker:
        def __init__(self, initial_size):
            self.ba = bytearray(initial_size)
        
        def ensure_size(self, idx):
            if idx >= len(self.ba):
                # Extend by 50% or to idx+1
                new_size = max(idx + 1, int(len(self.ba) * 1.5))
                self.ba.extend(b'\x00' * (new_size - len(self.ba)))
        
        def is_used(self, idx):
            if idx >= len(self.ba):
                return False
            return self.ba[idx] == 1
        
        def mark_used(self, idx):
            self.ensure_size(idx)
            self.ba[idx] = 1
        
        def unmark_used(self, idx):
            if idx < len(self.ba):
                self.ba[idx] = 0
        
        def copy(self):
            new_tracker = DiffTracker(len(self.ba))
            new_tracker.ba = bytearray(self.ba)
            return new_tracker

    # ---- MRV interleaved construction (structural core) ----
    def build_interleaved(seed):
        random.seed(seed)
        rows = [[0] for _ in range(n_rows)]
        tracker = DiffTracker(initial_ba_size)
        total_steps = n_rows * k
        steps = 0
        current_max = 0

        while steps < total_steps:
            if time.time() > deadline - 5:
                return None
            candidates = []
            for r in range(n_rows):
                if len(rows[r]) == k + 1:
                    continue
                cur = rows[r]
                last = cur[-1]
                valid_xs = []
                x = last + 1
                checks = 0
                # Wider candidate window: up to 20 valid or 500 checks
                while len(valid_xs) < 20 and checks < 500:
                    if tracker.is_used(x):
                        x += 1
                        checks += 1
                        continue
                    conflict = False
                    for a in cur[1:]:
                        if tracker.is_used(x - a):
                            conflict = True
                            break
                    if not conflict:
                        valid_xs.append(x)
                    x += 1
                    checks += 1
                if not valid_xs:
                    return None
                candidates.append((valid_xs[0], r, valid_xs))

            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])

            # Pick the most constrained (smallest feasible) with slight randomness
            if random.random() < 0.75:
                chosen_idx = 0
            else:
                chosen_idx = random.randint(0, min(5, len(candidates) - 1))
            _, r, valid_xs = candidates[chosen_idx]

            # Choose a value; mostly the tightest, occasionally explore
            if random.random() < 0.70:
                chosen_x = valid_xs[0]
            else:
                chosen_x = valid_xs[random.randint(0, min(5, len(valid_xs) - 1))]

            rows[r].append(chosen_x)
            for a in rows[r][:-1]:
                tracker.mark_used(chosen_x - a)
            if chosen_x > current_max:
                current_max = chosen_x
            steps += 1

        if not validate(rows):
            return None
        return rows

    # ---- Epigraph local search with explicit scope variable t ----
    def local_search(rows, time_budget):
        rows = [r[:] for r in rows]
        # Use DiffTracker for global state
        max_val = scope_of(rows)
        tracker = DiffTracker(max_val + 100)
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    tracker.mark_used(row[j] - ri)
        
        t = scope_of(rows)  # auxiliary variable t = scope
        best_rows = [r[:] for r in rows]
        best_t = t
        end = time.time() + time_budget
        temp = max(1.0, (t - lower_bound) / 8.0)
        no_improve_count = 0
        max_no_improve = 500

        while time.time() < end:
            if no_improve_count > max_no_improve:
                break
            r = random.randint(0, n_rows - 1)
            row = rows[r]
            if len(row) < 2:
                continue
            idx = random.randint(1, len(row) - 1)
            # Wider delta window for better exploration
            deltas = list(range(-40, 41))
            random.shuffle(deltas)
            improved = False
            for delta in deltas:
                if delta == 0:
                    continue
                new_val = row[idx] + delta
                if new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
                
                # Calculate old and new diffs
                old_diffs = []
                new_diffs = []
                for j in range(idx):
                    old_diffs.append(row[idx] - row[j])
                    new_diffs.append(new_val - row[j])
                for j in range(idx + 1, len(row)):
                    old_diffs.append(row[j] - row[idx])
                    new_diffs.append(row[j] - new_val)
                
                # Check conflicts: new_diffs must not collide with global used, 
                # except for those in old_diffs which we are removing.
                
                # First, mark old_diffs as "to be removed" logically? 
                # Easier: Check if any new_diff is used in tracker AND not in old_diffs.
                
                # To speed up, we can check new_diffs against tracker.
                # If a new_diff is in old_diffs, it's okay (it will be unmarked).
                # If a new_diff is NOT in old_diffs, it must NOT be in tracker.
                
                old_set = set(old_diffs)
                conflict = False
                for d in new_diffs:
                    if d in old_set:
                        continue
                    if tracker.is_used(d):
                        conflict = True
                        break
                        
                if conflict:
                    continue
                
                # If no conflict, apply move
                new_t = max(t, new_val)
                # Accept if t improves, or t equal with small probability
                if new_t < t or (new_t == t and random.random() < 0.15):
                    # Unmark old
                    for d in old_diffs:
                        tracker.unmark_used(d)
                    # Mark new
                    for d in new_diffs:
                        tracker.mark_used(d)
                    
                    rows[r][idx] = new_val
                    t = new_t
                    if t < best_t:
                        best_t = t
                        best_rows = [rr[:] for rr in rows]
                    improved = True
                    no_improve_count = 0
                    break
            if not improved:
                no_improve_count += 1
                temp *= 0.999

        return best_rows, best_t

    # ---- High-precision polish pass (tighter deltas, more iterations) ----
    def polish(rows, time_budget):
        rows = [r[:] for r in rows]
        max_val = scope_of(rows)
        tracker = DiffTracker(max_val + 100)
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    tracker.mark_used(row[j] - ri)

        t = scope_of(rows)
        best_rows = [r[:] for r in rows]
        best_t = t
        end = time.time() + time_budget
        no_improve_count = 0
        max_no_improve = 300

        while time.time() < end:
            if no_improve_count > max_no_improve:
                break
            r = random.randint(0, n_rows - 1)
            row = rows[r]
            if len(row) < 2:
                continue
            idx = random.randint(1, len(row) - 1)
            # Tighter delta window for precision
            deltas = list(range(-8, 9))
            random.shuffle(deltas)
            improved = False
            for delta in deltas:
                if delta == 0:
                    continue
                new_val = row[idx] + delta
                if new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
                
                old_diffs = []
                new_diffs = []
                for j in range(idx):
                    old_diffs.append(row[idx] - row[j])
                    new_diffs.append(new_val - row[j])
                for j in range(idx + 1, len(row)):
                    old_diffs.append(row[j] - row[idx])
                    new_diffs.append(row[j] - new_val)
                
                old_set = set(old_diffs)
                conflict = False
                for d in new_diffs:
                    if d in old_set:
                        continue
                    if tracker.is_used(d):
                        conflict = True
                        break
                        
                if conflict:
                    continue
                
                new_t = max(t, new_val)
                if new_t < t:
                    for d in old_diffs:
                        tracker.unmark_used(d)
                    for d in new_diffs:
                        tracker.mark_used(d)
                    
                    rows[r][idx] = new_val
                    t = new_t
                    if t < best_t:
                        best_t = t
                        best_rows = [rr[:] for rr in rows]
                    improved = True
                    no_improve_count = 0
                    break
            if not improved:
                no_improve_count += 1
        return best_rows, best_t

    # ---- Fallback simple greedy (row-by-row) ----
    def greedy_fallback():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
        return rows

    best_rows = None
    best_scope = float('inf')

    # Phase 1: many interleaved MRV constructions
    seed = 1
    while time.time() < deadline - 35:
        rows = build_interleaved(seed)
        if rows is not None:
            s = scope_of(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
        seed += 1

    if best_rows is None:
        best_rows = greedy_fallback()
        best_scope = scope_of(best_rows)

    # Phase 2: epigraph local search on the best found
    if time.time() < deadline - 15:
        ls_rows, ls_scope = local_search(best_rows, max(0.0, deadline - 15 - time.time()))
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope

    # Phase 3: high-precision polish pass
    if time.time() < deadline - 2:
        pl_rows, pl_scope = polish(best_rows, max(0.0, deadline - 2 - time.time()))
        if pl_scope < best_scope:
            best_rows = pl_rows
            best_scope = pl_scope

    return best_rows
