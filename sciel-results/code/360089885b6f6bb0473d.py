import time
import random
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    random.seed(42)

    def gen_frame(used, max_val, rng):
        """Generate a single row (frame) greedily, avoiding used differences."""
        row = [0]
        while len(row) < k + 1:
            x = rng.randint(1, max_val)
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        return row

    def scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        seen = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def try_repair(rows, max_val, rng, used_set):
        """Attempt to repair invalid rows by regenerating conflicting ones."""
        # Find which rows have conflicts
        seen = set()
        conflicts = set()
        for idx, r in enumerate(rows):
            row_diffs = set()
            valid = True
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen or d in row_diffs:
                        valid = False
                        conflicts.add(idx)
                    row_diffs.add(d)
            if valid:
                seen |= row_diffs
        if not conflicts:
            return rows, used_set
        # Regenerate conflicting rows
        new_rows = list(rows)
        new_used = set()
        for idx, r in enumerate(new_rows):
            if idx not in conflicts:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        new_used.add(r[j] - r[i])
        for idx in sorted(conflicts):
            new_rows[idx] = gen_frame(new_used, max_val, rng)
        return new_rows, new_used

    # Initial solution: greedy construction
    def initial_solution(max_val, rng):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = gen_frame(used, max_val, rng)
            rows.append(row)
        return rows

    # Start with a reasonable max_val
    max_val = 112  # best known
    rng = random.Random(42)
    best_rows = initial_solution(max_val, rng)
    best_scope = scope(best_rows)

    # Try to reduce max_val
    for mv in range(max_val, 104, -1):
        if time.time() > deadline - 5:
            break
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            rows = initial_solution(mv, rng)
            if is_valid(rows):
                best_rows = rows
                best_scope = mv
                break
        if best_scope < max_val:
            max_val = best_scope

    # Simulated annealing to reduce scope
    if time.time() < deadline - 10:
        temp = max_val * 0.5
        cool_rate = 0.999
        current_rows = list(best_rows)
        current_scope = best_scope
        no_improve = 0

        while time.time() < deadline - 5 and temp > 0.1:
            # Choose a move
            move_type = rng.random()
            new_rows = list(current_rows)

            if move_type < 0.4:
                # Regenerate one random row
                idx = rng.randint(0, n_rows - 1)
                # Remove its differences from used
                used = set()
                for i, r in enumerate(new_rows):
                    if i != idx:
                        for a in range(len(r)):
                            for b in range(a + 1, len(r)):
                                used.add(r[b] - r[a])
                new_rows[idx] = gen_frame(used, current_scope, rng)
            elif move_type < 0.6:
                # Swap two rows
                i, j = rng.sample(range(n_rows), 2)
                new_rows[i], new_rows[j] = new_rows[j], new_rows[i]
            elif move_type < 0.8:
                # Regenerate two random rows
                i, j = rng.sample(range(n_rows), 2)
                used = set()
                for idx, r in enumerate(new_rows):
                    if idx != i and idx != j:
                        for a in range(len(r)):
                            for b in range(a + 1, len(r)):
                                used.add(r[b] - r[a])
                new_rows[i] = gen_frame(used, current_scope, rng)
                new_rows[j] = gen_frame(used, current_scope, rng)
            else:
                # Perturb: shift a random row by a small amount and re-greedy
                idx = rng.randint(0, n_rows - 1)
                shift = rng.randint(-3, 3)
                r = new_rows[idx]
                new_r = [max(0, x + shift) for x in r]
                # Ensure strictly increasing and starts at 0
                if new_r[0] != 0:
                    new_r[0] = 0
                valid_strict = all(new_r[i] < new_r[i+1] for i in range(len(new_r)-1))
                if valid_strict and len(new_r) == k + 1:
                    new_rows[idx] = new_r
                # If invalid, just regenerate
                else:
                    used = set()
                    for ii, rr in enumerate(new_rows):
                        if ii != idx:
                            for a in range(len(rr)):
                                for b in range(a + 1, len(rr)):
                                    used.add(rr[b] - rr[a])
                    new_rows[idx] = gen_frame(used, current_scope, rng)

            # Check validity
            if not is_valid(new_rows):
                # Try repair
                new_rows, _ = try_repair(new_rows, current_scope, rng, set())
                if not is_valid(new_rows):
                    temp *= cool_rate
                    continue

            new_scope = scope(new_rows)
            delta = new_scope - current_scope

            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(temp, 0.01))):
                current_rows = new_rows
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = list(current_rows)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            temp *= cool_rate

    # Final validation
    if not is_valid(best_rows):
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_rows
