import time
import random
import math

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 112.0  # Safety margin for 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            if len(row) != row_len:
                return None
            if row[0] != 0:
                return None
            for i in range(len(row)):
                ri = row[i]
                if i > 0 and ri <= row[i-1]:
                    return None
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    def compute_all_diffs(rows):
        """Compute the set of all differences in the configuration."""
        all_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.add(r[j] - r[i])
        return all_diffs

    def get_remove_set(rows, r_idx, c_idx):
        """Get the set of differences involving position (r_idx, c_idx)."""
        row = rows[r_idx]
        remove_set = set()
        curr_val = row[c_idx]
        for i in range(c_idx):
            remove_set.add(curr_val - row[i])
        for j in range(c_idx + 1, len(row)):
            remove_set.add(row[j] - curr_val)
        return remove_set

    def check_move_valid(rows, r_idx, c_idx, new_val, fixed_diffs):
        """Check if placing new_val at (r_idx, c_idx) is valid given fixed_diffs."""
        row = rows[r_idx]
        # Check strictly increasing
        if c_idx > 0 and new_val <= row[c_idx - 1]:
            return False
        if c_idx < len(row) - 1 and new_val >= row[c_idx + 1]:
            return False
        
        new_left_diffs = [new_val - row[i] for i in range(c_idx)]
        new_right_diffs = [row[j] - new_val for j in range(c_idx + 1, len(row))]
        
        # Check for duplicates within new diffs
        all_new = new_left_diffs + new_right_diffs
        if len(set(all_new)) != len(all_new):
            return False
        
        # Check against fixed_diffs
        for d in all_new:
            if d in fixed_diffs:
                return False
        
        return True

    def targeted_descent(rows, budget):
        """
        Structural change: Instead of generic local search, this focuses specifically
        on reducing the scope by shrinking the largest mark in the row that has it.
        Uses incremental difference sets for speed.
        """
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False
        all_diffs = compute_all_diffs(cur)

        while time.time() - start < budget:
            # Find the row(s) containing the current max
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            # The last element is the max of this row
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                continue

            # Try a set of candidate values: prefer values just above the lower bound
            # to maximize the reduction, but also sample a few random ones
            candidates = set()
            # Sequential near the bottom (best reduction)
            for v in range(lower, min(lower + 15, old_val)):
                candidates.add(v)
            # Some random samples in the middle
            for _ in range(10):
                candidates.add(rng.randint(lower, old_val - 1))

            remove_set = get_remove_set(cur, r, pos)
            fixed_diffs = all_diffs - remove_set

            for new_val in sorted(candidates, key=lambda x: -x):  # try larger reductions first
                if new_val >= old_val:
                    continue
                
                if check_move_valid(cur, r, pos, new_val, fixed_diffs):
                    # Apply move
                    row[pos] = new_val
                    # Update all_diffs
                    all_diffs -= remove_set
                    # Add new diffs
                    for i in range(pos):
                        all_diffs.add(new_val - row[i])
                    
                    new_scope = get_scope(cur)
                    if new_scope < cur_scope:
                        cur_scope = new_scope
                        improved = True
                        break
                    else:
                        # Revert
                        row[pos] = old_val
                        all_diffs -= {new_val - row[i] for i in range(pos)}
                        all_diffs |= remove_set

            if not improved:
                # Try reducing a non-last element in the max row
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                
                remove_set2 = get_remove_set(cur, r, pos2)
                fixed_diffs2 = all_diffs - remove_set2
                
                for _ in range(5):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    
                    if check_move_valid(cur, r, pos2, nv, fixed_diffs2):
                        row[pos2] = nv
                        all_diffs -= remove_set2
                        for i in range(pos2):
                            all_diffs.add(nv - row[i])
                        for j in range(pos2 + 1, len(row)):
                            all_diffs.add(row[j] - nv)
                        
                        new_scope = get_scope(cur)
                        if new_scope < cur_scope:
                            cur_scope = new_scope
                            improved = True
                            break
                        else:
                            # Revert
                            row[pos2] = old2
                            all_diffs -= {nv - row[i] for i in range(pos2)}
                            all_diffs -= {row[j] - nv for j in range(pos2 + 1, len(row))}
                            all_diffs |= remove_set2
                
                if not improved:
                    break

        if improved:
            return cur, cur_scope
        return None, None

    # Simulated annealing with restarts
    max_restarts = 300
    steps_per_restart = 80000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Start from best or perturbed
        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            # Perturb: random row, random mark, random new value
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 10
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Annealing
        T = max(0.05, 1.0 - restart / max_restarts)
        all_diffs = compute_all_diffs(cur)

        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 20
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            # Compute remove set and fixed diffs
            remove_set = get_remove_set(cur, ri, pos)
            fixed_diffs = all_diffs - remove_set
            
            if check_move_valid(cur, ri, pos, new_val, fixed_diffs):
                # Apply move
                cur[ri][pos] = new_val
                all_diffs -= remove_set
                for i in range(pos):
                    all_diffs.add(new_val - cur[ri][i])
                for j in range(pos + 1, len(cur[ri])):
                    all_diffs.add(cur[ri][j] - new_val)
                
                new_scope = get_scope(cur)
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                else:
                    # Revert
                    cur[ri][pos] = old_val
                    all_diffs -= {new_val - cur[ri][i] for i in range(pos)}
                    all_diffs -= {cur[ri][j] - new_val for j in range(pos + 1, len(cur[ri]))}
                    all_diffs |= remove_set
            else:
                # Invalid: reject (no change to state)
                pass

            T *= 0.99995

        # After annealing, run a short targeted descent to try to squeeze scope further
        if time.time() < deadline - 5.0:
            desc_rows, desc_scope = targeted_descent(cur, budget=2.0)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
