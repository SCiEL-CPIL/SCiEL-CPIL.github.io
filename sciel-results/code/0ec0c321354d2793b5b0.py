import random
import time
import math
from typing import List, Set, Optional

# EVOLVE-BLOCK-START

def construct(n_rows: int, k: int) -> List[List[int]]:
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimizes the largest entry (scope).
    
    Uses bitset-based difference tracking for fast constraint checking
    within a simulated annealing local search framework.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def time_up() -> bool:
        return time.time() > deadline

    def get_diffs_bitset(row: List[int]) -> int:
        """Compute the bitset of differences for a row."""
        bits = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                bits |= (1 << d)
        return bits

    def get_max(rows: List[List[int]]) -> int:
        return max(max(r) for r in rows)

    # Initial greedy construction to get a valid starting point
    # We use a simple greedy approach with bitsets for speed
    def greedy_construct(seed: int) -> Optional[tuple]:
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        
        # Estimate max_val to avoid infinite loops in greedy
        # Counting bound is n_rows * k*(k+1)/2. Optimal is higher.
        # For (4,7), optimal is 134.
        max_val = max(150, k * (k + 1) // 2 + n_rows * 40 + 50)
        
        def build_row(row, used_bits, max_val_limit):
            if len(row) == k + 1:
                return True
            last = row[-1]
            candidates = []
            for x in range(last + 1, max_val_limit + 1):
                new_diff_bits = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if not conflict:
                    candidates.append((x, new_diff_bits))
                if len(candidates) > 20:
                    break
            
            if not candidates:
                return False
            
            rng.shuffle(candidates)
            for x, new_diff_bits in candidates:
                row.append(x)
                if build_row(row, used_bits | new_diff_bits, max_val_limit):
                    return True
                row.pop()
            return False

        for _ in range(n_rows):
            row = [0]
            if not build_row(row, used_bits, max_val):
                return None
            rows.append(row)
            used_bits |= get_diffs_bitset(row)

        return rows, used_bits

    # Find initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used_bits = 0

    for seed in range(10):
        if time_up():
            break
        res = greedy_construct(seed)
        if res:
            rows, used_bits = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = rows
                current_used_bits = used_bits

    if best_rows is None:
        # Fallback: very loose greedy
        for seed in range(50):
            if time_up():
                break
            # Increase max_val for fallback
            max_val_fallback = 300
            used_bits = 0
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                last = 0
                while len(row) < k + 1:
                    found = False
                    for x in range(last + 1, max_val_fallback + 1):
                        new_diff_bits = 0
                        conflict = False
                        for a in row:
                            d = x - a
                            if (used_bits >> d) & 1:
                                conflict = True
                                break
                            new_diff_bits |= (1 << d)
                        if not conflict:
                            row.append(x)
                            used_bits |= new_diff_bits
                            last = x
                            found = True
                            break
                    if not found:
                        success = False
                        break
                if not success:
                    break
                rows.append(row)
            if success:
                best_rows = rows
                current_rows = rows
                current_used_bits = used_bits
                best_max = get_max(rows)
                break

    if best_rows is None:
        return []

    current_max = best_max
    rng = random.Random(42)

    # Rebuild a single row with pressure toward compactness
    # Uses bitsets for fast conflict checking
    def rebuild_row(row_k, forbidden_bits, rng, max_val_limit, pressure=1.0):
        row = [0]
        local_used_bits = 0

        for step in range(row_k):
            last = row[-1]
            valid_candidates = []

            # We can limit the search range slightly by noting that
            # the remaining elements must fit within max_val_limit
            # But for simplicity and safety, we scan up to max_val_limit
            # The bitset check is O(len(row)) which is small (k+1 <= 8)
            
            for x in range(last + 1, max_val_limit + 1):
                new_diff_bits = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (forbidden_bits >> d) & 1 or (local_used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if not conflict:
                    # Score candidate
                    # Prefer smaller x (compactness) and smaller max diff?
                    # The goal is to minimize the global max.
                    # A good heuristic is to prefer x that doesn't force future
                    # elements to be large.
                    # Here we just use x as a proxy, weighted by pressure.
                    score = x * (1.0 - pressure) + (x - last) * pressure
                    valid_candidates.append((x, new_diff_bits, score))
                
                # Optimization: If we have many candidates, we can stop early
                # if we are in "greedy" mode (high pressure) and x is already large
                if len(valid_candidates) > 50:
                    break

            if not valid_candidates:
                return None

            # Sort by score
            valid_candidates.sort(key=lambda t: t[2])
            n_top = min(10, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]

            # Probabilistic selection
            weights = [1.0 / (1.0 + c[2] * 0.03) for c in top_candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = top_candidates[0]
            for c, w in zip(top_candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c
                    break

            x, new_diff_bits, _ = chosen
            row.append(x)
            local_used_bits |= new_diff_bits

        return row

    # Simulated annealing
    initial_temp = 1.5
    cooling_rate = 0.998
    min_temp = 0.003
    temp = initial_temp
    no_improve_count = 0

    while not time_up():
        elapsed = time.time() - start_time
        remaining = deadline - elapsed

        polish_mode = remaining < 15.0

        if polish_mode:
            target_max = current_max - 1
            accept_neutral = False
            pressure = 1.0
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.25 if temp > 0.1 else 0.1)
            pressure = 0.6 + 0.4 * temp

        # Occasionally rebuild two rows together
        use_multi = (not polish_mode) and (rng.random() < 0.2) and (n_rows >= 2)

        if use_multi:
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                j = (j + 1) % n_rows

            old_row_i = current_rows[i]
            old_row_j = current_rows[j]
            old_diffs_i_bits = get_diffs_bitset(old_row_i)
            old_diffs_j_bits = get_diffs_bitset(old_row_j)

            # Remove old diffs from global used bits
            # Note: A & ~B is not bitwise subtraction, but since bits are unique,
            # we can use XOR if we know they are set, or just & ~
            current_used_bits &= ~old_diffs_i_bits
            current_used_bits &= ~old_diffs_j_bits

            new_row_i = rebuild_row(k, current_used_bits, rng, target_max if target_max > k else current_max, pressure)
            if new_row_i is None:
                new_row_i = rebuild_row(k, current_used_bits, rng, current_max, pressure)

            if new_row_i is None:
                # Restore
                current_used_bits |= old_diffs_i_bits
                current_used_bits |= old_diffs_j_bits
                no_improve_count += 1
                temp *= cooling_rate
                continue

            diffs_i_new_bits = get_diffs_bitset(new_row_i)
            # Check if new_row_i conflicts with remaining used bits
            if diffs_i_new_bits & current_used_bits:
                current_used_bits |= old_diffs_i_bits
                current_used_bits |= old_diffs_j_bits
                no_improve_count += 1
                temp *= cooling_rate
                continue

            # Update used bits with new_row_i
            current_used_bits |= diffs_i_new_bits

            new_row_j = rebuild_row(k, current_used_bits, rng, target_max if target_max > k else current_max, pressure)
            if new_row_j is None:
                new_row_j = rebuild_row(k, current_used_bits, rng, current_max, pressure)

            if new_row_j is None:
                # Rollback new_row_i
                current_used_bits &= ~diffs_i_new_bits
                # Restore old
                current_used_bits |= old_diffs_i_bits
                current_used_bits |= old_diffs_j_bits
                no_improve_count += 1
                temp *= cooling_rate
                continue

            diffs_j_new_bits = get_diffs_bitset(new_row_j)
            # Check conflict
            if diffs_j_new_bits & current_used_bits:
                # Rollback new_row_i
                current_used_bits &= ~diffs_i_new_bits
                # Restore old
                current_used_bits |= old_diffs_i_bits
                current_used_bits |= old_diffs_j_bits
                no_improve_count += 1
                temp *= cooling_rate
                continue

            # Accept
            current_rows[i] = new_row_i
            current_rows[j] = new_row_j
            current_used_bits |= diffs_j_new_bits
            new_global_max = get_max(current_rows)

            if new_global_max < current_max:
                current_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
                temp = max(temp * 1.2, min_temp)
            else:
                if accept_neutral:
                    no_improve_count += 1
                else:
                    # Rollback
                    current_used_bits &= ~diffs_i_new_bits
                    current_used_bits &= ~diffs_j_new_bits
                    current_used_bits |= old_diffs_i_bits
                    current_used_bits |= old_diffs_j_bits
                    current_rows[i] = old_row_i
                    current_rows[j] = old_row_j
                    no_improve_count += 1

            temp *= cooling_rate

        else:
            i = rng.randint(0, n_rows - 1)
            old_row = current_rows[i]
            old_diffs_bits = get_diffs_bitset(old_row)

            current_used_bits &= ~old_diffs_bits

            new_row = rebuild_row(k, current_used_bits, rng, target_max if target_max > k else current_max, pressure)
            if new_row is None:
                new_row = rebuild_row(k, current_used_bits, rng, current_max, pressure)

            if new_row is None:
                current_used_bits |= old_diffs_bits
                no_improve_count += 1
                temp *= cooling_rate
                continue

            new_diffs_bits = get_diffs_bitset(new_row)
            if new_diffs_bits & current_used_bits:
                current_used_bits |= old_diffs_bits
                no_improve_count += 1
                temp *= cooling_rate
                continue

            current_rows[i] = new_row
            current_used_bits |= new_diffs_bits
            new_global_max = get_max(current_rows)

            if new_global_max < current_max:
                current_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
                temp = max(temp * 1.2, min_temp)
            else:
                if accept_neutral:
                    no_improve_count += 1
                else:
                    current_used_bits &= ~new_diffs_bits
                    current_used_bits |= old_diffs_bits
                    current_rows[i] = old_row
                    no_improve_count += 1

            temp *= cooling_rate

    # Final polish: aggressive single-row reduction
    if not time_up():
        polish_deadline = min(deadline, time.time() + 8.0)
        while time.time() < polish_deadline:
            improved = False
            for i in range(n_rows):
                old_row = current_rows[i]
                old_diffs_bits = get_diffs_bitset(old_row)
                current_used_bits &= ~old_diffs_bits

                target_max = current_max - 1
                if target_max > k:
                    new_row = rebuild_row(k, current_used_bits, rng, target_max, 1.0)
                    if new_row is not None:
                        new_diffs_bits = get_diffs_bitset(new_row)
                        if not (new_diffs_bits & current_used_bits):
                            current_rows[i] = new_row
                            current_used_bits |= new_diffs_bits
                            current_max = get_max(current_rows)
                            best_rows = [r[:] for r in current_rows]
                            improved = True
                            break
                        else:
                            current_used_bits |= old_diffs_bits
                    else:
                        current_used_bits |= old_diffs_bits
                else:
                    current_used_bits |= old_diffs_bits

            if not improved:
                break

    return best_rows

# EVOLVE-BLOCK-END
