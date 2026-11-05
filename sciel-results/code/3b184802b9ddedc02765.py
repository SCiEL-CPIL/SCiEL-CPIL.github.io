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
    
    Uses bitset-based difference tracking with a multi-phase local search:
    1. Greedy initialization with multiple seeds
    2. Simulated annealing with adaptive temperature
    3. Aggressive polish pass
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

    # Counting bound: n_rows * k*(k+1)/2 distinct differences needed
    # Each row has k*(k+1)/2 differences, so total is n_rows * k*(k+1)/2
    # The max difference in a row with max entry M is at most M
    # So we need M >= n_rows * k*(k+1)/2 (rough lower bound)
    # For (4,7): 4 * 28 = 112, best known is 134
    
    # Initial greedy construction
    def greedy_construct(seed: int, max_val_limit: int) -> Optional[List[List[int]]]:
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        
        def build_row(row, used_bits):
            if len(row) == k + 1:
                return True
            last = row[-1]
            candidates = []
            
            # Try candidates in increasing order with some randomization
            # Limit search to avoid infinite loops
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
                if len(candidates) > 30:
                    break
            
            if not candidates:
                return False
            
            # Random selection from candidates, weighted toward smaller values
            # Use exponential weighting for compactness
            weights = [math.exp(-x / 50.0) for x, _ in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = candidates[0]
            for c, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c
                    break
            
            x, new_diff_bits = chosen
            row.append(x)
            if build_row(row, used_bits | new_diff_bits):
                return True
            row.pop()
            return False

        for _ in range(n_rows):
            row = [0]
            if not build_row(row, used_bits):
                return None
            rows.append(row)
            used_bits |= get_diffs_bitset(row)

        return rows

    # Find initial best solution with multiple seeds and increasing max_val
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used_bits = 0

    # Start with a reasonable max_val estimate
    base_max = max(120, n_rows * k * (k + 1) // 2 + 20)
    
    for seed in range(20):
        if time_up():
            break
        # Use a slightly larger max_val for early seeds to ensure success
        max_val = base_max + seed * 5
        res = greedy_construct(seed, max_val)
        if res:
            m = get_max(res)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in res]
                current_rows = res
                current_used_bits = 0
                for row in current_rows:
                    current_used_bits |= get_diffs_bitset(row)

    if best_rows is None:
        # Fallback: very loose greedy with large max_val
        max_val_fallback = 400
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
        
        if not success:
            return []
        
        best_rows = rows
        current_rows = rows
        current_used_bits = used_bits
        best_max = get_max(rows)

    current_max = best_max
    rng = random.Random(12345)

    # Rebuild a single row with pressure toward compactness
    def rebuild_row(row_k, forbidden_bits, rng, max_val_limit, pressure=1.0):
        row = [0]
        local_used_bits = 0

        for step in range(row_k):
            last = row[-1]
            valid_candidates = []

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
                    # Score: prefer smaller x for compactness
                    score = x
                    valid_candidates.append((x, new_diff_bits, score))
                
                if len(valid_candidates) > 40:
                    break

            if not valid_candidates:
                return None

            # Sort by score
            valid_candidates.sort(key=lambda t: t[2])
            n_top = min(8, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]

            # Probabilistic selection with temperature-like behavior
            # Higher pressure means more uniform selection
            weights = []
            for c in top_candidates:
                # Exponential weighting based on score difference
                base_score = top_candidates[0][2]
                delta = c[2] - base_score
                w = math.exp(-delta * pressure * 0.05)
                weights.append(w)
            
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
    initial_temp = 2.0
    cooling_rate = 0.9995
    min_temp = 0.001
    temp = initial_temp
    no_improve_count = 0
    best_no_improve = 0

    while not time_up():
        elapsed = time.time() - start_time
        remaining = deadline - elapsed

        polish_mode = remaining < 20.0

        if polish_mode:
            target_max = current_max - 1
            accept_neutral = False
            pressure = 1.0
        else:
            target_max = current_max - 1
            # Accept neutral moves with probability based on temperature
            accept_neutral = rng.random() < (0.3 if temp > 0.2 else 0.15)
            pressure = 0.5 + 0.5 * min(temp, 1.0)

        # Occasionally rebuild two rows together for better coordination
        use_multi = (not polish_mode) and (rng.random() < 0.25) and (n_rows >= 2)

        if use_multi:
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            while j == i:
                j = rng.randint(0, n_rows - 1)

            old_row_i = current_rows[i]
            old_row_j = current_rows[j]
            old_diffs_i_bits = get_diffs_bitset(old_row_i)
            old_diffs_j_bits = get_diffs_bitset(old_row_j)

            # Remove old diffs from global used bits
            current_used_bits &= ~old_diffs_i_bits
            current_used_bits &= ~old_diffs_j_bits

            # Rebuild row i
            new_row_i = rebuild_row(k, current_used_bits, rng, 
                                    target_max if target_max > k else current_max, pressure)
            if new_row_i is None:
                new_row_i = rebuild_row(k, current_used_bits, rng, current_max, pressure)

            if new_row_i is None:
                current_used_bits |= old_diffs_i_bits
                current_used_bits |= old_diffs_j_bits
                no_improve_count += 1
                temp *= cooling_rate
                continue

            diffs_i_new_bits = get_diffs_bitset(new_row_i)
            if diffs_i_new_bits & current_used_bits:
                current_used_bits |= old_diffs_i_bits
                current_used_bits |= old_diffs_j_bits
                no_improve_count += 1
                temp *= cooling_rate
                continue

            current_used_bits |= diffs_i_new_bits

            # Rebuild row j
            new_row_j = rebuild_row(k, current_used_bits, rng,
                                    target_max if target_max > k else current_max, pressure)
            if new_row_j is None:
                new_row_j = rebuild_row(k, current_used_bits, rng, current_max, pressure)

            if new_row_j is None:
                current_used_bits &= ~diffs_i_new_bits
                current_used_bits |= old_diffs_i_bits
                current_used_bits |= old_diffs_j_bits
                no_improve_count += 1
                temp *= cooling_rate
                continue

            diffs_j_new_bits = get_diffs_bitset(new_row_j)
            if diffs_j_new_bits & current_used_bits:
                current_used_bits &= ~diffs_i_new_bits
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
                best_no_improve = 0
                temp = min(temp * 1.5, initial_temp)
            else:
                if accept_neutral:
                    no_improve_count += 1
                    best_no_improve = max(best_no_improve, no_improve_count)
                else:
                    current_used_bits &= ~diffs_i_new_bits
                    current_used_bits &= ~diffs_j_new_bits
                    current_used_bits |= old_diffs_i_bits
                    current_used_bits |= old_diffs_j_bits
                    current_rows[i] = old_row_i
                    current_rows[j] = old_row_j
                    no_improve_count += 1
                    best_no_improve = max(best_no_improve, no_improve_count)

            temp *= cooling_rate

        else:
            i = rng.randint(0, n_rows - 1)
            old_row = current_rows[i]
            old_diffs_bits = get_diffs_bitset(old_row)

            current_used_bits &= ~old_diffs_bits

            new_row = rebuild_row(k, current_used_bits, rng,
                                  target_max if target_max > k else current_max, pressure)
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
                best_no_improve = 0
                temp = min(temp * 1.5, initial_temp)
            else:
                if accept_neutral:
                    no_improve_count += 1
                    best_no_improve = max(best_no_improve, no_improve_count)
                else:
                    current_used_bits &= ~new_diffs_bits
                    current_used_bits |= old_diffs_bits
                    current_rows[i] = old_row
                    no_improve_count += 1
                    best_no_improve = max(best_no_improve, no_improve_count)

            temp *= cooling_rate

    # Final aggressive polish: try to reduce max by 1 repeatedly
    if not time_up():
        polish_deadline = min(deadline, time.time() + 10.0)
        while time.time() < polish_deadline:
            improved = False
            for i in range(n_rows):
                old_row = current_rows[i]
                old_diffs_bits = get_diffs_bitset(old_row)
                current_used_bits &= ~old_diffs_bits

                target_max = current_max - 1
                if target_max > k:
                    # Try multiple times with different random seeds
                    for attempt in range(5):
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
                    if improved:
                        break
                else:
                    current_used_bits |= old_diffs_bits

            if not improved:
                break

    return best_rows

# EVOLVE-BLOCK-END
