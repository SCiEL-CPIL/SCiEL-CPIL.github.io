import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute total number of differences needed
    total_diffs = n_rows * (k + 1) * k // 2

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    if r[j] <= r[i]:
                        return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def greedy_row(used_diffs, max_val, rng, skip_prob=0.0):
        """
        Build a single row of length k+1 starting at 0, with all differences
        not in used_diffs, and max value <= max_val.
        """
        row = [0]
        local_diffs = set()
        while len(row) < k + 1:
            x = row[-1] + 1
            found = False
            while x <= max_val:
                # Optional random skip for diversity
                if skip_prob > 0 and rng.random() < skip_prob and x < max_val:
                    x += 1
                    continue
                
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        bad = True
                        break
                    new_diffs.append(d)
                
                if not bad:
                    row.append(x)
                    local_diffs.update(new_diffs)
                    found = True
                    break
                x += 1
            
            if not found:
                return None
        return row

    def greedy_construct(M_limit, seed_val=0, skip_prob=0.0):
        """
        Greedy construction: build rows one by one.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = greedy_row(used_diffs, M_limit, rng, skip_prob)
            if row is None:
                return None
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        return rows

    def interleave_greedy(M_limit, seed_val=0, skip_prob=0.0):
        """
        Interleaved greedy: extend the row with the smallest current maximum at each step.
        Maintains local diffs incrementally for speed.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        local_diffs = [set() for _ in range(n_rows)]
        
        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break
            
            target_row_idx = min(candidates, key=lambda idx: rows[idx][-1])
            row = rows[target_row_idx]
            ldiffs = local_diffs[target_row_idx]
            
            x = row[-1] + 1
            found = False
            while x <= M_limit:
                if skip_prob > 0 and rng.random() < skip_prob and x < M_limit:
                    x += 1
                    continue
                
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in ldiffs:
                        bad = True
                        break
                    new_diffs.append(d)
                
                if not bad:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    ldiffs.update(new_diffs)
                    found = True
                    break
                x += 1
            
            if not found:
                return None
        
        for r in rows:
            if len(r) != k + 1:
                return None
        
        return rows

    def backtrack_search(M_limit, seed_val=0, time_budget=10.0):
        """
        Backtracking search to find a solution within M_limit.
        Uses randomized backtracking with time limit.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        def _backtrack(row_idx):
            if time.time() > start_time + time_budget:
                return None
            
            if row_idx == n_rows:
                return [r[:] for r in rows]
            
            # Try several randomizations
            for attempt in range(50):
                sp = rng.uniform(0.0, 0.3) if attempt > 0 else 0.0
                row = greedy_row(used_diffs, M_limit, rng, skip_prob=sp)
                if row is not None:
                    # Check internal consistency
                    row_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in used_diffs:
                                break
                            row_diffs.add(d)
                    else:
                        # No conflicts, commit
                        rows.append(row)
                        used_diffs.update(row_diffs)
                        result = _backtrack(row_idx + 1)
                        if result is not None:
                            return result
                        rows.pop()
                        used_diffs.difference_update(row_diffs)
            
            return None
        
        return _backtrack(0)

    def simulated_annealing(rows, time_budget, seed_val=0):
        """
        Simulated annealing to reduce the scope.
        Moves: replace one row with a new valid row that fits in the current scope.
        """
        if rows is None:
            return None, None
        
        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = max(max(r) for r in best)
        current = [r[:] for r in best]
        current_scope = best_scope
        
        used = get_used_diffs(current)
        deadline = time.time() + time_budget
        T = 20.0
        cooling = 0.998
        step = 0
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and T > 0.05:
            step += 1
            
            # Pick a random row to replace
            ri = rng.randint(0, n_rows - 1)
            old_row = current[ri]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            used_without = used - old_diffs
            
            # Try to build a new row within current_scope
            max_mark = current_scope
            new_row = None
            
            # Try more randomizations of greedy with varying skip probs
            for attempt in range(40):
                sp = rng.uniform(0.0, 0.4)
                new_row = greedy_row(used_without, max_mark, rng, skip_prob=sp)
                if new_row is not None:
                    break
            
            if new_row is None:
                T *= cooling
                no_improve += 1
                if no_improve > max_no_improve:
                    T = max(T, 15.0)
                    no_improve = 0
                continue
            
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            if new_diffs & used_without:
                T *= cooling
                no_improve += 1
                if no_improve > max_no_improve:
                    T = max(T, 15.0)
                    no_improve = 0
                continue
            
            # Compute new scope
            other_max = 0
            for i, r in enumerate(current):
                if i != ri:
                    other_max = max(other_max, max(r))
            new_scope = max(other_max, max(new_row))
            
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                used = used_without | new_diffs
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                T = max(T, 15.0)
                no_improve = 0
            
            T *= cooling
        
        return best, best_scope

    # Main logic
    best_rows = None
    best_M = float('inf')
    
    # Phase 1: Get a baseline with greedy and backtracking
    # Start M from a reasonable lower bound
    start_M = total_diffs
    end_M = total_diffs + 50
    
    # First try greedy approaches
    for M in range(start_M, end_M):
        if time.time() > deadline - 30:
            break
        
        # Try a few seeds for greedy
        for seed in range(5):
            if time.time() > deadline - 25:
                break
            
            # Alternate between standard greedy and interleaved greedy
            if seed % 2 == 0:
                rows = greedy_construct(M, seed)
            else:
                rows = interleave_greedy(M, seed)
            
            if rows is not None and is_valid(rows):
                current_max = max(max(r) for r in rows)
                if current_max < best_M:
                    best_M = current_max
                    best_rows = [r[:] for r in rows]
                break
        
        if best_rows is not None and best_M <= M:
            # We found a solution for this M, no need to try larger M for baseline
            break

    # If no baseline found, try backtracking
    if best_rows is None:
        for M in range(start_M, start_M + 20):
            if time.time() > deadline - 20:
                break
            rows = backtrack_search(M, seed_val=0, time_budget=5.0)
            if rows is not None and is_valid(rows):
                best_M = max(max(r) for r in rows)
                best_rows = [r[:] for r in rows]
                break

    # If still no solution, try larger M
    if best_rows is None:
        for M in range(end_M, end_M + 100):
            if time.time() > deadline - 10:
                break
            rows = greedy_construct(M, 0)
            if rows is not None and is_valid(rows):
                best_M = max(max(r) for r in rows)
                best_rows = [r[:] for r in rows]
                break

    # Emergency fallback if still no solution
    if best_rows is None:
        M_huge = 1000
        rows = greedy_construct(M_huge, 0)
        if rows is not None:
            best_rows = rows
        else:
            # Construct a valid solution with large gaps
            best_rows = []
            base = 0
            for i in range(n_rows):
                row = [0]
                for j in range(1, k + 1):
                    base += 100
                    row.append(base)
                best_rows.append(row)
            best_M = max(max(r) for r in best_rows)

    # Phase 2: Improve with Simulated Annealing
    # Use remaining time for SA
    time_left = deadline - time.time()
    if time_left > 5:
        # Run SA for most of the remaining time
        sa_budget = min(time_left - 5, 60)
        if sa_budget > 0:
            # Run multiple SA runs with different seeds if time permits
            improved_rows, improved_M = simulated_annealing(best_rows, sa_budget, seed_val=42)
            
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows
            
            # If time remains, try another SA run with a different seed
            time_left = deadline - time.time()
            if time_left > 5:
                sa_budget2 = min(time_left - 5, 30)
                if sa_budget2 > 0:
                    improved_rows2, improved_M2 = simulated_annealing(best_rows, sa_budget2, seed_val=123)
                    if improved_rows2 is not None and improved_M2 < best_M:
                        best_M = improved_M2
                        best_rows = improved_rows2

    return best_rows

# Test / Main guard (not required for evaluation but good for local testing)
if __name__ == "__main__":
    # Example for (7, 5)
    rows = construct(7, 5)
    if rows:
        print("Solution found:")
        for r in rows:
            print(r)
        print("Scope:", max(max(r) for r in rows))
        print("Valid:", is_valid(rows))
    else:
        print("No solution found")
