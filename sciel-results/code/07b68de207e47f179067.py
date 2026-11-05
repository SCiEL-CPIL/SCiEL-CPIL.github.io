import time
import random
import math
import sys

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 145.0  # seconds, safety margin before 150s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2

    # Helper to calculate differences for a row given deltas
    def get_diffs(deltas):
        """
        deltas: list of k positive integers
        returns: set of all sums of consecutive deltas
        """
        diffs = set()
        n = len(deltas)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += deltas[j]
                diffs.add(s)
        return diffs

    # Helper to check validity and get max sum
    def evaluate(deltas_list):
        """
        deltas_list: list of n_rows lists of k deltas
        returns: (num_collisions, max_sum)
        """
        all_diffs = {}
        max_sum = 0
        for deltas in deltas_list:
            row_sum = sum(deltas)
            if row_sum > max_sum:
                max_sum = row_sum
            diffs = get_diffs(deltas)
            for d in diffs:
                if d in all_diffs:
                    all_diffs[d] += 1
                else:
                    all_diffs[d] = 1
        
        collisions = 0
        for count in all_diffs.values():
            if count > 1:
                collisions += 1
        return collisions, max_sum

    # Greedy construction to get a valid starting point
    def greedy_construct():
        deltas_list = []
        used_diffs = set()
        for _ in range(n_rows):
            deltas = []
            # Try to find small deltas that don't conflict
            # Start with small values, increase if conflict
            base_delta = 1
            while True:
                # Generate a candidate row
                cand_deltas = [base_delta + random.randint(0, 5) for _ in range(k)]
                # Ensure strictly positive
                cand_deltas = [max(1, d) for d in cand_deltas]
                
                cand_diffs = get_diffs(cand_deltas)
                if not (cand_diffs & used_diffs):
                    deltas_list.append(cand_deltas)
                    used_diffs |= cand_diffs
                    break
                base_delta += 1
                if base_delta > 200: # Safety break
                     # Force a unique large delta
                     cand_deltas = [1]*k
                     cand_deltas[-1] = 1000 + random.randint(0,100)
                     cand_diffs = get_diffs(cand_deltas)
                     if not (cand_diffs & used_diffs):
                         deltas_list.append(cand_deltas)
                         used_diffs |= cand_diffs
                         break
                     else:
                         continue
            # If we didn't break in the inner while, it means we found a solution
        return deltas_list

    # Initialize with greedy
    best_deltas = greedy_construct()
    best_collisions, best_max = evaluate(best_deltas)
    # Greedy should produce 0 collisions
    if best_collisions != 0:
         # Fallback to random if greedy fails (unlikely)
         best_deltas = [[random.randint(1, 10) for _ in range(k)] for _ in range(n_rows)]
         best_collisions, best_max = evaluate(best_deltas)

    # Simulated Annealing to minimize collisions and max_sum
    # We want to drive collisions to 0 first, then minimize max_sum
    
    # State
    current_deltas = [list(d) for d in best_deltas]
    current_collisions, current_max = evaluate(current_deltas)
    
    # Objective function
    def cost(collisions, max_sum):
        # Prioritize removing collisions
        return collisions * 1000.0 + max_sum

    current_cost = cost(current_collisions, current_max)
    best_cost = current_cost
    best_deltas_sa = [list(d) for d in current_deltas]
    
    T = 100.0
    T_min = 0.1
    alpha = 0.999 # Cooling rate
    steps_per_second = 50000 # Estimate
    
    # SA Loop
    while _time_left() > 10.0 and T > T_min:
        # Pick a random move
        r = random.randint(0, n_rows - 1)
        t = random.randint(0, k - 1)
        
        old_val = current_deltas[r][t]
        # New value: random walk
        delta_change = random.randint(-5, 5)
        new_val = max(1, old_val + delta_change)
        
        if new_val == old_val:
            continue
            
        # Apply move
        current_deltas[r][t] = new_val
        
        # Evaluate new state
        new_collisions, new_max = evaluate(current_deltas)
        new_cost = cost(new_collisions, new_max)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_deltas_sa = [list(d) for d in current_deltas]
        else:
            # Revert
            current_deltas[r][t] = old_val
            
        T *= alpha

    # Update best if SA found something better
    if best_cost < cost(best_collisions, best_max):
        best_deltas = best_deltas_sa
        best_collisions, best_max = evaluate(best_deltas)

    # If we still have collisions, we need to force a valid solution
    # Run a greedy repair or restart SA with higher T if needed
    if best_collisions > 0:
        # Simple repair: increase deltas in rows with collisions
        # Or just use the greedy one which was valid
        if cost(best_collisions, best_max) > cost(0, 10000):
             # Greedy was valid, use it
             best_deltas = greedy_construct()
             best_collisions, best_max = evaluate(best_deltas)

    # Now we have a valid solution (0 collisions).
    # Try to minimize the max_sum (scope) using iterative tightening.
    
    # Convert deltas to rows for output later
    def deltas_to_rows(deltas_list):
        rows = []
        for deltas in deltas_list:
            row = [0]
            current = 0
            for d in deltas:
                current += d
                row.append(current)
            rows.append(row)
        return rows

    # Optimization Phase: Tightening
    # We try to reduce the maximum sum.
    # We use a penalty method: minimize collisions + penalty * overflow
    # Where overflow = sum(max(0, sum_r - target))
    
    current_max = best_max
    # We can try to reduce current_max by 1 each time
    
    # Time budget for tightening: 50% of remaining time
    tighten_deadline = time.time() + (_time_left() * 0.5)
    
    while _time_left() > 5.0 and time.time() < tighten_deadline:
        target_max = current_max - 1
        if target_max < total_diffs: # Theoretical lower bound
            break
            
        # Run a short SA to find a solution with max_sum <= target_max
        # Initialize with current best
        sa_deltas = [list(d) for d in best_deltas]
        
        def cost_tighten(deltas_list, target):
            collisions, max_sum = evaluate(deltas_list)
            overflow = 0
            for deltas in deltas_list:
                s = sum(deltas)
                if s > target:
                    overflow += s - target
            # We want 0 collisions and 0 overflow
            return collisions * 100.0 + overflow * 10.0 + max_sum * 0.1

        sa_cost = cost_tighten(sa_deltas, target_max)
        best_sa_cost = sa_cost
        best_sa_deltas = [list(d) for d in sa_deltas]
        
        T = 50.0
        T_min = 0.1
        alpha = 0.9995
        
        # Short SA run
        steps = 20000
        for step in range(steps):
            if time.time() > tighten_deadline:
                break
            r = random.randint(0, n_rows - 1)
            t = random.randint(0, k - 1)
            
            old_val = sa_deltas[r][t]
            delta_change = random.randint(-3, 3)
            new_val = max(1, old_val + delta_change)
            
            if new_val == old_val:
                continue
                
            sa_deltas[r][t] = new_val
            new_cost = cost_tighten(sa_deltas, target_max)
            
            delta_cost = new_cost - sa_cost
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                sa_cost = new_cost
                if new_cost < best_sa_cost:
                    best_sa_cost = new_cost
                    best_sa_deltas = [list(d) for d in sa_deltas]
            else:
                sa_deltas[r][t] = old_val
                
            T *= alpha
            
        # Check if we found a valid solution with reduced max
        # Valid means 0 collisions and 0 overflow (all sums <= target_max)
        # cost_tighten returns 0 only if collisions=0 and overflow=0?
        # My cost function: collisions*100 + overflow*10 + max_sum*0.1
        # If collisions=0 and overflow=0, cost is max_sum*0.1.
        # We need to explicitly check validity.
        
        test_collisions, test_max = evaluate(best_sa_deltas)
        if test_collisions == 0 and test_max <= target_max:
            best_deltas = best_sa_deltas
            current_max = test_max
            # Continue tightening
        else:
            # If we couldn't reduce by 1, maybe we can reduce by less?
            # Or just break.
            # Try one more step with target_max - 2?
            # For now, break to avoid wasting time.
            break

    # Final Validation and Conversion
    # Ensure we have a valid solution
    final_collisions, final_max = evaluate(best_deltas)
    if final_collisions != 0:
        # Fallback to the initial greedy solution which was valid
        best_deltas = greedy_construct()
        final_collisions, final_max = evaluate(best_deltas)
        
    # Convert to rows
    result_rows = deltas_to_rows(best_deltas)
    
    # Double check validity of result_rows
    def validate_rows(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    if not validate_rows(result_rows, n_rows, k):
        # Emergency fallback
        result_rows = greedy_construct()
        result_rows = deltas_to_rows(result_rows)

    return result_rows
