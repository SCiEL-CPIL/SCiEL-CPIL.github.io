import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Difference triangle set (n_rows, k) using Simulated Annealing with
    optimized O(k) conflict tracking for fast neighbor evaluation.
    """
    deadline = time.time() + 115  # safety margin before 150s kill
    
    def greedy_initial(seed):
        """Generate a valid solution via greedy with random tie-breaking."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def compute_diffs(rows):
        """Return dict mapping each difference to the set of (row_idx, i, j) that produce it."""
        diff_map = defaultdict(list)
        for ri, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diff_map[d].append((ri, i, j))
        return diff_map
    
    def count_conflicts(diff_map):
        """Count number of differences that appear more than once."""
        return sum(1 for v in diff_map.values() if len(v) > 1)
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def get_cost(rows, diff_map):
        """Total cost = scope + large penalty for conflicts."""
        conflicts = count_conflicts(diff_map)
        return get_scope(rows) + conflicts * 10000
    
    # Generate initial solution
    best_rows = greedy_initial(42)
    best_diff_map = compute_diffs(best_rows)
    best_scope = get_scope(best_rows)
    best_cost = get_cost(best_rows, best_diff_map)
    
    # Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_diff_map = compute_diffs(current_rows)
    current_scope = get_scope(current_rows)
    current_cost = get_cost(current_rows, current_diff_map)
    
    T_start = 50.0
    T_end = 0.1
    total_steps = 500000  # will be limited by deadline
    
    # Precompute all possible (row_idx, pos) pairs for mutation
    positions = [(ri, pos) for ri in range(n_rows) for pos in range(1, k + 1)]
    
    step = 0
    while time.time() < deadline:
        # Cooling schedule
        progress = min(1.0, step / total_steps)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random position to modify
        ri, pos = random.choice(positions)
        old_val = current_rows[ri][pos]
        
        # Generate a new value: try to reduce scope or perturb
        if random.random() < 0.7 and old_val > current_rows[ri][pos - 1] + 1:
            # Try to reduce
            delta = random.randint(1, max(1, old_val - current_rows[ri][pos - 1] - 1))
            new_val = old_val - delta
        else:
            # Perturb: move up or down randomly
            max_move = max(5, int(0.1 * current_scope))
            delta = random.randint(-max_move, max_move)
            new_val = old_val + delta
        
        # Ensure increasing
        if new_val <= current_rows[ri][pos - 1]:
            new_val = current_rows[ri][pos - 1] + 1
        if new_val >= old_val and pos < k:
            # Check it doesn't violate the next element
            if pos < k and new_val >= current_rows[ri][pos + 1]:
                new_val = current_rows[ri][pos + 1] - 1
                if new_val <= current_rows[ri][pos - 1]:
                    continue  # Can't move here
        
        if new_val == old_val:
            continue
        
        # Compute change in differences for this row
        # Old differences involving position pos
        row = current_rows[ri]
        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            d = abs(row[pos] - row[i])
            old_diffs.add(d)
        
        # New differences
        new_row = row[:]
        new_row[pos] = new_val
        new_diffs = set()
        for i in range(len(new_row)):
            if i == pos:
                continue
            d = abs(new_row[pos] - new_row[i])
            new_diffs.add(d)
        
        # Diff of diffs: what's removed and what's added
        removed_diffs = old_diffs - new_diffs
        added_diffs = new_diffs - old_diffs
        
        # Compute new conflict count efficiently
        old_conflicts = count_conflicts(current_diff_map)
        
        # Adjust conflict count
        delta_conflicts = 0
        for d in removed_diffs:
            if d in current_diff_map:
                cnt = len(current_diff_map[d])
                if cnt > 1:
                    delta_conflicts -= 1  # was conflict, now one fewer
                # if cnt == 1, no change in conflict count
        for d in added_diffs:
            if d in current_diff_map:
                cnt = len(current_diff_map[d])
                if cnt >= 1:
                    delta_conflicts += 1  # now has conflict (cnt+1 >= 2)
            else:
                pass  # new diff, no conflict
        
        new_conflicts = old_conflicts + delta_conflicts
        
        # New scope
        new_scope = max(current_scope, new_val) if new_val > current_scope else (
            max(max(r) for r in current_rows) if new_val < current_scope else current_scope
        )
        # More careful scope calculation
        if new_val > current_scope:
            new_scope = new_val
        else:
            # Check if old_val was the max and we reduced it
            # For simplicity, recompute only if needed
            if old_val == current_scope:
                new_scope = max(max(r) for r in current_rows if r is not row)
                if new_val > new_scope:
                    new_scope = new_val
            else:
                new_scope = current_scope
        
        new_cost = new_scope + new_conflicts * 10000
        old_cost = current_cost
        
        # Accept?
        if new_cost < old_cost:
            accept = True
        else:
            delta = new_cost - old_cost
            if T > 0:
                import math
                accept = random.random() < math.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            # Update state
            current_rows[ri][pos] = new_val
            
            # Update diff_map
            for d in removed_diffs:
                if d in current_diff_map:
                    current_diff_map[d] = [x for x in current_diff_map[d] if not (x[0] == ri and (x[1] == pos or x[2] == pos))]
                    if not current_diff_map[d]:
                        del current_diff_map[d]
            for d in added_diffs:
                if d not in current_diff_map:
                    current_diff_map[d] = []
                # Add the new (ri, i, pos) entries
                row_new = current_rows[ri]
                for i in range(len(row_new)):
                    if i == pos:
                        continue
                    if row_new[pos] - row_new[i] == d:
                        current_diff_map[d].append((ri, i, pos))
                    elif row_new[i] - row_new[pos] == d:
                        current_diff_map[d].append((ri, pos, i))
            
            current_scope = new_scope
            current_cost = new_cost
            
            # Update best
            if new_conflicts == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                best_diff_map = {d: lst[:] for d, lst in current_diff_map.items()}
        
        step += 1
    
    # Final: if we have a valid best, return it
    if best_rows is not None:
        # Verify
        used = set()
        valid = True
        for row in best_rows:
            if row[0] != 0:
                valid = False
                break
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= 0 or d in used:
                        valid = False
                        break
                    used.add(d)
            if not valid:
                break
        if valid:
            return best_rows
    
    # Fallback
    return greedy_initial(0)
# EVOLVE-BLOCK-END
