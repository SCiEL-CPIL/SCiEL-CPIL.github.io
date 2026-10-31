import time
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all pairwise differences
    within each row are distinct across all rows.
    """
    deadline = time.time() + 135  # safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    def get_all_diffs(rows):
        """Get all pairwise differences from all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def get_row_diffs(row):
        """Get pairwise differences of a single row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all rows are valid (strictly increasing, start at 0, unique diffs)."""
        used = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
            rd = get_row_diffs(row)
            if rd & used:
                return False
            used |= rd
        return True
    
    def greedy_build(n_rows, k):
        """Greedy construction: build rows one by one, picking smallest valid next element."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def greedy_randomized(n_rows, k, seed_offset=0):
        """Greedy with randomization: occasionally skip the smallest valid candidate."""
        rng = random.Random(seed_offset)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    # With some probability, skip this candidate
                    if rng.random() < 0.1 and x < 200:
                        continue
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def local_search_improve(rows, n_rows, k, deadline):
        """
        Iterated local search: try to reduce the max entry by replacing individual rows.
        Uses a backtracking approach to find a row with smaller max.
        """
        best_rows = [list(r) for r in rows]
        best_max = max(r[-1] for r in best_rows)
        
        def backtrack_find_row(used_others, k, max_allowed, deadline, max_nodes=200000):
            """
            Find a row of k+1 increasing ints starting at 0, all <= max_allowed,
            such that all pairwise diffs are not in used_others.
            Returns the row or None.
            """
            if time.time() > deadline:
                return None
            
            found = []
            node_count = [0]
            
            def bt(pos, current_row, current_diffs):
                if time.time() > deadline:
                    return False
                node_count[0] += 1
                if node_count[0] > max_nodes:
                    return False
                
                if pos == k + 1:
                    found.append(list(current_row))
                    return True
                
                if pos == 0:
                    current_row.append(0)
                    if bt(pos + 1, current_row, current_diffs):
                        current_row.pop()
                        return True
                    current_row.pop()
                    return False
                
                start = current_row[-1] + 1
                end = min(max_allowed, 200)  # limit search space
                
                for candidate in range(start, end + 1):
                    new_diffs = set()
                    valid = True
                    for a in current_row:
                        d = candidate - a
                        if d in used_others or d in current_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        current_row.append(candidate)
                        current_diffs.update(new_diffs)
                        if bt(pos + 1, current_row, current_diffs):
                            current_diffs -= new_diffs
                            current_row.pop()
                            return True
                        current_diffs -= new_diffs
                        current_row.pop()
                
                return False
            
            bt(0, [], set())
            if found:
                return found[0]
            return None
        
        iterations = 0
        max_iterations = 50
        stuck_count = 0
        
        while time_left() > 15 and iterations < max_iterations and stuck_count < 10:
            iterations += 1
            improved = False
            order = list(range(n_rows))
            random.shuffle(order)
            
            for idx in order:
                if time_left() < 15:
                    break
                
                current_row = rows[idx]
                current_max = current_row[-1]
                
                # Try to find a row with smaller max
                target_max = current_max - 1
                if target_max < k:  # minimum possible max is k
                    continue
                
                # Compute used diffs from other rows
                used_others = set()
                for i, row in enumerate(rows):
                    if i == idx:
                        continue
                    rd = get_row_diffs(row)
                    used_others |= rd
                
                new_row = backtrack_find_row(used_others, k, target_max, deadline, max_nodes=150000)
                if new_row is not None:
                    rows[idx] = new_row
                    new_max = max(r[-1] for r in rows)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in rows]
                    improved = True
                    stuck_count = 0
                else:
                    stuck_count += 1
        
        return best_rows, best_max
    
    def simulated_annealing(rows, n_rows, k, deadline):
        """
        Simulated annealing: perturb rows by shifting elements, accept/reject based on max entry.
        """
        best_rows = [list(r) for r in rows]
        best_max = max(r[-1] for r in best_rows)
        current_rows = [list(r) for r in rows]
        current_max = best_max
        
        def get_max(rows):
            return max(r[-1] for r in rows)
        
        def is_valid_rows(rows):
            used = set()
            for row in rows:
                if not row or row[0] != 0:
                    return False
                for i in range(1, len(row)):
                    if row[i] <= row[i-1]:
                        return False
                rd = get_row_diffs(row)
                if rd & used:
                    return False
                used |= rd
            return True
        
        temperature = 10.0
        cooling = 0.995
        min_temp = 0.1
        
        while time_left() > 15 and temperature > min_temp:
            # Pick a random row to perturb
            idx = random.randint(0, n_rows - 1)
            old_row = list(current_rows[idx])
            
            # Perturbation: shift one element
            pos = random.randint(1, len(old_row) - 1)
            shift = random.choice([-2, -1, 1, 2, 3])
            new_val = old_row[pos] + shift
            
            # Check if perturbation is valid
            if new_val <= old_row[pos - 1] or (pos < len(old_row) - 1 and new_val >= old_row[pos + 1]):
                temperature *= cooling
                continue
            
            new_row = list(old_row)
            new_row[pos] = new_val
            current_rows[idx] = new_row
            
            # Check validity
            if is_valid_rows(current_rows):
                new_max = get_max(current_rows)
                delta = new_max - current_max
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    current_max = new_max
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in current_rows]
                else:
                    current_rows[idx] = old_row
            else:
                current_rows[idx] = old_row
            
            temperature *= cooling
        
        return best_rows, best_max
    
    # Phase 1: Multiple greedy starts
    best_rows = None
    best_max = float('inf')
    
    # Deterministic greedy
    rows = greedy_build(n_rows, k)
    cur_max = max(r[-1] for r in rows)
    best_rows = [list(r) for r in rows]
    best_max = cur_max
    
    # Randomized greedy variants
    for seed in range(20):
        if time_left() < 30:
            break
        rows = greedy_randomized(n_rows, k, seed_offset=seed * 1000 + 42)
        if is_valid(rows):
            cur_max = max(r[-1] for r in rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in rows]
    
    # Phase 2: Local search improvement
    if time_left() > 25:
        best_rows, best_max = local_search_improve(best_rows, n_rows, k, deadline)
    
    # Phase 3: Simulated annealing
    if time_left() > 20:
        sa_rows, sa_max = simulated_annealing(best_rows, n_rows, k, deadline)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Final validation
    if not is_valid(best_rows):
        # Fallback to deterministic greedy
        best_rows = greedy_build(n_rows, k)
        best_max = max(r[-1] for r in best_rows)
    
    return best_rows

# EVOLVE-BLOCK-END
