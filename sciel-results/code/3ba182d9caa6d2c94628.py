import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_solution = None
    best_scope = float('inf')
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    def scope(rows):
        return max(max(row) for row in rows)
    
    def greedy_build(used_diffs, k, max_val=None):
        """Build one row greedily given already-used differences."""
        if max_val is None:
            max_val = 10000
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return None
            new_diffs = set()
            conflict = False
            for a in row:
                d = x - a
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            if not conflict:
                row.append(x)
                used_diffs |= new_diffs
        return row
    
    def greedy_build_full(n_rows, k, seed=None):
        """Build all rows greedily."""
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_build(used, k)
            if row is None:
                return None
            rows.append(row)
        return rows
    
    def try_improve_row(row, used_diffs_except_row, k, max_val=None):
        """Try to reduce the scope of a single row while keeping it valid
        with respect to used_diffs_except_row (diffs from all other rows)."""
        if max_val is None:
            max_val = 10000
        
        # Current row's own diffs
        current_diffs = get_diffs(row)
        # Diffs available: not in used_diffs_except_row
        # We want to find a new row with same length, starting at 0, increasing,
        # with all diffs disjoint from used_diffs_except_row, and max entry <= current max
        
        # Try to greedily rebuild with a tighter bound
        target_max = row[-1]
        # Try progressively smaller max values
        for new_max in range(target_max, 0, -1):
            if new_max < k:  # minimum possible
                break
            new_row = [0]
            x = 0
            local_diffs = set()
            success = True
            while len(new_row) < k + 1:
                x += 1
                if x > new_max:
                    success = False
                    break
                conflict = False
                for a in new_row:
                    d = x - a
                    if d in used_diffs_except_row or d in local_diffs:
                        conflict = True
                        break
                    local_diffs.add(d)
                if not conflict:
                    new_row.append(x)
            if success and len(new_row) == k + 1:
                return new_row
        return None
    
    def local_search(rows, n_rows, k, time_limit=5.0):
        """Improve solution by trying to reduce individual row scopes."""
        start_ls = time.time()
        current_rows = [list(r) for r in rows]
        current_scope = scope(current_rows)
        best_local = [list(r) for r in current_rows]
        best_local_scope = current_scope
        
        # Compute used diffs for each row
        def compute_used_except(idx):
            used = set()
            for i, row in enumerate(current_rows):
                if i == idx:
                    continue
                used |= get_diffs(row)
            return used
        
        # Try improving each row multiple times
        for iteration in range(200):
            if time.time() - start_ls > time_limit:
                break
            if time.time() > deadline:
                break
            
            for idx in range(n_rows):
                if time.time() > deadline:
                    break
                used_except = compute_used_except(idx)
                new_row = try_improve_row(current_rows[idx], used_except, k)
                if new_row is not None:
                    old_max = current_rows[idx][-1]
                    new_max = new_row[-1]
                    if new_max < old_max:
                        current_rows[idx] = new_row
                        new_scope = scope(current_rows)
                        if new_scope < best_local_scope:
                            best_local_scope = new_scope
                            best_local = [list(r) for r in current_rows]
        
        return best_local, best_local_scope
    
    def swap_and_rebuild(rows, n_rows, k, time_limit=3.0):
        """Try regenerating one row at a time with different strategies."""
        start_sw = time.time()
        current_rows = [list(r) for r in rows]
        current_scope = scope(current_rows)
        best_local = [list(r) for r in current_rows]
        best_local_scope = current_scope
        
        for iteration in range(100):
            if time.time() - start_sw > time_limit:
                break
            if time.time() > deadline:
                break
            
            idx = random.randint(0, n_rows - 1)
            # Remove this row's diffs
            used_except = set()
            for i, row in enumerate(current_rows):
                if i != idx:
                    used_except |= get_diffs(row)
            
            # Try to build a new row with a lower max
            target_max = current_rows[idx][-1] - 1
            if target_max < k:
                continue
            
            new_row = greedy_build(used_except, k, max_val=target_max)
            if new_row is not None:
                current_rows[idx] = new_row
                new_scope = scope(current_rows)
                if new_scope < best_local_scope:
                    best_local_scope = new_scope
                    best_local = [list(r) for r in current_rows]
        
        return best_local, best_local_scope
    
    # Phase 1: Initial greedy solution
    initial = greedy_build_full(n_rows, k, seed=42)
    if initial is not None:
        best_solution = initial
        best_scope = scope(initial)
    
    # Try a few different greedy seeds
    for seed in [0, 1, 7, 13, 99, 123, 456, 789]:
        if time.time() > deadline:
            break
        sol = greedy_build_full(n_rows, k, seed=seed)
        if sol is not None:
            s = scope(sol)
            if s < best_scope:
                best_scope = s
                best_solution = sol
    
    # Phase 2: Local search improvement
    if best_solution is not None:
        if time.time() < deadline - 10:
            improved, imp_scope = local_search(best_solution, n_rows, k, time_limit=30)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_solution = improved
        
        # Phase 3: More aggressive search with restarts
        if time.time() < deadline - 5:
            for restart in range(10):
                if time.time() > deadline - 2:
                    break
                # Perturb: regenerate one random row
                current = [list(r) for r in best_solution]
                idx = random.randint(0, n_rows - 1)
                used_except = set()
                for i, row in enumerate(current):
                    if i != idx:
                        used_except |= get_diffs(row)
                
                # Try to build with a slightly lower target
                target = current[idx][-1] - random.randint(1, 5)
                if target < k:
                    target = current[idx][-1]
                new_row = greedy_build(used_except, k, max_val=target)
                if new_row is not None:
                    current[idx] = new_row
                    s = scope(current)
                    if s < best_scope:
                        best_scope = s
                        best_solution = current
                
                # Local search from here
                if time.time() < deadline - 5:
                    improved, imp_scope = local_search(current, n_rows, k, time_limit=10)
                    if imp_scope < best_scope:
                        best_scope = imp_scope
                        best_solution = improved
        
        # Phase 4: Final local search polish
        if time.time() < deadline - 2:
            improved, imp_scope = local_search(best_solution, n_rows, k, time_limit=15)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_solution = improved
    
    # Validate final solution
    if best_solution is not None and is_valid(best_solution):
        return best_solution
    elif best_solution is not None:
        # Should not happen, but return anyway
        return best_solution
    else:
        # Fallback: simple greedy
        return greedy_build_full(n_rows, k, seed=0)
# EVOLVE-BLOCK-END
