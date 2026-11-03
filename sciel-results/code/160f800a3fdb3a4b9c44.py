import time
import random
import math
import heapq

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
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

    # Greedy construction
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_row_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                forbidden = used | current_row_diffs
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        row.append(x)
                        current_row_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
            used |= current_row_diffs
        return rows

    # Bitset-accelerated row generation
    def generate_row_bitset(used_diffs, k_val, max_scope, randomize=False, p_smallest=0.7, window=5):
        """Generate a single row using bitset for fast conflict checking."""
        # Create a bitset of forbidden differences
        max_d = max_scope + 1
        # Use a Python int as bitset: bit d is set if difference d is forbidden
        forbidden_bits = 0
        for d in used_diffs:
            if d < max_d:
                forbidden_bits |= (1 << d)
        
        row = [0]
        row_diffs_bits = 0  # Bitset of differences within current row
        
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            
            while x < max_scope:
                # Check if x creates any forbidden differences
                conflict = False
                for a in row:
                    d = x - a
                    if d < max_d and (forbidden_bits >> d) & 1:
                        conflict = True
                        break
                    if d < max_d and (row_diffs_bits >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            
            if not candidates:
                return None
            
            if randomize:
                if random.random() < p_smallest:
                    next_x = candidates[0]
                else:
                    next_x = random.choice(candidates)
            else:
                next_x = candidates[0]
            
            # Update row_diffs_bits
            for a in row:
                d = next_x - a
                if d < max_d:
                    row_diffs_bits |= (1 << d)
            
            row.append(next_x)
        
        return row

    # Branch and bound for maximum independent set on a small graph
    # Returns a set of positions (indices into 0..max_scope-1)
    def max_independent_set_branch_and_bound(conflict_matrix, n_positions, target_size):
        """
        Find a maximum independent set of size target_size in a graph.
        conflict_matrix: list of sets, where conflict_matrix[i] is the set of j that conflict with i.
        Returns a list of positions, or None if no independent set of size target_size exists.
        """
        if n_positions <= 0:
            return [] if target_size == 0 else None
        
        # Use a simple branch and bound
        # Order vertices by degree (heuristic: pick vertex with fewest conflicts first)
        degrees = [len(conflict_matrix[i]) for i in range(n_positions)]
        order = list(range(n_positions))
        order.sort(key=lambda i: degrees[i])
        
        # Map from original index to position in order
        pos_in_order = [0] * n_positions
        for pos, idx in enumerate(order):
            pos_in_order[idx] = pos
        
        # Try to build an independent set
        def backtrack(start_pos, current_set, remaining_needed):
            if remaining_needed == 0:
                return current_set[:]
            if start_pos >= len(order):
                return None
            
            # Upper bound: can we still reach target_size?
            if remaining_needed > len(order) - start_pos:
                return None
            
            # Try including order[start_pos]
            v = order[start_pos]
            # Check if v conflicts with any in current_set
            can_include = True
            for u in current_set:
                if u in conflict_matrix[v]:
                    can_include = False
                    break
            
            if can_include:
                result = backtrack(start_pos + 1, current_set + [v], remaining_needed - 1)
                if result is not None:
                    return result
            
            # Try excluding order[start_pos]
            result = backtrack(start_pos + 1, current_set, remaining_needed)
            if result is not None:
                return result
            
            return None
        
        result = backtrack(0, [], target_size)
        if result is not None:
            return sorted(result)
        return None

    # Local compaction: try to move each element to a smaller position
    def local_compact(rows, used_diffs, max_iterations=1000):
        """Try to reduce the scope by moving elements to smaller positions."""
        current_scope = max(max(r) for r in rows)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for p_idx in range(len(row) - 1, 0, -1):  # Start from last element
                    old_val = row[p_idx]
                    if old_val <= 1:
                        continue
                    
                    # Remove old_val from row
                    row.pop(p_idx)
                    
                    # Remove its differences from used_diffs
                    diffs_to_remove = set()
                    for a in row:
                        d = abs(old_val - a)
                        diffs_to_remove.add(d)
                    used_diffs -= diffs_to_remove
                    
                    # Try to find a smaller position
                    found = False
                    for x in range(old_val - 1, 0, -1):
                        if x in row:
                            continue
                        valid = True
                        for a in row:
                            d = abs(x - a)
                            if d == 0 or d in used_diffs:
                                valid = False
                                break
                        if valid:
                            # Insert x in sorted position
                            pos = 0
                            while pos < len(row) and row[pos] < x:
                                pos += 1
                            row.insert(pos, x)
                            for a in row:
                                if a != x:
                                    used_diffs.add(abs(x - a))
                            found = True
                            improved = True
                            break
                    
                    if not found:
                        # Re-insert old_val
                        pos = 0
                        while pos < len(row) and row[pos] < old_val:
                            pos += 1
                        row.insert(pos, old_val)
                        for a in row:
                            if a != old_val:
                                used_diffs.add(abs(old_val - a))
        
        return rows, used_diffs

    # 1. Initial Solution: Deterministic Greedy
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # 2. Shortest-Path Inspired Search: Dijkstra-like with row states
    # Treat each row as a node in a DAG where edges represent adding a row pattern
    # Use Dijkstra to find minimum-scope configurations by treating scope as edge weight
    
    # First, try a Dijkstra-based approach to construct rows with minimal scope
    def dijkstra_row_search(n_rows, k, max_scope_limit, time_limit=30):
        """
        Use Dijkstra-like search to construct rows with minimal scope.
        State: (scope, row_index, used_diffs_fingerprint, rows_so_far)
        We use a priority queue ordered by scope.
        """
        dijk_start = time.time()
        
        # For efficiency, we'll use a beam search approach since full Dijkstra
        # with all possible used_diffs states is too large
        beam_width = 50
        best_solution = None
        best_scope = float('inf')
        
        # Initial state: no rows, no used diffs
        # State representation: (scope, row_idx, frozenset(used_diffs), tuple(rows))
        # But this is too large, so we use a greedy heuristic with Dijkstra-like ordering
        
        # Instead, we'll use a layered approach: for each row, generate candidate rows
        # and use Dijkstra to pick the best sequence
        
        used = set()
        rows_found = []
        
        for row_idx in range(n_rows):
            if time.time() - dijk_start > time_limit:
                break
            
            # Generate multiple candidate rows using different strategies
            candidates = []
            
            # Strategy 1: Greedy smallest
            row_greedy = generate_row_bitset(used, k, max_scope_limit, randomize=False)
            if row_greedy:
                candidates.append(row_greedy)
            
            # Strategy 2: Randomized with different parameters
            for _ in range(10):
                row_rand = generate_row_bitset(used, k, max_scope_limit, randomize=True, 
                                               p_smallest=random.uniform(0.5, 0.9), 
                                               window=random.randint(2, 8))
                if row_rand:
                    candidates.append(row_rand)
            
            if not candidates:
                break
            
            # Sort candidates by their maximum value (scope contribution)
            candidates.sort(key=lambda r: max(r))
            
            # Use a Dijkstra-like selection: pick the candidate that leads to best overall scope
            # For simplicity, pick the one with smallest max value
            best_row = candidates[0]
            rows_found.append(best_row)
            used.update({best_row[j] - best_row[i] for i in range(len(best_row)) for j in range(i + 1, len(best_row))})
        
        if len(rows_found) == n_rows:
            scope = max(max(r) for r in rows_found)
            if scope < best_scope:
                best_scope = scope
                best_solution = rows_found
        
        return best_solution, best_scope

    # Try Dijkstra-inspired search with increasing scope limits
    dijkstra_deadline = deadline - 60
    dijk_solution = None
    dijk_scope = float('inf')
    
    # Start with a reasonable scope limit and increase
    for scope_limit in range(best_max - 5, min(best_max + 10, 150), 2):
        if time.time() > dijkstra_deadline:
            break
        
        sol, scope = dijkstra_row_search(n_rows, k, scope_limit, time_limit=10)
        if sol and scope < dijk_scope:
            dijk_scope = scope
            dijk_solution = sol
    
    # Use Dijkstra solution if better
    if dijk_solution and dijk_scope < best_max:
        rows = dijk_solution
        best_rows = [r[:] for r in rows]
        best_max = dijk_scope
        current_max = best_max

    # 2. Simulated Annealing with Row Regeneration
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    used = get_used_diffs(rows)
    
    # Use more of the time budget for SA
    sa_deadline = deadline - 15
    while time.time() < sa_deadline:
        if T < T_min:
            T = 5.0
            cooling = 0.998
            
        r_idx = random.randint(0, n_rows - 1)
        
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used -= old_diffs
        
        # Regenerate row with randomization
        new_row = generate_row_bitset(used, k, best_max + 20, randomize=True, p_smallest=0.8, window=3)
        
        if new_row is None:
            used |= old_diffs
            continue
        
        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            rows[r_idx] = new_row
            used |= set(new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i + 1, len(new_row)))
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            used |= old_diffs
            
        T *= cooling

    # 3. Iterative Compaction Local Search
    # This phase aggressively tries to reduce the scope by moving elements
    # to smaller positions. It repeats until no further improvements are found
    # or the time limit is reached.
    
    current_rows = [list(r) for r in best_rows]
    used_diffs = get_used_diffs(current_rows)
    current_scope = max(max(r) for r in current_rows)
    
    ls_deadline = deadline - 5
    improved = True
    while improved and time.time() < ls_deadline:
        improved = False
        
        # Iterate through rows and elements in reverse order to allow
        # larger elements to move down, potentially freeing space for
        # smaller ones later (though we check all, order helps).
        for r_idx in range(n_rows):
            row = current_rows[r_idx]
            # Iterate from last element to first (excluding 0)
            for p_idx in range(len(row) - 1, 0, -1):
                old_val = row[p_idx]
                if old_val <= 1:
                    continue
                
                # Remove old_val from consideration
                row.pop(p_idx)
                # Remove its diffs from used_diffs
                diffs_to_remove = set()
                for a in row:
                    d = abs(old_val - a)
                    diffs_to_remove.add(d)
                used_diffs -= diffs_to_remove
                
                # Find the smallest valid position for this element
                found_pos = None
                # We can search from 1 up to old_val
                # Optimization: if we just want to reduce scope, we search 1..old_val-1
                for x in range(1, old_val):
                    if x in row:
                        continue
                    valid = True
                    for a in row:
                        d = abs(x - a)
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        found_pos = x
                        break
                
                if found_pos is not None:
                    # Insert found_pos
                    pos = 0
                    while pos < len(row) and row[pos] < found_pos:
                        pos += 1
                    row.insert(pos, found_pos)
                    # Update used_diffs
                    for a in row:
                        if a != found_pos:
                            used_diffs.add(abs(found_pos - a))
                    improved = True
                    # Update current_scope if this was the max
                    # We'll recalculate at the end of the row/loop to save time
                else:
                    # Revert
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for d in diffs_to_remove:
                        used_diffs.add(d)
        
        # Recalculate scope
        new_scope = max(max(r) for r in current_rows)
        if new_scope < current_scope:
            current_scope = new_scope
            best_rows = [r[:] for r in current_rows]
            best_max = current_scope
        else:
            # If no scope reduction, we might still have improved internal structure
            # but for this problem, scope is the only metric.
            # If no improvement in scope, we can break to save time, 
            # unless we want to try random perturbations.
            # Given the time constraints, breaking is safer.
            if not improved:
                break
                
    # Final validation and fallback
    if not is_valid(best_rows):
        # Fallback to a simple greedy construction if something went wrong
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)

    return best_rows
