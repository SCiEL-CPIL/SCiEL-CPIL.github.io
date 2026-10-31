"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across rows.
    Minimize the largest entry.
    
    Strategy:
    1. Greedy initialization (fast baseline)
    2. Simulated Annealing to escape local minima, optimizing the binding (largest) row first
    3. Per-row local polish to refine
    4. Deadline guard to return best result before 150s wall-clock limit
    """
    deadline = time.time() + 140  # 140s to leave safety margin
    
    # Total differences per row: k*(k+1)/2
    # Total differences across all rows: n_rows * k*(k+1)/2
    # The differences must be in range [1, max_val]
    
    def greedy_construct(n_rows, k):
        """Greedy construction: each row takes smallest next mark with no conflict."""
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
    
    def compute_conflicts(rows, k):
        """Count total number of conflicting differences (duplicates across rows)."""
        diff_counts = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts
    
    def get_scope(rows):
        """Return the largest entry across all rows."""
        return max(max(row) for row in rows)
    
    def is_valid(rows, k):
        """Check if all differences are distinct across all rows."""
        return compute_conflicts(rows, k) == 0
    
    # Step 1: Greedy initialization
    rows = greedy_construct(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = get_scope(best_rows)
    best_conflicts = compute_conflicts(best_rows, k)
    
    # If already valid, we're done
    if best_conflicts == 0:
        return best_rows
    
    # Step 2: Simulated Annealing
    # We'll try to reduce conflicts and then minimize scope
    
    def sa_optimize(rows, k, n_rows, time_limit):
        """Simulated annealing to minimize conflicts, then scope."""
        start_time = time.time()
        current_rows = [list(r) for r in rows]
        current_conflicts = compute_conflicts(current_rows, k)
        current_scope = get_scope(current_rows)
        
        best_rows_sa = [list(r) for r in current_rows]
        best_conflicts_sa = current_conflicts
        best_scope_sa = current_scope
        
        # Temperature schedule
        T = 10.0
        cooling = 0.995
        min_T = 0.01
        
        while time.time() - start_time < time_limit and T > min_T:
            # Pick a random row and a random position to modify
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Don't change position 0 (always 0)
            
            # Try a random perturbation
            old_val = current_rows[r_idx][pos]
            # Propose a new value: random walk
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            
            # Ensure increasing order
            if new_val <= current_rows[r_idx][pos - 1]:
                new_val = current_rows[r_idx][pos - 1] + 1
            if pos < k and new_val >= current_rows[r_idx][pos + 1]:
                new_val = current_rows[r_idx][pos + 1] - 1
            
            if new_val == old_val:
                continue
            
            # Compute delta conflicts
            # Remove old differences involving position pos
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - current_rows[r_idx][i])
            for j in range(pos + 1, k + 1):
                old_diffs.add(current_rows[r_idx][j] - old_val)
            
            # Count how many of these old diffs are shared with other rows
            old_conflict_count = 0
            for d in old_diffs:
                # Count occurrences of d in all rows
                total_count = 0
                for r in range(n_rows):
                    row = current_rows[r]
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            if row[j] - row[i] == d:
                                total_count += 1
                if total_count > 1:
                    old_conflict_count += 1
            
            # Apply new value
            current_rows[r_idx][pos] = new_val
            
            # Compute new differences involving position pos
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - current_rows[r_idx][i])
            for j in range(pos + 1, k + 1):
                new_diffs.add(current_rows[r_idx][j] - new_val)
            
            # Count how many of these new diffs are shared with other rows
            new_conflict_count = 0
            for d in new_diffs:
                total_count = 0
                for r in range(n_rows):
                    row = current_rows[r]
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            if row[j] - row[i] == d:
                                total_count += 1
                if total_count > 1:
                    new_conflict_count += 1
            
            delta_conflicts = new_conflict_count - old_conflict_count
            new_scope = get_scope(current_rows)
            delta_scope = new_scope - current_scope
            
            # Acceptance criterion
            if delta_conflicts < 0:
                accept = True
            elif delta_conflicts == 0 and delta_scope < 0:
                accept = True
            else:
                # Metropolis acceptance
                delta_energy = delta_conflicts * 100 + delta_scope
                if delta_energy < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta_energy / T)
            
            if accept:
                current_conflicts = compute_conflicts(current_rows, k)
                current_scope = new_scope
                if current_conflicts < best_conflicts_sa or \
                   (current_conflicts == best_conflicts_sa and current_scope < best_scope_sa):
                    best_conflicts_sa = current_conflicts
                    best_scope_sa = current_scope
                    best_rows_sa = [list(r) for r in current_rows]
            else:
                # Revert
                current_rows[r_idx][pos] = old_val
            
            T *= cooling
        
        return best_rows_sa, best_conflicts_sa, best_scope_sa
    
    # Run SA with time budget
    remaining = deadline - time.time()
    if remaining > 5:
        sa_rows, sa_conflicts, sa_scope = sa_optimize(rows, k, n_rows, min(remaining - 5, 100))
        if sa_conflicts < best_conflicts or \
           (sa_conflicts == best_conflicts and sa_scope < best_scope):
            best_rows = sa_rows
            best_conflicts = sa_conflicts
            best_scope = sa_scope
    
    # Step 3: Per-row local polish
    # Try to reduce the scope by moving the largest element down
    def local_polish(rows, k, n_rows, time_limit):
        """Local search to reduce scope while maintaining validity."""
        start_time = time.time()
        current_rows = [list(r) for r in rows]
        current_scope = get_scope(current_rows)
        current_conflicts = compute_conflicts(current_rows, k)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for r_idx in range(n_rows):
                # Try to reduce the last element
                row = current_rows[r_idx]
                if len(row) < 2:
                    continue
                last_idx = len(row) - 1
                old_last = row[last_idx]
                
                # Try values from old_last-1 down to row[last_idx-1]+1
                for new_last in range(old_last - 1, row[last_idx - 1], -1):
                    row[last_idx] = new_last
                    new_conflicts = compute_conflicts(current_rows, k)
                    new_scope = get_scope(current_rows)
                    
                    if new_conflicts <= current_conflicts and new_scope < current_scope:
                        current_conflicts = new_conflicts
                        current_scope = new_scope
                        improved = True
                        break
                    else:
                        row[last_idx] = old_last
                        break
        
        return current_rows, current_conflicts, current_scope
    
    remaining = deadline - time.time()
    if remaining > 3:
        polished_rows, pol_conflicts, pol_scope = local_polish(
            best_rows, k, n_rows, min(remaining - 3, 30)
        )
        if pol_conflicts < best_conflicts or \
           (pol_conflicts == best_conflicts and pol_scope < best_scope):
            best_rows = polished_rows
            best_conflicts = pol_conflicts
            best_scope = pol_scope
    
    return best_rows
# EVOLVE-BLOCK-END
