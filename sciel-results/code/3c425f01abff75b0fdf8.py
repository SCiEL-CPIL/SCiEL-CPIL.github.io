import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    L = k + 1  # Length of each row
    diffs_per_row = k * (k + 1) // 2

    def scope_of(rows):
        return max(max(r) for r in rows)

    def get_diffs(row):
        """Generate the list of differences for a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def pure_greedy():
        """Construct a valid solution using a greedy approach."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows

    def simulated_annealing(rows, time_limit):
        """
        SA with a composite energy function that allows invalid states.
        Energy = w_scope * scope^2 + w_coll * collisions
        """
        rows = [r[:] for r in rows]
        n = len(rows)
        
        # Precompute differences for each row
        row_diffs = [get_diffs(r) for r in rows]
        
        # Global difference counts
        # Max possible difference is roughly 2 * scope. We'll use a dict for sparse counts.
        diff_counts = {}
        for diffs in row_diffs:
            for d in diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        def compute_energy(rows, diff_counts):
            scope = max(max(r) for r in rows)
            # Collisions: sum of (count - 1) for all d where count > 1
            collisions = 0
            for d, c in diff_counts.items():
                if c > 1:
                    collisions += c - 1
            # Weights: Scope is the primary objective. Collisions are a penalty to guide towards validity.
            # Tuning weights: Scope squared is large. Collisions should be significant enough to push towards validity.
            w_scope = 1.0
            w_coll = 50.0  # Penalize collisions heavily
            return w_scope * (scope * scope) + w_coll * collisions

        current_energy = compute_energy(rows, diff_counts)
        best_rows = [r[:] for r in rows]
        best_energy = current_energy
        best_scope = scope_of(rows)
        
        T = 100.0
        T_min = 0.1
        alpha = 0.9995
        it = 0
        last_improvement = time.time()
        
        while time.time() < time_limit:
            it += 1
            if it % 1000 == 0:
                T *= alpha
                if T < T_min: T = T_min
                if time.time() - last_improvement > 5.0:
                    T = 50.0  # Reheat
                    last_improvement = time.time()

            # Pick a random row and a random element (except the first 0)
            ri = random.randrange(n)
            ci = random.randrange(1, L)
            
            old_val = rows[ri][ci]
            old_diffs = row_diffs[ri]
            
            # Determine range for new value
            lower = rows[ri][ci - 1] + 1
            upper = (rows[ri][ci + 1] - 1) if ci + 1 < L else max(old_val, scope_of(rows)) + 20
            
            if upper < lower:
                continue
                
            # Generate new value
            # Bias towards smaller values to reduce scope, but allow some exploration
            shift = random.randint(-15, 5)
            nv = old_val + shift
            if nv < lower: nv = lower
            if nv > upper: nv = upper
            
            if nv == old_val:
                continue
                
            # Update differences for the modified row
            new_row = rows[ri][:]
            new_row[ci] = nv
            new_diffs = get_diffs(new_row)
            
            # Check strict increasing (should be guaranteed by bounds, but safe to check)
            # Since we picked ci, and bounds are based on neighbors, it's strictly increasing.
            
            # Update global diff_counts
            # Remove old diffs
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Add new diffs
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            # Compute new energy
            new_energy = compute_energy(rows, diff_counts)
            
            delta = new_energy - current_energy
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept move
                current_energy = new_energy
                rows[ri] = new_row
                row_diffs[ri] = new_diffs
                
                # Update best if valid and better scope
                # Check validity: all counts must be 1
                is_valid = True
                for d, c in diff_counts.items():
                    if c > 1:
                        is_valid = False
                        break
                
                if is_valid:
                    current_scope = scope_of(rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                        last_improvement = time.time()
            else:
                # Reject move: revert diff_counts
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1

        # Ensure we return a valid solution. If best_rows is not valid, we might need to fix it?
        # The problem requires a valid solution. If we end up with an invalid best_rows, we should return the last valid one or try to repair.
        # In this implementation, best_rows is only updated if is_valid is True.
        # If no valid solution was found during SA (unlikely if start is valid), we return the initial rows.
        return best_rows, best_scope

    # ---------- Main Search ----------
    
    # 1. Initial Greedy Construction
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)
    
    # 2. Simulated Annealing
    # Allocate most of the time to SA
    t_sa_end = deadline - 5.0
    
    # Run SA starting from the greedy solution
    # We might run multiple SA instances if time permits, but one long SA with reheat is often robust.
    improved, new_sc = simulated_annealing(best_rows, time_limit=t_sa_end)
    
    if new_sc < best_scope:
        best_rows = improved
        best_scope = new_sc

    return best_rows
