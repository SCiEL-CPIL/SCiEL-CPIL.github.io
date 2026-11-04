"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are globally distinct.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def row_diffs(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check if rows form a valid difference set."""
        all_diffs = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            rd = row_diffs(row)
            if rd & all_diffs:
                return False
            all_diffs |= rd
        return True

    def scope(rows):
        """Return the largest entry across all rows."""
        return max(max(r) for r in rows)

    def greedy_construct(n_rows, k, seed=None):
        """Greedy construction with optional randomization."""
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try to find the next smallest value that doesn't conflict
                found = False
                # Start from row[-1] + 1 and go up
                candidate = row[-1] + 1
                while not found:
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        row.append(candidate)
                        used |= new_diffs
                        found = True
                    else:
                        candidate += 1
            rows.append(row)
        
        return rows

    def local_search_improve(rows, deadline):
        """
        Local search: try to reduce the scope by moving elements to smaller positions.
        We iterate over all rows and all positions (from right to left) and try to
        decrease the value while maintaining validity.
        """
        rows = [list(r) for r in rows]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Compute current used diffs
            used = set()
            for r in rows:
                used |= row_diffs(r)
            
            for i in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[i]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    lower_bound = row[pos - 1] + 1
                    # Try to decrease the value
                    for new_val in range(lower_bound, old_val):
                        # Check if this move is valid
                        # Remove old diffs involving pos
                        old_diffs = set()
                        for j in range(pos):
                            old_diffs.add(old_val - row[j])
                        
                        # Compute new diffs involving pos
                        new_diffs = set()
                        for j in range(pos):
                            new_diffs.add(new_val - row[j])
                        
                        # Check conflicts
                        # The "available" diffs are used - old_diffs
                        available = used - old_diffs
                        
                        # New diffs must not be in available
                        if new_diffs & available:
                            continue
                        
                        # Also, new diffs must not conflict with other diffs in the same row
                        # that don't involve pos
                        other_row_diffs = set()
                        for a in range(pos):
                            for b in range(a + 1, pos):
                                other_row_diffs.add(row[b] - row[a])
                        
                        if new_diffs & other_row_diffs:
                            continue
                        
                        # If we get here, the move is valid
                        row[pos] = new_val
                        improved = True
                        # Update used set
                        used -= old_diffs
                        used |= new_diffs
                        # Recompute row diffs for consistency? No, we updated used.
                        # But we need to be careful: row_diffs(row) would be different.
                        # Let's just break and recompute used from scratch in the next iteration
                        # for safety, or continue carefully.
                        # For now, let's break out of the inner loop and recompute used.
                        break
                    # If we broke out of the new_val loop due to success, we should
                    # recompute used to be safe, or at least update it correctly.
                    # The update above was: used -= old_diffs; used |= new_diffs.
                    # This is correct for the global set.
        
        # Final verification
        if is_valid(rows):
            return rows
        else:
            return [list(r) for r in rows]  # Return original if invalid

    def simulated_annealing(rows, deadline):
        """
        Simulated annealing to further reduce scope.
        Moves: pick a random row and position, try to change value to a random smaller
        or larger value (within bounds) that maintains validity.
        Accept moves that reduce scope, or with probability exp(-delta/T).
        """
        rows = [list(r) for r in rows]
        current_scope = scope(rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        
        T = 10.0
        cooling = 0.999
        
        used = set()
        for r in rows:
            used |= row_diffs(r)
        
        while time.time() < deadline and T > 0.1:
            # Pick a random row and position
            i = random.randint(0, len(rows) - 1)
            pos = random.randint(1, len(rows[i]) - 1)
            
            old_val = rows[i][pos]
            lower_bound = rows[i][pos - 1] + 1
            upper_bound = old_val + 10  # Allow some expansion
            
            # Try a random new value
            new_val = random.randint(lower_bound, upper_bound)
            if new_val == old_val:
                continue
            
            # Compute old diffs involving pos
            old_diffs = set()
            for j in range(pos):
                old_diffs.add(old_val - rows[i][j])
            
            # Compute new diffs involving pos
            new_diffs = set()
            for j in range(pos):
                new_diffs.add(new_val - rows[i][j])
            
            # Check validity
            available = used - old_diffs
            if new_diffs & available:
                continue
            
            # Check internal row conflicts
            other_row_diffs = set()
            for a in range(pos):
                for b in range(a + 1, pos):
                    other_row_diffs.add(rows[i][b] - rows[i][a])
            
            if new_diffs & other_row_diffs:
                continue
            
            # Move is valid
            # Compute new scope
            new_scope = max(max(rows[i]), new_val)
            # Actually, we need to check if new_val is the max in the row
            # and if that row is the max in the set.
            # Simplified: just compute scope of the whole set after the move
            # But that's expensive. Let's approximate.
            # If new_val > old_val, scope might increase.
            # If new_val < old_val, scope might decrease if old_val was the max.
            
            # Let's just do the move and compute scope if it's promising
            if new_val <= old_val:
                # Potential improvement
                rows[i][pos] = new_val
                used -= old_diffs
                used |= new_diffs
                new_scope = scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                # Accept (always accept improvements)
                current_scope = new_scope
            else:
                # Potential worsening, accept with probability
                delta = 1.0
                prob = pow(2.71828, -delta / T)
                if random.random() < prob:
                    rows[i][pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    current_scope = scope(rows)
            
            T *= cooling
        
        # Return the best found
        if is_valid(best_rows):
            return best_rows
        else:
            return rows

    # Main algorithm
    # Phase 1: Greedy baseline
    rows = greedy_construct(n_rows, k, seed=42)
    best_rows = rows
    best_scope = scope(rows)
    
    # Phase 2: Local search improvement
    if time.time() < deadline - 20.0:
        improved = local_search_improve(best_rows, deadline - 20.0)
        if scope(improved) < best_scope:
            best_rows = improved
            best_scope = scope(improved)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5.0:
        sa_rows = simulated_annealing(best_rows, deadline - 5.0)
        if scope(sa_rows) < best_scope:
            best_rows = sa_rows
            best_scope = scope(sa_rows)
    
    # Final verification
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_construct(n_rows, k, seed=42)
    
    return best_rows
