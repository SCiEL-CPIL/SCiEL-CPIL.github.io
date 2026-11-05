import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(r) for r in rows)
    
    def get_row_diffs(row):
        """Get set of all positive differences within a single row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            for d in get_row_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return len(all_diffs) == total_diffs
    
    def greedy_construct():
        """Build initial solution using greedy method."""
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
    
    def simulated_annealing(initial_rows, max_iterations=200000):
        """Simulated Annealing with incremental updates and biased mutation."""
        current = [r[:] for r in initial_rows]
        current_scope = get_scope(current)
        
        # Precompute difference sets for each row
        row_diffs = [get_row_diffs(r) for r in current]
        all_diffs = set()
        for rd in row_diffs:
            all_diffs |= rd
        
        best = [r[:] for r in current]
        best_scope = current_scope
        best_row_diffs = [set(rd) for rd in row_diffs]
        best_all_diffs = set(all_diffs)
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iterations))
        T = T_start
        
        rng = random.Random(42)
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random row and position to mutate
            ri = rng.randint(0, len(current) - 1)
            pi = rng.randint(1, k)
            
            old_val = current[ri][pi]
            
            # Determine valid range for new value
            lo = current[ri][pi - 1] + 1
            if pi == k:
                hi = current_scope + 5  # Allow slight expansion
            else:
                hi = current[ri][pi + 1] - 1
            
            if lo > hi:
                T *= cooling_rate
                continue
            
            # Biased mutation: prefer smaller values to reduce scope
            # Generate candidate with bias toward smaller values
            if hi - lo < 20:
                # Small range: try all or random subset
                candidates = list(range(lo, hi + 1))
                rng.shuffle(candidates)
                new_val = candidates[0]
            else:
                # Large range: biased toward smaller values
                # Use exponential bias: higher probability for smaller values
                span = hi - lo + 1
                # Sample with bias toward lo
                bias_factor = 0.3  # Lower = more bias toward small values
                weighted_pos = int(span * (bias_factor + rng.random() * (1 - bias_factor)))
                new_val = lo + weighted_pos
            
            if new_val == old_val:
                T *= cooling_rate
                continue
            
            # Check strictly increasing
            new_row = current[ri][:]
            new_row[pi] = new_val
            if new_row != sorted(new_row):
                T *= cooling_rate
                continue
            
            # Compute new differences for this row
            new_row_diffs = get_row_diffs(new_row)
            
            # Check internal duplicates in new row
            if len(new_row_diffs) != k * (k + 1) // 2:
                T *= cooling_rate
                continue
            
            # Incremental update: remove old diffs, add new diffs
            # Old diffs that are being removed
            old_row_diffs = row_diffs[ri]
            
            # Check if new diffs conflict with other rows' diffs
            # Other rows' diffs = all_diffs - old_row_diffs
            other_diffs = all_diffs - old_row_diffs
            
            # New diffs that would be added
            new_diffs_to_add = new_row_diffs - old_row_diffs
            old_diffs_to_remove = old_row_diffs - new_row_diffs
            
            # Check conflicts: new_diffs_to_add must not be in other_diffs
            if new_diffs_to_add & other_diffs:
                T *= cooling_rate
                continue
            
            # Compute new scope
            new_scope = max(
                max(max(r) for idx, r in enumerate(current) if idx != ri),
                max(new_row)
            )
            
            delta_scope = new_scope - current_scope
            
            # Acceptance criterion
            if delta_scope <= 0:
                # Always accept improvements or equal
                accept = True
            else:
                # Accept with probability exp(-delta/T)
                if T > 0:
                    accept = rng.random() < math.exp(-delta_scope / T)
                else:
                    accept = False
            
            if accept:
                # Update state
                current[ri] = new_row
                row_diffs[ri] = new_row_diffs
                
                # Update all_diffs incrementally
                all_diffs -= old_diffs_to_remove
                all_diffs |= new_diffs_to_add
                
                current_scope = new_scope
                
                # Update best if improved
                if new_scope < best_scope:
                    best = [r[:] for r in current]
                    best_scope = new_scope
                    best_row_diffs = [set(rd) for rd in row_diffs]
                    best_all_diffs = set(all_diffs)
            
            # Cool down
            T *= cooling_rate
        
        return best, best_scope
    
    # Start with greedy construction
    best_rows = greedy_construct()
    best_scope = get_scope(best_rows)
    
    # Run simulated annealing
    if time.time() < deadline - 3:
        sa_rows, sa_scope = simulated_annealing(best_rows, max_iterations=150000)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Final validity check
    if not is_valid(best_rows):
        # Fallback to greedy if SA produced invalid result
        greedy_rows = greedy_construct()
        if is_valid(greedy_rows):
            best_rows = greedy_rows
            best_scope = get_scope(best_rows)
    
    return best_rows
