# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a simulated annealing approach with incremental penalty updates
    to efficiently search for compact difference triangle sets.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Greedy baseline
    def greedy_construct(n_rows, k):
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
    
    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
    def validate(rows, n_rows, k):
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
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
        best_max = max(max(r) for r in best_rows)
    
    # Simulated annealing
    def simulated_annealing(n_rows, k, initial_rows, max_iterations=500000):
        """
        Simulated annealing to minimize the largest entry.
        """
        rng = random.Random(42)
        
        # Work with a copy
        rows = [list(r) for r in initial_rows]
        current_max = max(max(r) for r in rows)
        
        # Precompute all differences and their counts
        def compute_penalty(rows):
            """Compute the number of duplicate differences (penalty)"""
            diff_count = {}
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        diff_count[d] = diff_count.get(d, 0) + 1
            # Penalty is the number of differences that appear more than once
            penalty = 0
            for d, cnt in diff_count.items():
                if cnt > 1:
                    penalty += cnt - 1
            return penalty
        
        def compute_max_entry(rows):
            return max(max(r) for r in rows)
        
        # Initial state
        current_penalty = compute_penalty(rows)
        current_cost = current_max * 1000 + current_penalty  # Weight max entry heavily
        
        best_local = [list(r) for r in rows]
        best_local_cost = current_cost
        best_local_max = current_max
        best_local_penalty = current_penalty
        
        T = 100.0  # Initial temperature
        cooling = 0.9995  # Cooling rate
        min_T = 0.1
        
        iteration = 0
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 2.0:
            iteration += 1
            
            if T < min_T:
                break
            
            if no_improve > max_no_improve:
                # Restart with perturbation
                T = 50.0
                no_improve = 0
                # Perturb current solution
                for _ in range(5):
                    ri = rng.randrange(n_rows)
                    pos = rng.randrange(1, k + 1)
                    delta = rng.choice([-1, 1]) * rng.randint(1, 3)
                    old_val = rows[ri][pos]
                    new_val = old_val + delta
                    if new_val > rows[ri][pos - 1] and (pos == k or new_val < rows[ri][pos + 1]):
                        rows[ri][pos] = new_val
                current_max = compute_max_entry(rows)
                current_penalty = compute_penalty(rows)
                current_cost = current_max * 1000 + current_penalty
                continue
            
            # Choose a random move
            ri = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)
            old_val = rows[ri][pos]
            
            # Choose a new value
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
            
            # Check validity of new position
            valid = True
            if new_val <= rows[ri][pos - 1]:
                valid = False
            if pos < k and new_val >= rows[ri][pos + 1]:
                valid = False
            if new_val <= 0:
                valid = False
            
            if not valid:
                T *= cooling
                continue
            
            # Compute cost change incrementally
            # Remove old differences involving position pos
            old_diffs = set()
            for i in range(len(rows[ri])):
                if i != pos:
                    d = abs(rows[ri][pos] - rows[ri][i])
                    old_diffs.add(d)
            
            # Add new differences
            new_diffs = set()
            for i in range(len(rows[ri])):
                if i != pos:
                    d = abs(new_val - rows[ri][i])
                    new_diffs.add(d)
            
            # Simpler: just compute new cost
            rows[ri][pos] = new_val
            new_max = compute_max_entry(rows)
            
            # Quick penalty check - only check affected differences
            # For efficiency, we'll do a partial check
            affected_diffs = old_diffs | new_diffs
            penalty_change = 0
            for d in affected_diffs:
                # Count how many times d appears
                count = 0
                for row in rows:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            if row[j] - row[i] == d:
                                count += 1
                if d in old_diffs and d not in new_diffs:
                    # d was removed from this row
                    pass
                elif d in new_diffs and d not in old_diffs:
                    # d was added to this row
                    pass
            
            # For simplicity, compute full penalty for now (slower but correct)
            new_penalty = compute_penalty(rows)
            new_cost = new_max * 1000 + new_penalty
            
            delta_cost = new_cost - current_cost
            
            # Accept or reject
            if delta_cost < 0 or rng.random() < pow(2.718281828, -delta_cost / T):
                # Accept
                current_cost = new_cost
                current_max = new_max
                current_penalty = new_penalty
                
                if new_max < best_local_max or (new_max == best_local_max and new_penalty < best_local_penalty):
                    best_local_max = new_max
                    best_local_penalty = new_penalty
                    best_local = [list(r) for r in rows]
                    best_local_cost = new_cost
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Reject
                rows[ri][pos] = old_val
                no_improve += 1
            
            T *= cooling
        
        return best_local, best_local_max, best_local_penalty
    
    # Run simulated annealing
    sa_rows, sa_max, sa_penalty = simulated_annealing(n_rows, k, best_rows)
    
    if sa_max < best_max and sa_penalty == 0:
        best_rows = sa_rows
        best_max = sa_max
    
    # Final validation
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
