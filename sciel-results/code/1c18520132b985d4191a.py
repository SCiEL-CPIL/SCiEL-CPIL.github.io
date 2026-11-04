import time
import random
from collections import Counter

def construct(n_rows, k):
    deadline = time.time() + 145.0
    
    # Total number of differences needed: n_rows * k(k+1)/2
    # Each difference must be unique, so max entry must be at least the largest difference.
    # The differences are from 1 up to max_entry.
    # We need n_rows * k(k+1)/2 distinct differences.
    # So max_entry >= n_rows * k(k+1)/2 is a loose bound (since diffs are 1..max_entry, we need max_entry >= count).
    # Actually, the differences are a_j - a_i. Since a_0=0, a_k is the max.
    # The largest difference in a row is a_k - a_0 = a_k.
    # So the global max entry is the largest a_k across all rows.
    # The set of all differences is a subset of {1, ..., max_entry}.
    # So we need max_entry >= total_diffs.
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Helper to check validity
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_max(rows):
        return max(max(r) for r in rows)

    # Phase 1: Greedy Initialization
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_max = get_max(best_rows)
    
    # Phase 2: Local Search with Simulated Annealing
    # We will try to minimize the max entry.
    # We start from a valid configuration (greedy).
    # Move: pick a row r and an index i (1 <= i <= k) and change the value.
    # We must maintain strictly increasing order within the row.
    # We also must maintain global uniqueness of differences.
    # Cost function: 
    # 1. Number of duplicate differences (violations)
    # 2. Max entry
    # We want to minimize violations first, then max entry.
    
    # To make local search effective, we might want to start from a "relaxed" state?
    # No, greedy gives a valid state. 
    # The problem is that greedy gets stuck in local optima where max is high.
    # We need to allow temporary invalid states (violations) to escape.
    
    current_rows = [r[:] for r in best_rows]
    
    # Precompute difference counts for fast update
    diff_counts = Counter()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff_counts[row[j] - row[i]] += 1
                
    def get_violations():
        # Count of differences that appear more than once
        v = 0
        for c in diff_counts.values():
            if c > 1:
                v += c - 1
        return v

    current_violations = get_violations()
    current_max = get_max(current_rows)
    
    # Cost: violations * large_weight + max_entry
    # We want to drive violations to 0 first.
    weight = 10000
    current_cost = current_violations * weight + current_max
    
    T = 100.0
    T_min = 0.01
    cooling = 0.999
    
    # Indices to pick: (row_idx, pos_idx) where pos_idx is 1..k
    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        r, i = random.choice(indices)
        old_val = current_rows[r][i]
        
        # Determine valid range for new_val
        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else best_max + 50 # Allow expansion
        
        # If we allow expansion, upper can be large.
        # But we want to shrink max.
        # Let's try values in a window around old_val, or random in [lower, upper].
        # To explore better, let's try a random value in [lower, min(upper, best_max + 10)]
        
        if random.random() < 0.5:
            # Local move
            new_val = old_val + random.choice([-1, 1])
            if new_val < lower or (i < k and new_val > upper):
                # Try other side
                new_val = old_val - random.choice([-1, 1])
                if new_val < lower or (i < k and new_val > upper):
                    continue
        else:
            # Random move
            # Bias towards smaller values if possible
            # But we must respect constraints.
            # Let's pick a random integer in [lower, upper]
            # To avoid huge jumps that break many constraints, maybe limit range?
            # For now, uniform.
            new_val = random.randint(lower, upper)
            
        if new_val == old_val:
            continue
            
        # Calculate delta in violations
        # Remove old differences, add new differences
        row = current_rows[r]
        removed_diffs = []
        added_diffs = []
        
        for j in range(len(row)):
            if j == i:
                continue
            if j < i:
                old_d = old_val - row[j]
                new_d = new_val - row[j]
                removed_diffs.append(old_d)
                added_diffs.append(new_d)
            else:
                old_d = row[j] - old_val
                new_d = row[j] - new_val
                removed_diffs.append(old_d)
                added_diffs.append(new_d)
                
        # Update counts
        for d in removed_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        for d in added_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
        new_violations = get_violations()
        new_max = get_max(current_rows)
        new_cost = new_violations * weight + new_max
        
        delta = new_cost - current_cost
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            # Accept
            current_rows[r][i] = new_val
            current_violations = new_violations
            current_max = new_max
            current_cost = new_cost
            
            if current_violations == 0 and current_max < best_max:
                best_max = current_max
                best_rows = [rr[:] for rr in current_rows]
        else:
            # Reject
            # Revert counts
            for d in added_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in removed_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            current_rows[r][i] = old_val
            
        T *= cooling
        
    return best_rows
