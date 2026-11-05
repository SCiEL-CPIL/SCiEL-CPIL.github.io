import time
import random
import copy
import math

def construct(n_rows, k):
    start_time = time.time()
    # Evaluator kills at 150s, task says 120s. Use 110s to be safe.
    deadline = start_time + 110
    
    # 1. Greedy Initialization with randomization
    def greedy_init(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                diffs = {x - a for a in row}
                if not (diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(diffs)
            rows.append(row)
        return rows

    # 2. Compute Initial State
    def get_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_penalty(counts):
        pen = 0
        for c in counts.values():
            if c > 1:
                pen += c - 1
        return pen

    def get_M(rows):
        return max(max(r) for r in rows)

    def compute_energy(M, penalty, T):
        if penalty == 0:
            return float(M)
        # Smooth surrogate: M + lambda * penalty, but with temperature-dependent lambda
        # Higher T -> lower lambda (allow more conflicts to explore)
        # Lower T -> higher lambda (strictly avoid conflicts)
        # Use a smooth transition
        lambda_pen = 100.0 * (1.0 + T / 50.0)
        return M + lambda_pen * penalty

    # Multi-start SA
    best_rows = None
    best_M = float('inf')
    
    # Number of restarts
    max_restarts = 5
    time_per_restart = (deadline - start_time) / max_restarts
    
    for restart in range(max_restarts):
        # Check if we have enough time for this restart
        if time.time() > deadline - 5:
            break
            
        # Initialize
        if restart == 0:
            curr_rows = greedy_init()
        else:
            # Random perturbation of the best solution found so far, or fresh greedy
            if random.random() < 0.5:
                curr_rows = greedy_init()
            else:
                curr_rows = copy.deepcopy(best_rows)
                # Perturb some elements
                for _ in range(3):
                    r_idx = random.randint(0, n_rows - 1)
                    j_idx = random.randint(1, k)
                    row = curr_rows[r_idx]
                    low = row[j_idx - 1] + 1
                    high = row[j_idx + 1] - 1 if j_idx < k else max(max(r) for r in curr_rows) + 10
                    if high >= low:
                        row[j_idx] = random.randint(low, high)
                        # Fix ordering if violated
                        for i in range(1, k + 1):
                            if row[i] <= row[i-1]:
                                row[i] = row[i-1] + 1
        
        curr_counts = get_diff_counts(curr_rows)
        curr_penalty = get_penalty(curr_counts)
        curr_M = get_M(curr_rows)
        
        # SA Parameters
        T = 50.0
        cooling = 0.999
        curr_energy = compute_energy(curr_M, curr_penalty, T)
        
        # Local restart deadline
        local_deadline = start_time + (restart + 1) * time_per_restart
        
        # SA Loop
        while time.time() < local_deadline:
            # Generate Move
            r_idx = random.randint(0, n_rows - 1)
            row = curr_rows[r_idx]
            
            # Pick an index to change. Index 0 is fixed at 0.
            j_idx = random.randint(1, k)
            
            old_val = row[j_idx]
            
            # Determine valid range for new value
            low = row[j_idx - 1] + 1
            # Allow expansion slightly beyond current M to escape local minima
            high = row[j_idx + 1] - 1 if j_idx < k else curr_M + 10
            
            if high < low:
                continue
                
            new_val = random.randint(low, high)
            
            if new_val == old_val:
                continue
            
            # Remove old diffs and update penalty incrementally
            for i in range(j_idx):
                d = old_val - row[i]
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
                    
            for h in range(j_idx + 1, k + 1):
                d = row[h] - old_val
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
            
            # Tentatively set new value
            row[j_idx] = new_val
            
            # Add new diffs and update penalty incrementally
            for i in range(j_idx):
                d = new_val - row[i]
                c = curr_counts.get(d, 0)
                if c >= 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1
                
            for h in range(j_idx + 1, k + 1):
                d = row[h] - new_val
                c = curr_counts.get(d, 0)
                if c >= 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1
                
            # Calculate new M
            other_max = 0
            for r in curr_rows:
                if r is not row:
                    if max(r) > other_max:
                        other_max = max(r)
            new_M = max(max(row), other_max)
            
            new_energy = compute_energy(new_M, curr_penalty, T)
            
            delta_E = new_energy - curr_energy
            
            # Acceptance criterion
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                curr_energy = new_energy
                curr_M = new_M
                
                if curr_penalty == 0:
                    if curr_M < best_M:
                        best_M = curr_M
                        best_rows = copy.deepcopy(curr_rows)
            else:
                # Revert
                row[j_idx] = old_val
                
                # Remove new diffs (which were added)
                for i in range(j_idx):
                    d = new_val - row[i]
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                        
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                
                # Add old diffs (which were removed)
                for i in range(j_idx):
                    d = old_val - row[i]
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1
                    
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - old_val
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1

                # Recalculate M for simplicity
                curr_M = max(max(r) for r in curr_rows)

            # Cool down
            T *= cooling
            if T < 1.0:
                T = 1.0

    if best_rows is None:
        best_rows = greedy_init()
        
    return best_rows
