"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

def _interleaved_greedy(n_rows, k, seed=0):
    """Interleaved greedy: build all rows simultaneously, one element at a time."""
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    # Build up to k+1 elements in each row
    for step in range(1, k + 1):
        # Shuffle row order for diversity
        order = list(range(n_rows))
        random.shuffle(order)
        for r in order:
            row = rows[r]
            x = row[-1] + 1
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    
    return rows

def _construct_from_gaps(n_rows, k, gap_list):
    """
    Construct rows from a list of gaps. Each row has k gaps between k+1 elements.
    gap_list[r][g] is the gap between element g and g+1 in row r.
    """
    rows = []
    for r in range(n_rows):
        row = [0]
        for g in range(k):
            row.append(row[-1] + gap_list[r][g])
        rows.append(row)
    return rows

def _validate_rows(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _optimize(rows, n_rows, k, time_limit=100.0, seed=42):
    """
    Optimized local search using difference counts with efficient move checking.
    Returns the best rows found.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 20
    diff_counts = [0] * (diff_size + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    
    def try_reduce(r, pos):
        """Try to reduce current_rows[r][pos] to its minimum valid value. Returns True if changed."""
        row = current_rows[r]
        old_val = row[pos]
        lower_bound = row[pos - 1] + 1 if pos > 0 else 1
        
        if lower_bound >= old_val:
            return False
        
        # Precompute old differences
        old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
        
        # Remove old diffs from counts
        for d in old_diffs:
            if d < len(diff_counts):
                diff_counts[d] -= 1
        
        # Try values from lower_bound upward until we find a valid one
        for new_val in range(lower_bound, old_val):
            # Check if this new_val is valid
            valid = True
            for i in range(len(row)):
                if i == pos:
                    continue
                d = abs(new_val - row[i])
                if d < len(diff_counts) and diff_counts[d] > 0:
                    valid = False
                    break
            if valid:
                # Apply move
                row[pos] = new_val
                new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                for d in new_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] += 1
                return True
        
        # Restore old diffs
        for d in old_diffs:
            if d < len(diff_counts):
                diff_counts[d] += 1
        return False

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            for pos in range(len(row) - 1, 0, -1):
                if (time.time() - start_time) > time_limit:
                    break
                if try_reduce(r, pos):
                    improved = True
                    
    current_scope = get_scope()
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [list(r) for r in current_rows]
        
    return best_rows

def _optimize_simulated_annealing(rows, n_rows, k, time_limit=100.0, seed=42):
    """
    Simulated annealing with adaptive step size and efficient move checking.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 50
    diff_counts = [0] * (diff_size + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    current_scope = best_scope
    
    T = 15.0
    T_min = 0.5
    alpha = 0.995
    step_scale = max_val * 0.3  # Initial step size proportional to scope
    
    while (time.time() - start_time) < time_limit and T > T_min:
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        row = current_rows[r]
        old_val = row[pos]
        
        # Adaptive step size: larger when T is high, smaller when T is low
        step = max(1, int(step_scale * (T / 15.0)))
        delta = random.randint(-step, step)
        new_val = old_val + delta
        
        lower_bound = row[pos - 1] + 1
        upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 30
        
        if new_val < lower_bound:
            new_val = lower_bound
        if new_val > upper_bound:
            new_val = upper_bound
            
        if new_val == old_val:
            T *= alpha
            continue
            
        old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
        new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
        
        for d in old_diffs:
            if d < len(diff_counts):
                diff_counts[d] -= 1
            
        valid = True
        for d in new_diffs:
            if d < len(diff_counts) and diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            for d in old_diffs:
                if d < len(diff_counts):
                    diff_counts[d] += 1
            T *= alpha
            continue
            
        row[pos] = new_val
        for d in new_diffs:
            if d < len(diff_counts):
                diff_counts[d] += 1
            
        new_scope = get_scope()
        
        if new_scope < current_scope:
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            delta_e = new_scope - current_scope
            if random.random() < pow(2.71828, -delta_e / T):
                current_scope = new_scope
            else:
                row[pos] = old_val
                for d in new_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] -= 1
                for d in old_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] += 1
                    
        T *= alpha
        
    return best_rows

def _final_polish(rows, n_rows, k, time_limit=10.0):
    """
    Final high-precision polish: repeatedly try to reduce each element to its minimum valid value.
    """
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    
    max_val = max(max(r) for r in current_rows)
    diff_counts = [0] * (max_val + 30)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                if lower_bound >= old_val:
                    continue
                
                old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                
                for d in old_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] -= 1
                
                found = False
                for new_val in range(lower_bound, old_val):
                    valid = True
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(new_val - row[i])
                        if d < len(diff_counts) and diff_counts[d] > 0:
                            valid = False
                            break
                    if valid:
                        row[pos] = new_val
                        new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                        for d in new_diffs:
                            if d < len(diff_counts):
                                diff_counts[d] += 1
                        improved = True
                        found = True
                        break
                
                if not found:
                    for d in old_diffs:
                        if d < len(diff_counts):
                            diff_counts[d] += 1
    
    return current_rows

def _matching_based_construct(n_rows, k, time_limit=30.0):
    """
    Construct rows using a matching-based approach.
    We model this as an assignment problem where we assign distinct differences to pairs.
    """
    start_time = time.time()
    
    # Number of pairs per row: C(k+1, 2)
    pairs_per_row = k * (k + 1) // 2
    total_pairs = n_rows * pairs_per_row
    
    # We need to assign total_pairs distinct positive integers
    # Start with an estimate of the maximum value needed
    # The counting bound suggests around 105 for (7,5), but let's be safe
    M = 150  # Upper bound on difference values
    
    # Create all possible pairs (row, i, j) where i < j
    all_pairs = []
    for r in range(n_rows):
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                all_pairs.append((r, i, j))
    
    # For each pair, we need to assign a distinct difference value
    # The difference for pair (r, i, j) is row[r][j] - row[r][i]
    # We want to minimize the maximum difference
    
    # Approach: Start with a greedy solution, then use local search to improve
    # First, get a greedy solution
    greedy_rows = _interleaved_greedy(n_rows, k, seed=0)
    greedy_scope = max(max(r) for r in greedy_rows)
    
    # Now try to improve using matching
    # We'll use the Hungarian algorithm to find a better assignment of differences
    
    # Build cost matrix: for each pair, cost = assigned_difference
    # But we need to ensure consistency within each row
    
    # Alternative approach: Use the greedy solution as a starting point,
    # then try to swap differences between pairs to reduce the maximum
    
    best_rows = [list(r) for r in greedy_rows]
    best_scope = greedy_scope
    
    # Extract current differences
    def get_differences(rows):
        diffs = {}
        for r in range(n_rows):
            row = rows[r]
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    diffs[(r, i, j)] = row[j] - row[i]
        return diffs
    
    current_diffs = get_differences(best_rows)
    
    # Local search: try to swap differences between pairs if it reduces the max
    def try_swap(diff1, diff2):
        """Try to swap differences between two pairs. Returns new rows if successful."""
        (r1, i1, j1), (r2, i2, j2) = diff1, diff2
        
        # Check if pairs are in the same row
        if r1 == r2:
            return None
        
        # Swap the differences
        d1 = current_diffs[diff1]
        d2 = current_diffs[diff2]
        
        # Check if swapping is valid
        # We need to check if the new differences are consistent with the row constraints
        
        # For simplicity, let's just try to reduce the maximum difference
        # by moving a large difference to a position where it fits better
        
        # This is complex, so let's use a simpler heuristic:
        # Try to reduce the largest difference by swapping with a smaller one
        # from a row where it would still be valid
        
        return None
    
    # Since the matching approach is complex, let's focus on improving the local search
    # with better move generation
    
    # Try multiple seeds for greedy
    for seed in range(10):
        if time.time() - start_time > time_limit:
            break
        
        rows = _interleaved_greedy(n_rows, k, seed=seed)
        sc = max(max(r) for r in rows)
        
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]
    
    return best_rows

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # 1. Generate diverse initial solutions using interleaved greedy
    max_greedy_seeds = 10
    
    for seed in range(max_greedy_seeds):
        if time.time() > deadline - 100.0: # Leave 100s for optimization
            break
            
        rows = _interleaved_greedy(n_rows, k, seed=seed)
        sc = max(max(r) for r in rows)
        
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # 2. Optimization Phase
    time_remaining = deadline - time.time()
    
    if time_remaining > 15.0:
        # Allocate: 40% HC, 40% SA, 15% polish, 5% buffer
        t_hc = time_remaining * 0.40
        t_sa = time_remaining * 0.40
        t_polish = time_remaining * 0.15
        
        if t_hc > 5.0:
            optimized_hc = _optimize(best_rows, n_rows, k, time_limit=t_hc, seed=42)
            sc_hc = max(max(r) for r in optimized_hc)
            if sc_hc < best_scope:
                best_scope = sc_hc
                best_rows = optimized_hc
        
        if (deadline - time.time()) > t_sa + 5.0:
            optimized_sa = _optimize_simulated_annealing(best_rows, n_rows, k, time_limit=t_sa, seed=43)
            sc_sa = max(max(r) for r in optimized_sa)
            if sc_sa < best_scope:
                best_scope = sc_sa
                best_rows = optimized_sa
        
        t_polish_actual = deadline - time.time() - 5.0
        if t_polish_actual > 2.0:
            polished = _final_polish(best_rows, n_rows, k, time_limit=t_polish_actual)
            sc_polish = max(max(r) for r in polished)
            if sc_polish < best_scope:
                best_scope = sc_polish
                best_rows = polished
                
    return best_rows
# EVOLVE-BLOCK-END
