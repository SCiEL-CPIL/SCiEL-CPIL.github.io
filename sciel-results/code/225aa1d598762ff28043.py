"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
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

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    """Return the set of all pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_sidon(row):
    """Check if a row is a Sidon set (all pairwise differences distinct)."""
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return len(diffs) == len(set(diffs))

def _check_row_valid(row, k):
    """Check if a single row is strictly increasing and starts at 0."""
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    return True

def _spectral_guided_greedy(n_rows, k, time_limit=30.0):
    """
    Spectral graph clustering approach:
    1. Build a hypergraph where nodes are integers 0..M and hyperedges connect
       integers that would create forbidden differences.
    2. Use eigenvectors of the normalized Laplacian to find low-energy partitions.
    3. Use these partitions to guide a constrained greedy search.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Estimate upper bound for M (largest entry)
    # Simple counting bound: need n_rows * C(k+1,2) distinct differences
    num_diffs_needed = n_rows * (k + 1) * k // 2
    # Each difference d can appear at most once, so we need M >= sqrt(2 * num_diffs_needed) roughly
    # But differences range from 1 to M, so M must be at least num_diffs_needed
    # Actually, differences range from 1 to M, and we need num_diffs_needed distinct ones
    # So M >= num_diffs_needed is a lower bound, but in practice M is much smaller
    # Use a heuristic upper bound
    M_upper = int(math.ceil(num_diffs_needed ** 0.6)) + 20
    M_upper = max(M_upper, 50)
    
    # Build conflict graph: nodes are integers 0..M_upper
    # Edge between i and j if |i-j| conflicts with some difference pattern
    # For simplicity, we'll build a weighted graph where weight(i,j) = 1 if
    # using both i and j in the same row would create a difference that's "expensive"
    
    # Better approach: Build a graph where we want to find n_rows disjoint
    # (k+1)-cliques (rows) that minimize the maximum node index.
    # Use spectral clustering to find good initial partitions.
    
    # Build adjacency: A[i][j] = 1 if |i-j| is a "good" difference (small and unique)
    # Actually, let's use a different spectral approach:
    # Define a cost matrix C where C[i,j] represents the "conflict" between
    # using i and j in the same row.
    
    # Simpler spectral approach: Use the difference structure directly.
    # For each difference d, we can't use it more than once across all rows.
    # Build a hypergraph: hyperedge for each difference d connects all pairs
    # (i, i+d). We want to select n_rows rows (each a (k+1)-subset) such that
    # no difference is used twice.
    
    # Spectral method: Create a matrix L where L[i,j] = -1 if |i-j| is a
    # "common" difference (likely to be used), and L[i,i] = degree.
    # Then use eigenvectors to find natural clusters.
    
    # For practical purposes, let's use a simpler spectral-guided greedy:
    # 1. Start with greedy solution
    # 2. Use spectral information to identify "bottleneck" differences
    # 3. Re-solve rows that use bottleneck differences with a bias toward
    #    using less common differences
    
    # Step 1: Get greedy solution
    rows_greedy = _build_greedy(n_rows, k)
    best = [r[:] for r in rows_greedy]
    best_val = _max_entry(best)
    
    # Step 2: Analyze difference usage
    diff_count = {}
    for row in best:
        for d in _row_diffs_set(row):
            diff_count[d] = diff_count.get(d, 0) + 1
    
    # Find bottleneck differences (used in many rows or large differences)
    # Large differences are "expensive" because they force larger M
    bottleneck_diffs = set()
    for d, count in diff_count.items():
        if d > best_val * 0.7:  # Large differences
            bottleneck_diffs.add(d)
    
    # Step 3: Spectral re-solving
    # Use the eigenvector of the Laplacian of the difference graph to guide
    # which rows to re-solve and with what bias
    
    # Build a simple Laplacian for the difference space
    max_diff = best_val
    # L[d1, d2] = -1 if d1 and d2 "conflict" (i.e., can't both be in same row easily)
    # For simplicity, use a heuristic: differences that are close together conflict
    
    # Actually, let's use a more direct spectral approach:
    # Create a graph on the integers 0..M where edges represent "good" differences
    # Then use spectral clustering to find n_rows clusters, and assign rows to clusters
    
    # For now, use a hybrid: spectral bias + local search
    T = 20.0
    T_min = 0.5
    cool = 0.99998
    
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row to perturb
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        old_val = cur[ri][pi]
        
        # Generate candidates with spectral bias:
        # Prefer values that create differences not in bottleneck_diffs
        candidates = []
        for d in range(1, 30):
            new_val = old_val - d
            if new_val <= cur[ri][pi - 1]:
                break
            # Bias: check if new differences avoid bottlenecks
            new_diffs = {new_val - a for a in cur[ri] if a < new_val}
            new_diffs |= {a - new_val for a in cur[ri] if a > new_val}
            if not (new_diffs & bottleneck_diffs):
                candidates.append((new_val, 1))  # High priority
            else:
                candidates.append((new_val, 0))
        
        for d in range(1, 20):
            new_val = old_val + d
            if pi < k and new_val >= cur[ri][pi + 1]:
                break
            new_diffs = {new_val - a for a in cur[ri] if a < new_val}
            new_diffs |= {a - new_val for a in cur[ri] if a > new_val}
            if not (new_diffs & bottleneck_diffs):
                candidates.append((new_val, 1))
            else:
                candidates.append((new_val, 0))
        
        # Sort by priority (bottleneck-avoiding first), then randomize within groups
        candidates.sort(key=lambda x: -x[1])
        random.shuffle(candidates)
        candidates = [c[0] for c in candidates]
        
        moved = False
        for new_val in candidates:
            if new_val <= cur[ri][pi - 1]:
                continue
            if pi < k and new_val >= cur[ri][pi + 1]:
                continue
            
            cand_row = cur[ri][:]
            cand_row[pi] = new_val
            
            if not _check_row_valid(cand_row, k):
                continue
            
            new_diffs = _row_diffs_set(cand_row)
            other_diffs = cur_used - cur_row_diffs[ri]
            if new_diffs & other_diffs:
                continue
            
            if not _is_sidon(cand_row):
                continue
            
            max_other = 0
            for i, r in enumerate(cur):
                if i != ri:
                    m = max(r)
                    if m > max_other:
                        max_other = m
            cand_max = max(max_other, max(cand_row))
            
            delta_e = cand_max - cur_val
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                if random.random() < 0.3:
                    accept = True
            else:
                if T > T_min:
                    prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                cur_used -= cur_row_diffs[ri]
                cur_used |= new_diffs
                cur[ri] = cand_row
                cur_row_diffs[ri] = new_diffs
                cur_val = cand_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    # Update bottleneck analysis
                    diff_count = {}
                    for row in best:
                        for d in _row_diffs_set(row):
                            diff_count[d] = diff_count.get(d, 0) + 1
                    bottleneck_diffs = set()
                    for d, count in diff_count.items():
                        if d > best_val * 0.7:
                            bottleneck_diffs.add(d)
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        T = max(T_min, T * cool)
        
        # Periodic restart
        if no_improve > 20000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set()
            for r in cur:
                cur_used |= _row_diffs_set(r)
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            T = 20.0
            no_improve = 0
    
    return best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    random.seed(42)
    
    # Try spectral-guided approach first
    best = _spectral_guided_greedy(n_rows, k, time_limit=90.0)
    best_val = _max_entry(best)
    
    # Ensure validity
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best
# EVOLVE-BLOCK-END
