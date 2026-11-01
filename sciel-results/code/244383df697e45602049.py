"""
Strategy: Minimum-cost set packing over a bounded pool of perfect-difference rows.
We generate a large pool of candidate rows (all valid internally) with max entry <= M.
Then we solve a set-packing problem: choose n_rows rows such that their difference
sets are disjoint, minimizing the maximum entry across chosen rows.
We use branch-and-bound with MRV and forward checking.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145.0  # 5s safety margin from 150s limit
    
    # 1. Greedy baseline
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_score = _max_entry(baseline)
    
    if n_rows <= 1:
        return baseline
    
    # 2. Generate a pool of candidate rows with max entry <= M
    #    M is chosen so that the number of differences per row (k*(k+1)/2)
    #    times n_rows is less than M*(M+1)/2 (total possible diffs up to M)
    #    For (7,5): diffs per row = 15, total diffs needed = 105.
    #    M=112 gives 6328 possible diffs, plenty of room.
    #    We'll try M from a lower bound up to best_score - 1.
    
    # Lower bound: counting bound
    # n_rows * k*(k+1)/2 <= M*(M+1)/2  => M ~ sqrt(2*n_rows*k*(k+1)/2)
    # For (7,5): 7*15=105, M ~ sqrt(210) ~ 14.5... but that's for the max entry
    # Actually the counting bound says: total distinct diffs = n_rows * C(k+1,2)
    # These must all be <= M (since max diff in a row with max entry M is M)
    # So M*(M+1)/2 >= n_rows * C(k+1,2)
    # For (7,5): 105 <= M*(M+1)/2 => M >= 14.5... wait, that's not right.
    # The diffs can be up to M (max entry), so there are M possible diff values (1..M).
    # We need n_rows * C(k+1,2) distinct diffs, each in [1, M].
    # So M >= n_rows * C(k+1,2). For (7,5): M >= 105.
    # The simple counting bound is 105.
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs_needed  # M must be at least this
    
    # Try M from lower_bound up to best_score - 1
    # For efficiency, we'll generate rows up to max_M = best_score - 1
    max_M = best_score - 1
    
    if max_M < lower_bound:
        return baseline
    
    # Generate pool of candidate rows
    # A row is a tuple of k+1 increasing integers starting at 0, max <= M
    # with distinct internal differences.
    # We'll generate rows for each M from lower_bound to max_M, but that's expensive.
    # Instead, generate all valid rows with max entry <= max_M, then filter by M.
    
    # For efficiency, we'll generate rows incrementally.
    # For (7,5): k=5, so rows have 6 elements.
    # Number of possible rows with max <= 112 is large, but we can prune.
    
    # Alternative: generate a large random sample of valid rows, then solve packing.
    # This is more practical for the time budget.
    
    # Generate pool
    pool = _generate_row_pool(n_rows, k, max_M, deadline)
    
    if not pool:
        return baseline
    
    # Solve set packing: choose n_rows rows from pool with disjoint diff sets,
    # minimizing the max entry.
    # We'll use branch-and-bound.
    
    result = _solve_packing(pool, n_rows, k, lower_bound, deadline)
    
    if result is not None:
        score = _max_entry(result)
        if score < best_score:
            best_score = score
            best_rows = result
    
    return best_rows

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
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

def _generate_row_pool(n_rows, k, max_M, deadline):
    """
    Generate a pool of valid rows (internally distinct diffs) with max entry <= max_M.
    We use a mix of systematic and random generation.
    """
    pool = []
    seen = set()
    
    # Systematic: try all combinations? Too many for k=5, max_M=112.
    # Instead, use random generation with bias towards smaller entries.
    
    # First, add the greedy row as a starting point
    greedy_row = _greedy_construct(1, k)[0]
    if _max_entry(greedy_row) <= max_M:
        pool.append(tuple(greedy_row))
        seen.add(tuple(greedy_row))
    
    # Random generation
    attempts = 0
    max_attempts = 50000
    while len(pool) < 500 and attempts < max_attempts:
        if time.time() > deadline - 10:
            break
        attempts += 1
        row = _generate_random_row(k, max_M)
        if row is not None:
            t = tuple(row)
            if t not in seen:
                seen.add(t)
                pool.append(t)
    
    # Also add some structured rows: arithmetic-like sequences with perturbations
    for base in range(1, min(max_M // (k + 1) + 1, 30)):
        if time.time() > deadline - 10:
            break
        # Row: 0, base, 2*base, ..., k*base (arithmetic progression)
        # This has diffs: base, 2*base, ..., k*base, (k-1)*base, ..., base
        # Not all distinct! So APs are invalid for k >= 3.
        # Instead, use "Sidon-like" sequences.
        pass
    
    return pool

def _generate_random_row(k, max_M):
    """
    Generate a random valid row of k+1 increasing integers starting at 0,
    with max entry <= max_M and distinct internal differences.
    """
    for _ in range(20):
        row = [0]
        valid = True
        for _ in range(k):
            last = row[-1]
            # Pick a random next value
            min_next = last + 1
            max_next = max_M
            if min_next > max_next:
                valid = False
                break
            x = random.randint(min_next, max_next)
            # Check distinct internal differences
            new_diffs = {x - a for a in row}
            existing_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    existing_diffs.add(row[j] - row[i])
            if new_diffs & existing_diffs:
                valid = False
                break
            row.append(x)
        if valid and len(row) == k + 1 and row[-1] <= max_M:
            return row
    return None

def _solve_packing(pool, n_rows, k, lower_bound, deadline):
    """
    Solve set packing: choose n_rows rows from pool with disjoint diff sets,
    minimizing the max entry.
    Uses branch-and-bound with MRV.
    """
    # Precompute diff sets for each row
    row_diffs = []
    for row in pool:
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    # Sort pool by max entry (ascending) to try smaller rows first
    indices = list(range(len(pool)))
    indices.sort(key=lambda i: pool[i][-1])
    sorted_pool = [pool[i] for i in indices]
    sorted_diffs = [row_diffs[i] for i in indices]
    
    # Branch and bound
    best_solution = None
    best_score = float('inf')
    
    def backtrack(used_diffs, chosen, start_idx, current_max):
        nonlocal best_solution, best_score
        
        if time.time() > deadline - 5:
            return
        
        if len(chosen) == n_rows:
            score = current_max
            if score < best_score:
                best_score = score
                best_solution = [list(row) for row in chosen]
            return
        
        # Pruning: if current_max >= best_score, no improvement possible
        # (since we're trying to minimize max entry)
        if current_max >= best_score:
            return
        
        # MRV: choose the next row that has the fewest compatible options
        # For simplicity, we iterate in order but prune early.
        
        for i in range(start_idx, len(sorted_pool)):
            if time.time() > deadline - 5:
                return
            
            row = sorted_pool[i]
            diffs = sorted_diffs[i]
            
            # Check if diffs are disjoint with used_diffs
            if diffs & used_diffs:
                continue
            
            new_max = max(current_max, row[-1])
            if new_max >= best_score:
                break  # Since sorted by max entry, all subsequent rows have >= max
            
            backtrack(used_diffs | diffs, chosen + [row], i + 1, new_max)
            
            # If we found a solution with score = lower_bound, it's optimal
            if best_score == lower_bound:
                return
    
    # Start backtracking
    backtrack(set(), [], 0, 0)
    
    return best_solution

def _max_entry(rows):
    return max(max(row) for row in rows)
# EVOLVE-BLOCK-END
