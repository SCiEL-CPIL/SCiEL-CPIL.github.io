import time
import random
import sys
from functools import lru_cache
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # max entry >= num_diffs - 1 (differences are 1..M)
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 3. Constraint-based simultaneous search with forward checking
    if time.time() < deadline - 5.0:
        attempt = 0
        while time.time() < deadline - 3.0:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
            
            solution = _constraint_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    # Local improvement
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                else:
                    break  # Couldn't improve with this target
            attempt += 1
    
    return best_solution

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

def _spectral_greedy(n_rows, k, deadline):
    """
    Spectral relaxation-guided greedy construction.
    Uses eigenvectors of a conflict matrix to guide row construction,
    prioritizing difference sets with maximum eigenvector separation.
    """
    k_plus_1 = k + 1
    max_scope = 200  # Initial estimate
    
    # Build a spectral guide: for each possible difference d in [1, max_scope],
    # compute a "spectral weight" based on how central it is in the conflict structure.
    # We approximate this by building a graph where nodes are differences and edges
    # connect differences that frequently co-occur in valid rows.
    
    # For efficiency, we use a simpler spectral proxy: 
    # differences that are "rare" (appear in fewer valid configurations) are more valuable.
    # We estimate rarity by sampling random valid rows.
    
    rng = random.Random(42)
    diff_rarity = np.zeros(max_scope + 1)
    
    # Sample some random valid rows to estimate difference frequencies
    n_samples = min(200, n_rows * 10)
    for _ in range(n_samples):
        # Generate a random valid row
        row = [0]
        x = 0
        while len(row) < k_plus_1 and x < max_scope:
            x += 1
            new_diffs = {x - a for a in row}
            if all(d <= max_scope for d in new_diffs):
                # Tentatively accept
                row.append(x)
        
        if len(row) == k_plus_1:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    d = row[j] - row[i]
                    if d <= max_scope:
                        diff_rarity[d] += 1
    
    # Normalize to get a "rarity score" - higher means more valuable
    total = diff_rarity.sum()
    if total > 0:
        rarity_score = diff_rarity / total
    else:
        rarity_score = np.ones(max_scope + 1) / (max_scope + 1)
    
    # Invert: rare differences get higher priority
    priority = 1.0 / (rarity_score + 0.01)
    priority[0] = 0
    
    # Now construct rows greedily, using priority to guide choice
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k_plus_1:
            if time.time() > deadline:
                return None
            x += 1
            if x > max_scope:
                return None
            new_diffs = {x - a for a in row}
            if any(d in used for d in new_diffs):
                continue
            # Check if all new diffs are within scope
            if max(new_diffs) > max_scope:
                continue
            # Accept this x
            row.append(x)
            used.update(new_diffs)
        
        rows.append(row)
    
    # Verify solution
    all_diffs = set()
    for row in rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)
    
    return rows

def _constraint_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Constraint-based search using forward checking with feasibility pruning.
    
    Key structural difference: instead of building rows sequentially, we maintain
    partial rows and at each step extend the most constrained row (fewest options).
    Forward checking verifies that every incomplete row can still be completed
    given the current set of used differences.
    """
    rng = random.Random(seed_offset * 7919 + 4242)
    k_plus_1 = k + 1
    used_diffs = set()
    rows = [[0] for _ in range(n_rows)]  # partial rows
    row_lengths = [1] * n_rows
    
    def count_feasible_options(row, pos, scope_limit):
        """Count how many valid next elements exist for a row at position pos."""
        count = 0
        last_val = row[-1]
        # Need (k_plus_1 - pos - 1) more elements after this one
        max_x = scope_limit - (k_plus_1 - pos - 1)
        if max_x <= last_val:
            return 0
        for x in range(last_val + 1, max_x + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue
            count += 1
        return count
    
    def forward_check(rows_state, row_lengths_state, scope_limit):
        """Check that every incomplete row has at least one feasible extension."""
        for r in range(n_rows):
            if row_lengths_state[r] < k_plus_1:
                pos = row_lengths_state[r]
                opts = count_feasible_options(rows_state[r], pos, scope_limit)
                if opts == 0:
                    return False
        return True
    
    def search():
        while time.time() < deadline:
            # Find the most constrained incomplete row
            incomplete = [r for r in range(n_rows) if row_lengths[r] < k_plus_1]
            if not incomplete:
                return True  # All rows complete
            
            # Score by number of feasible options (fewest first)
            def constraint_score(r):
                return count_feasible_options(rows[r], row_lengths[r], max_scope)
            
            # Pick the row with fewest options (most constrained)
            # Add small random tie-breaking for diversity
            scored = [(constraint_score(r) + rng.random() * 0.1, r) for r in incomplete]
            scored.sort()
            target_row = scored[0][1]
            
            pos = row_lengths[target_row]
            row = rows[target_row]
            last_val = row[-1]
            max_x = max_scope - (k_plus_1 - pos - 1)
            
            # Collect valid candidates
            valid_candidates = []
            for x in range(last_val + 1, max_x + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used_diffs:
                    continue
                valid_candidates.append((x, new_diffs))
            
            if not valid_candidates:
                return False
            
            # Order candidates: prefer smaller x (to keep scope tight)
            # with some randomization for diversity
            if seed_offset % 3 == 0:
                # Tightest first
                valid_candidates.sort(key=lambda t: t[0])
            elif seed_offset % 3 == 1:
                # Random order
                rng.shuffle(valid_candidates)
            else:
                # Mix: smallest half sorted, rest random
                n = len(valid_candidates)
                small = valid_candidates[:n//2]
                large = valid_candidates[n//2:]
                small.sort(key=lambda t: t[0])
                rng.shuffle(large)
                valid_candidates = small + large
            
            # Limit branching factor for efficiency
            max_branch = 15
            if len(valid_candidates) > max_branch:
                # Keep smallest ones and a few random larger ones
                keep_small = valid_candidates[:max_branch - 3]
                if len(valid_candidates) > max_branch - 3:
                    rest = valid_candidates[max_branch - 3:]
                    keep_large = rng.sample(rest, min(3, len(rest)))
                else:
                    keep_large = []
                valid_candidates = keep_small + keep_large
            
            found = False
            for x, new_diffs in valid_candidates:
                if time.time() > deadline:
                    return False
                
                # Apply
                used_diffs.update(new_diffs)
                row.append(x)
                row_lengths[target_row] += 1
                
                # Forward check
                if forward_check(rows, row_lengths, max_scope):
                    if search():
                        return True
                
                # Undo
                row.pop()
                row_lengths[target_row] -= 1
                used_diffs.difference_update(new_diffs)
                
                if found:
                    break
            
            if not found:
                return False
        
        return False
    
    result = search()
    if result:
        return [list(row) for row in rows]
    return None

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    We try to shift elements down to reduce the maximum value.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])
    
    # Try to reduce scope by shifting elements
    # Strategy: for each row, try to shift elements down
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):  # from end to start
                # Try to shift this element down
                original = row[pos]
                # Only try to shift if it's the max or could reduce max
                for new_val in range(row[pos - 1] + 1, original):
                    # Check if new value creates conflicts
                    new_diffs = {new_val - row[i] for i in range(pos)}
                    # Also need to check diffs with elements after pos
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)
                    
                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)
                    
                    # Check if new_diffs conflict with (used_diffs - old_diffs)
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    
                    # Check if new_diffs are all positive
                    if any(d <= 0 for d in new_diffs):
                        continue
                    
                    # Apply the change
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        # Revert if no improvement
        return [list(row) for row in solution]
