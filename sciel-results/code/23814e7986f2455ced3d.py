"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    """
    # For the specific task: n_rows=7, k=5 (6 entries per row)
    # But we write a general solution that works for any n_rows, k.
    
    # The counting bound for k=5, n_rows=7 is 105.
    # Best known is 112. We aim to find something close to or better than 112.
    
    # Strategy: 
    # 1. Pre-compute all valid "Sidon-like" rows (sets of k+1 integers starting at 0)
    #    up to some maximum value M.
    # 2. Each row is characterized by the set of its positive differences.
    # 3. We need to select n_rows such rows such that their difference sets are disjoint.
    # 4. Minimize the maximum entry across all selected rows.
    
    # For efficiency, we'll use a randomized greedy approach with backtracking.
    # We'll try to build the set of rows incrementally, always picking the row
    # with the smallest maximum entry that doesn't conflict with already chosen rows.
    
    # First, let's determine a reasonable upper bound for the maximum entry.
    # The counting bound is 105, best known is 112. Let's try up to 120.
    
    # Generate all candidate rows up to MAX_VAL
    MAX_VAL = 120
    
    # A row is a tuple of k+1 integers starting at 0, strictly increasing.
    # We represent a row by its tuple of values.
    # The differences of a row are all positive differences a_j - a_i for i < j.
    
    # Pre-compute all candidate rows and their difference sets.
    # For k=5, each row has 6 entries, so C(6,2)=15 differences.
    
    from itertools import combinations
    
    def get_diffs(row):
        """Get the set of all positive differences in the row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Generate all possible rows of k+1 integers starting at 0, strictly increasing,
    # with maximum entry <= MAX_VAL.
    # This is equivalent to choosing k positions from 1 to MAX_VAL.
    
    # For k=5, MAX_VAL=120, the number of combinations is C(120,5) ~ 191M which is too many.
    # We need a smarter approach.
    
    # Alternative approach: Use a randomized local search / simulated annealing.
    # Start with a valid configuration (e.g., the greedy one) and try to improve it.
    
    # Let's first generate the greedy solution as a starting point.
    def greedy_construct(n_rows, k):
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
    
    # Get the greedy solution
    rows = greedy_construct(n_rows, k)
    current_scope = max(max(row) for row in rows)
    
    # Now try to improve using randomized local search.
    # We'll represent the solution as a list of rows, each a tuple.
    # We'll try to swap or modify rows to reduce the scope.
    
    # For efficiency, let's try a different approach:
    # Use a constraint satisfaction / backtracking approach with randomization.
    
    # Actually, let's try a more direct approach:
    # We'll use a randomized greedy algorithm that tries multiple random orderings
    # of candidate rows and picks the first valid set.
    
    # Generate candidate rows more efficiently:
    # For each possible maximum value m from 1 to MAX_VAL, generate all rows
    # with maximum entry exactly m, sort them, and try to pack them.
    
    # Let's use a randomized backtracking search.
    
    def solve_with_backtracking(n_rows, k, max_val):
        """
        Try to find n_rows non-conflicting rows with all entries <= max_val.
        Returns the list of rows if found, else None.
        """
        # Generate all candidate rows with max entry <= max_val
        # For k=5, a row is determined by 5 positive integers 0 < a1 < a2 < ... < a5 <= max_val
        # where a5 is the maximum entry.
        
        # We'll generate candidates and try to pack them.
        
        # For efficiency, we'll use a randomized greedy approach:
        # 1. Generate a pool of candidate rows.
        # 2. Randomly shuffle them.
        # 3. Greedily select rows that don't conflict.
        
        # To make this work well, we need a good pool of candidates.
        # Let's generate candidates by sampling.
        
        candidates = []
        
        # Generate random rows
        num_candidates = 5000
        for _ in range(num_candidates):
            # Generate a random row of k+1 integers starting at 0
            # with maximum entry <= max_val
            if k == 0:
                row = (0,)
            else:
                # Choose k integers from 1 to max_val
                vals = sorted(random.sample(range(1, max_val + 1), k))
                row = (0,) + tuple(vals)
            candidates.append(row)
        
        # Also add some structured candidates
        # Greedy rows
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            candidates.append(tuple(row))
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for row in candidates:
            if row not in seen:
                seen.add(row)
                unique_candidates.append(row)
        candidates = unique_candidates
        
        # Pre-compute difference sets
        cand_diffs = []
        for row in candidates:
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
            cand_diffs.append(diffs)
        
        # Try multiple random orderings
        best_solution = None
        for attempt in range(200):
            # Shuffle candidates
            indices = list(range(len(candidates)))
            random.shuffle(indices)
            
            used = set()
            selected = []
            for idx in indices:
                if len(selected) == n_rows:
                    break
                if not (cand_diffs[idx] & used):
                    selected.append(candidates[idx])
                    used |= cand_diffs[idx]
            
            if len(selected) == n_rows:
                scope = max(max(row) for row in selected)
                if best_solution is None or scope < max(max(row) for row in best_solution):
                    best_solution = selected
        
        return best_solution
    
    # Try to find a solution with decreasing max_val
    # Start from the greedy scope and try to reduce it
    
    best_result = rows
    best_scope = current_scope
    
    # Try to improve by searching for solutions with smaller scope
    for target_scope in range(best_scope - 1, max(100, best_scope - 20), -1):
        result = solve_with_backtracking(n_rows, k, target_scope)
        if result is not None:
            scope = max(max(row) for row in result)
            if scope < best_scope:
                best_result = result
                best_scope = scope
            break
    
    # Convert to list of lists
    return [list(row) for row in best_result]

# EVOLVE-BLOCK-END
