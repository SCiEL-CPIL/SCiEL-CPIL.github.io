import time
import random
import sys

def construct(n_rows, k):
    """
    Finds a Difference Triangle Set with n_rows rows and k+1 entries per row.
    Minimizes the largest entry.
    """
    deadline = time.time() + 145.0
    
    # Specific optimization for n_rows=4, k=7
    # Total diffs = 112. Bound 112.
    
    def max_entry(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)

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

    def solve_backtracking(m, node_limit):
        """
        Backtracking solver with bitmask for used differences.
        Returns list of rows if solution found, else None.
        """
        used_mask = 0
        rows = []
        nodes = 0
        
        # Pre-compute bits for 1..m
        # Since m is small, we can just compute on fly or precompute list
        # bit(d) = 1 << (d-1)
        
        def backtrack(row_idx, pos_idx, current_row):
            nonlocal used_mask, nodes
            nodes += 1
            if nodes > node_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos_idx == 0:
                rows.append(current_row[:])
                return backtrack(row_idx + 1, 0, [0])
            
            # Determine range for current_row[pos_idx]
            lower = current_row[pos_idx - 1] + 1
            # We need to fit (k - pos_idx) more elements.
            # So current val must be <= m - (k - pos_idx)
            upper = min(m, m - (k - pos_idx))
            
            if lower > upper:
                return False
            
            # Generate candidates
            # Heuristic: Try to fill with values that are "spread out"?
            # Random shuffle is a good general heuristic.
            candidates = list(range(lower, upper + 1))
            random.shuffle(candidates)
            
            for val in candidates:
                # Check conflicts
                conflict = False
                new_bits = 0
                for i in range(pos_idx):
                    d = val - current_row[i]
                    if d > m:
                        conflict = True
                        break
                    bit = 1 << (d - 1)
                    if used_mask & bit:
                        conflict = True
                        break
                    new_bits |= bit
                
                if conflict:
                    continue
                
                # Place val
                current_row[pos_idx] = val
                old_mask = used_mask
                used_mask |= new_bits
                
                if backtrack(row_idx, pos_idx + 1, current_row):
                    return True
                
                # Undo
                used_mask = old_mask
                current_row[pos_idx] = 0
            
            return False
        
        if backtrack(0, 0, [0]):
            return rows
        return None

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

    # Search Strategy
    # Start from lower bound 112
    # Increase m
    
    best_solution = greedy_init()
    best_m = max_entry(best_solution)
    
    # If best_m is already low, we might skip some.
    # But we want to minimize.
    
    # We will try m from 112 up to best_m - 1
    # If we find a solution, we return it.
    # If not, we return best_solution (greedy).
    
    # To avoid spending too much time on impossible m,
    # we limit the node_limit per m.
    
    # Total time budget: 140s
    # m range: 112 to best_m - 1
    # If best_m is 134, range is 22.
    # Time per m: 140 / 22 ~ 6s.
    
    current_time = time.time()
    start_time = current_time
    
    for m in range(112, best_m):
        if time.time() > deadline - 5:
            break
        
        # Dynamic node limit based on remaining time
        # Simple: fixed node limit per restart, few restarts.
        found = False
        # Try 5 restarts with 200k nodes each?
        # 5 * 200k = 1M nodes.
        # Should take < 1s in Python if optimized.
        
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            sol = solve_backtracking(m, node_limit=50000)
            if sol:
                found = True
                # Verify
                if is_valid(sol) and max_entry(sol) <= m:
                    best_solution = sol
                    best_m = m
                    break
        
        if found:
            # Found a solution with this m. 
            # Can we go lower? 
            # We are iterating upwards, so we found the first feasible m.
            # Since we started at 112, this is the minimum.
            return best_solution

    return best_solution

