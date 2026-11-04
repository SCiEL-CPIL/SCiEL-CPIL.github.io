import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145  # 145s deadline (5s margin from 150s limit)
    
    # Each row has k+1 integers starting at 0, strictly increasing.
    # We need n_rows rows, all positive differences distinct.
    
    # Strategy: randomized backtracking with restarts.
    # For small n_rows and k, we can do exhaustive-ish search.
    # Key insight: the total number of differences per row is k*(k+1)//2
    # Total differences across all rows: n_rows * k*(k+1)//2
    # All must be distinct positive integers.
    # The minimum possible max difference is bounded by: we need at least
    # n_rows * k*(k+1)//2 distinct positive differences, each <= max_entry.
    # So max_entry >= n_rows * k*(k+1)//2 (since differences are 1,2,...,max_entry at most)
    # Actually differences can be any subset of {1,...,max_entry}, so we need
    # max_entry >= n_rows * k*(k+1)//2.
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope
    lb = total_diffs
    
    # Try to find a solution with scope close to lb
    # Use randomized backtracking with iterative deepening on scope
    
    best_rows = None
    best_scope = float('inf')
    
    # First, get a quick greedy solution as a starting point
    best_rows = _greedy(n_rows, k)
    if best_rows is not None:
        best_scope = max(max(r) for r in best_rows)
    
    # Now try randomized backtracking to find smaller scope
    rng = random.Random(42)
    
    # Iterative deepening: try scopes from lb upward
    # But with time limit, we should focus on promising ranges
    
    # For the known case n_rows=4, k=7: lb = 4*7*8//2 = 112, best known = 134
    # So we might find 112-134 range solutions
    
    # Let's do randomized backtracking with various scope targets
    for scope_target in range(lb, min(lb + 50, best_scope + 1)):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts for this scope
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            result = _backtrack_randomized(n_rows, k, scope_target, rng, deadline - 10)
            if result is not None:
                s = max(max(r) for r in result)
                if s < best_scope:
                    best_scope = s
                    best_rows = result
                # If we hit the lower bound, we can't do better
                if best_scope == lb:
                    return best_rows
    
    # If time remains, try more aggressive search around best_scope - 1
    if time.time() < deadline - 15:
        for scope_target in range(max(lb, best_scope - 5), best_scope):
            if time.time() > deadline - 5:
                break
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                result = _backtrack_randomized(n_rows, k, scope_target, rng, deadline - 10)
                if result is not None:
                    s = max(max(r) for r in result)
                    if s < best_scope:
                        best_scope = s
                        best_rows = result
                        if best_scope == lb:
                            return best_rows
    
    return best_rows


def _greedy(n_rows, k):
    """Original greedy: each row starts at 0, takes smallest next mark."""
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


def _backtrack_randomized(n_rows, k, max_scope, rng, deadline):
    """Randomized backtracking to build n_rows rows with all differences distinct,
    each row's max entry <= max_scope."""
    used = set()
    rows = []
    
    def backtrack(row_idx, pos_in_row, current_row):
        nonlocal used, rows
        
        if time.time() > deadline:
            return None
        
        if row_idx == n_rows:
            return rows[:]
        
        if pos_in_row == k + 1:
            # Row complete
            rows.append(current_row[:])
            result = backtrack(row_idx + 1, 0, [0])
            if result is not None:
                return result
            rows.pop()
            return None
        
        if pos_in_row == 0:
            # Start new row with 0
            return backtrack(row_idx, 1, [0])
        
        # Try to place next element
        # Current last element
        last = current_row[-1]
        
        # Possible values for next element: last+1 to max_scope
        # We need all differences (x - a for a in current_row) to be in used or new
        # Actually: new differences must NOT be in used
        
        # Get candidate values, shuffled for randomization
        candidates = list(range(last + 1, max_scope + 1))
        
        # Heuristic: try values that give differences not in used
        # Filter: for candidate x, new_diffs = {x - a for a in current_row}
        # Must have new_diffs & used == set()
        
        valid_candidates = []
        for x in candidates:
            new_diffs = {x - a for a in current_row}
            if not new_diffs & used:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Shuffle for randomization, but bias toward smaller values (closer to optimal)
        rng.shuffle(valid_candidates)
        # Sort partially: put smaller ones first with some randomness
        # Actually just shuffle - for randomization
        # But bias: prefer values that leave more room
        
        for x in valid_candidates:
            if time.time() > deadline:
                return None
            
            new_diffs = {x - a for a in current_row}
            
            # Save state
            used_before = set(used)
            
            current_row.append(x)
            used |= new_diffs
            
            result = backtrack(row_idx, pos_in_row + 1, current_row)
            
            if result is not None:
                return result
            
            # Undo
            used = used_before
            current_row.pop()
        
        return None
    
    return backtrack(0, 0, [0])


# EVOLVE-BLOCK-END
