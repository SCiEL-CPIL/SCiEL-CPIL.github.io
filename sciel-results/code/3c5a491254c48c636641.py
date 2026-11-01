import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin before 150s kill
    
    total_diffs = n_rows * k * (k + 1) // 2
    min_M = total_diffs
    
    # Known best: 112 for (7,5). Try from min_M up to 112+margin
    max_M_to_try = max(min_M, 112) + 10
    
    best_solution = None
    
    for M in range(min_M, max_M_to_try + 1):
        if time.time() > deadline:
            break
        
        solution = solve_mis(n_rows, k, M, deadline)
        if solution:
            return solution
    
    # Fallback: greedy construction (valid but potentially large)
    return greedy_construct(n_rows, k)

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

def solve_mis(n_rows, k, M, deadline):
    """
    Formulate as Maximum Independent Set on conflict graph.
    Nodes: all valid rows (subsets of {0..M} of size k+1 starting at 0).
    Edge between two nodes if they share any difference.
    Find independent set of size n_rows.
    """
    # Step 1: Enumerate all valid rows (subsets of size k+1 from {0..M} with 0 included)
    # A row is valid if all its internal differences are distinct (always true for a set)
    # and all differences are <= M (always true since max element <= M)
    
    # Precompute: for each possible row, compute its set of differences
    # Represent rows as tuples for hashing
    
    # Generate all valid rows
    all_rows = []
    all_row_diffs = []  # parallel list of frozensets of differences
    
    # We need k more elements from {1, ..., M}
    # For small M, this is manageable
    # C(M, k) rows total
    
    if M <= 130 and k <= 6:
        # Enumerate all C(M, k) combinations
        from itertools import combinations as comb
        for combo in comb(range(1, M + 1), k):
            row = (0,) + combo
            # Compute all pairwise differences
            diffs = set()
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d in diffs:
                        # Duplicate difference within row - invalid (but shouldn't happen for strictly increasing)
                        break
                    diffs.add(d)
            else:
                all_rows.append(row)
                all_row_diffs.append(frozenset(diffs))
    else:
        # Too many rows, use a different approach
        return None
    
    n_nodes = len(all_rows)
    if n_nodes == 0:
        return None
    
    # Step 2: Build conflict graph using bitsets
    # Two rows conflict if they share any difference
    # Use Python's int as bitset for efficiency
    
    # First, map each difference to a bit position
    # Differences range from 1 to M
    # So we can use an integer with M bits
    
    # For each row, create a bitmask of its differences
    row_bitmasks = []
    for diffs in all_row_diffs:
        mask = 0
        for d in diffs:
            mask |= (1 << d)
        row_bitmasks.append(mask)
    
    # Build adjacency: for each node, bitmask of conflicting nodes
    # This is expensive: O(n_nodes^2) but each check is O(1) with bit operations
    # n_nodes can be up to C(130, 5) = ~1.8 billion - too many!
    
    # We need a smarter approach. Instead of building the full graph,
    # use a backtracking search that checks conflicts on the fly.
    
    # Alternative: use a constraint-based backtracking with memoization
    # State: (row_index, frozenset of used differences)
    # But the state space is huge.
    
    # Better approach: randomized greedy with backtracking
    # Pick rows one at a time, ensuring no conflict with previously picked rows.
    
    # Let's use a different strategy: 
    # 1. Randomly shuffle all rows
    # 2. Greedily pick non-conflicting rows
    # 3. If we get n_rows rows, we have a solution
    # 4. Repeat with different shuffles
    
    # But this might not find solutions for tight M.
    
    # Let's use a hybrid: 
    # - Generate a subset of candidate rows (e.g., those with small max element)
    # - Use backtracking on this subset
    
    # Filter rows by max element to reduce search space
    # Prioritize rows with smaller max elements
    max_elements = [row[-1] for row in all_rows]
    
    # Sort rows by max element
    sorted_indices = sorted(range(n_nodes), key=lambda i: max_elements[i])
    
    # Take the first N rows as candidates (those with smallest max)
    # This biases toward compact solutions
    n_candidates = min(n_nodes, 50000)  # Cap for performance
    candidate_indices = sorted_indices[:n_candidates]
    
    candidate_rows = [all_rows[i] for i in candidate_indices]
    candidate_diffs = [all_row_diffs[i] for i in candidate_indices]
    candidate_masks = [row_bitmasks[i] for i in candidate_indices]
    
    n_cand = len(candidate_rows)
    
    # Backtracking search
    # State: (pos, used_mask) where used_mask is bitmask of all differences used so far
    # But used_mask can be up to M bits, which is fine for M <= 130
    
    # Use iterative deepening or just try to find any valid set of size n_rows
    
    # Randomized restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomize the order of candidates
        indices = list(range(n_cand))
        random.shuffle(indices)
        
        # Backtracking
        chosen = []
        used_mask = 0
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if len(chosen) == n_rows:
                return True
            if pos >= n_cand:
                return False
            
            for i in range(pos, n_cand):
                idx = indices[i]
                # Check if this row conflicts with any chosen row
                if candidate_masks[idx] & used_mask:
                    continue
                
                # Try this row
                chosen.append(candidate_rows[idx])
                new_used_mask = used_mask | candidate_masks[idx]
                
                if backtrack(i + 1):
                    return True
                
                chosen.pop()
            
            return False
        
        if backtrack(0):
            return [list(row) for row in chosen]
    
    return None
