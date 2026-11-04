"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within each row are distinct across all rows.
    Minimizes the largest entry (scope).
    
    Strategy: Use a randomized greedy approach with backtracking-like restarts.
    We try multiple random seeds/orderings and keep the best result found within time.
    For each row, we greedily pick the smallest possible next mark that doesn't conflict
    with already-used differences. We also try different row orderings and initial perturbations.
    """
    import time
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer
    
    best_scope = float('inf')
    best_rows = None
    
    # Precompute: total number of differences needed = n_rows * k*(k+1)//2
    # The simple counting bound: if we have m+1 marks (0..m), max differences is m*(m+1)//2
    # We need n_rows * k*(k+1)//2 <= m*(m+1)//2 => m >= sqrt(2 * n_rows * k*(k+1))
    # Actually, differences are from 1 to m, so we need m >= n_rows * k*(k+1)//2 ... no.
    # Each row has C(k+1, 2) = k*(k+1)//2 differences.
    # Total differences = n_rows * k*(k+1)//2.
    # These must all be distinct values in [1, m] where m is the max entry.
    # So m >= n_rows * k*(k+1)//2.
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Lower bound on scope
    lb = total_diffs  # since differences are in 1..m, need m >= total_diffs
    
    # Try multiple random attempts
    max_attempts = 200
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(attempt * 1000003 + 7)
        
        used = set()
        rows = []
        success = True
        
        for r in range(n_rows):
            row = [0]
            # For each new element, try candidates in a slightly randomized order
            # but biased towards small values
            while len(row) < k + 1:
                # Try candidates from small to large, but with some randomization
                # to escape local optima
                found = False
                # Number of candidates to try before giving up on this row
                # Start with a small window, expand if needed
                base_x = row[-1] + 1
                max_candidate = lb + 50 + attempt  # upper bound for candidate search
                
                # Try a shuffled list of candidates but prioritize small ones
                # Use a greedy approach with random tie-breaking
                candidates = []
                x = base_x
                while x <= max_candidate and len(candidates) < 100:
                    candidates.append(x)
                    x += 1
                
                # Shuffle candidates but keep small ones more likely
                # Actually, let's just try from small to large first, but if we've tried
                # several attempts, add some randomness
                if attempt < 10:
                    # Pure greedy for first few attempts
                    for cand in candidates:
                        new_diffs = {cand - a for a in row}
                        if not new_diffs & used:
                            row.append(cand)
                            used |= new_diffs
                            found = True
                            break
                else:
                    # Randomized greedy: try candidates in random order but biased to small
                    # Create a weighted random selection
                    cand_list = candidates[:]
                    random.shuffle(cand_list)
                    # Sort by a combination of value and randomness
                    scored = [(c, random.random() * 10) for c in cand_list]
                    scored.sort(key=lambda t: t[0] + t[1])
                    for cand, _ in scored:
                        new_diffs = {cand - a for a in row}
                        if not new_diffs & used:
                            row.append(cand)
                            used |= new_diffs
                            found = True
                            break
                
                if not found:
                    success = False
                    break
            
            if not success:
                break
            rows.append(row)
        
        if success and rows:
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [row[:] for row in rows]
                # If we hit the lower bound, we can stop early
                if best_scope == lb:
                    break
    
    if best_rows is None:
        # Fallback: pure greedy
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
        best_rows = rows
    
    return best_rows
# EVOLVE-BLOCK-END
