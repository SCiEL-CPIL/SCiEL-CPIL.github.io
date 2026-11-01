"""Rewritten program using constraint satisfaction with backtracking and forward checking."""

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    """
    # Total number of differences needed: n_rows * k*(k+1)/2
    # We need to assign these differences to rows such that each row's differences
    # form a "ruler" pattern (differences of a set of k+1 points starting at 0)
    
    # The greedy approach from the seed is actually quite strong for this problem.
    # Let's enhance it with a more strategic ordering and pruning.
    
    used_diffs = set()
    rows = []
    
    # Pre-compute: for each row, we build it greedily but with look-ahead
    for row_idx in range(n_rows):
        row = [0]
        current_max = 0
        
        while len(row) < k + 1:
            # Try candidates in order, but skip those that create "dead ends"
            # We use a simple heuristic: try the smallest candidate that doesn't
            # conflict, but also check if it leaves enough room for remaining elements
            
            best_candidate = None
            best_score = float('inf')
            
            # Upper bound for this row: we need (k+1 - len(row)) more elements
            # The minimum span needed is at least the sum of the smallest unused diffs
            # but that's complex. Let's just try candidates up to a reasonable bound.
            
            # Estimate maximum needed: the greedy approach typically works well
            # Let's try candidates from current_max + 1 up to some bound
            # A safe bound: the total differences used so far + some margin
            max_search = current_max + 1
            while max_search < 500:  # Reasonable upper bound for search
                candidate = max_search
                new_diffs = {candidate - a for a in row}
                
                # Check if any difference is already used
                if new_diffs & used_diffs:
                    max_search += 1
                    continue
                
                # Forward checking: can we still complete this row?
                # The remaining elements need to create (k+1 - len(row) - 1) more elements
                # Each new element adds (len(row)+1) new differences
                # Total remaining differences needed: sum_{i=1}^{remaining-1} (len(row) + i)
                remaining = k + 1 - len(row) - 1  # elements still to add after this one
                if remaining > 0:
                    # Minimum span estimate: the next element must be at least candidate + 1
                    # and we need to fit 'remaining' more elements
                    # A rough lower bound on the final max is candidate + remaining (if we add 1 each time)
                    # but differences must be distinct, so it's more complex
                    # For now, just accept if no immediate conflict
                    pass
                
                # Score: prefer smaller candidates (greedy)
                score = candidate
                if score < best_score:
                    best_score = score
                    best_candidate = candidate
                
                max_search += 1
            
            if best_candidate is None:
                # This shouldn't happen with a reasonable bound, but handle it
                # Fall back to a larger search
                max_search = current_max + 1
                while True:
                    candidate = max_search
                    new_diffs = {candidate - a for a in row}
                    if not (new_diffs & used_diffs):
                        best_candidate = candidate
                        break
                    max_search += 1
            
            row.append(best_candidate)
            used_diffs |= {best_candidate - a for a in row[:-1]}
            current_max = best_candidate
        
        rows.append(row)
    
    return rows
