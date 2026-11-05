import random
import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    best_result = None
    best_scope = float('inf')
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer for the evaluator
    
    # Number of restarts
    max_restarts = 1000 # Reduced because each restart is more expensive
    
    def count_next_options(row, used_diffs, current_val, limit):
        """
        Counts how many valid next values exist after current_val.
        This is a lookahead heuristic.
        """
        count = 0
        # We only need to check a small window ahead to estimate constraint tightness
        # If we can find many options, the constraint is loose.
        # If we find few, it's tight.
        # We cap the search to keep it fast.
        for y in range(current_val + 1, limit + 1):
            new_diffs = {y - a for a in row}
            if not new_diffs & used_diffs:
                count += 1
                if count >= 10: # Sufficient to distinguish "loose" from "tight"
                    return count
        return count

    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        used_diffs = set()
        rows = []
        
        success = True
        for _ in range(n_rows):
            row = [0]
            # We need to build the rest of the row.
            # We use a greedy approach with lookahead for the next step.
            
            current_max = 0
            for step in range(k):
                last_val = row[-1]
                valid_candidates = []
                
                # Upper bound for x
                # We want to keep x small to minimize scope, but not too small to cause dead-ends.
                # Let's search up to a reasonable limit.
                # If we have a best_scope, we can't exceed it.
                upper_bound = best_scope - 1 if best_scope < float('inf') else 150
                
                # To make the heuristic faster, we might limit the range of x we consider
                # or just iterate.
                for x in range(last_val + 1, upper_bound + 1):
                    new_diffs = {x - a for a in row}
                    if new_diffs & used_diffs:
                        continue
                    
                    # Heuristic: Count next options
                    # We pass a limited range for the next step to keep it fast
                    # The next value y must be > x.
                    # We check how many y are valid given used_diffs U new_diffs.
                    # Note: used_diffs is NOT updated yet for the heuristic, 
                    # but we should consider that new_diffs will be added.
                    # So we check against used_diffs U new_diffs.
                    temp_used = used_diffs | new_diffs
                    # We estimate options for y in range (x+1, x+10) or so?
                    # Actually, let's just count how many small y are valid.
                    # A simpler heuristic: 
                    # Score = number of small differences in new_diffs.
                    # The directive suggests "minimizing the degree of future constraint violations".
                    # Let's stick to counting valid next steps within a small window.
                    
                    next_count = 0
                    # Check next 5 possible values
                    for y in range(x + 1, x + 6):
                        next_diffs = {y - a for a in row}
                        if not (next_diffs & temp_used):
                            next_count += 1
                    
                    # Prefer candidates with MORE next options (less constrained)
                    # And smaller x (to minimize scope)
                    # We sort by (-next_count, x)
                    valid_candidates.append((-next_count, x, new_diffs))
                    
                    # Early break if we find a very good candidate?
                    # If next_count is high, it's a good candidate.
                    if len(valid_candidates) >= 20:
                        break
                
                if not valid_candidates:
                    success = False
                    break
                
                # Sort by heuristic: most next options, then smallest x
                valid_candidates.sort(key=lambda item: (item[0], item[1]))
                
                # Pick the best one, with some randomness
                # If the best candidate has significantly more options, pick it.
                # Otherwise, random from top few.
                if random.random() < 0.8:
                    chosen = valid_candidates[0]
                else:
                    top_k = min(5, len(valid_candidates))
                    chosen = random.choice(valid_candidates[:top_k])
                
                x = chosen[1]
                new_diffs = chosen[2]
                row.append(x)
                used_diffs |= new_diffs
                
            if not success:
                break
            
            rows.append(row)
            current_scope = max(r[-1] for r in rows)
            
            if len(rows) == n_rows:
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_result = [r[:] for r in rows]
                    if best_scope <= 105:
                        break
        
    if best_result is None:
        # Fallback to deterministic greedy if random search failed (unlikely)
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
        
    return best_result
