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
    max_restarts = 5000 # Increase restarts as each step is slightly more expensive
    
    # Threshold for "small" differences
    # Small differences are critical because there are only 1..T of them.
    # If we use them up, we are forced to use larger gaps.
    small_threshold = 15 

    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        used_diffs = set()
        rows = []
        
        success = True
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last_val = row[-1]
                valid_candidates = []
                
                # We need to find x > last_val
                # The maximum possible value is bounded by the current best_scope
                upper_bound = best_scope - 1 if best_scope < float('inf') else 200
                
                for x in range(last_val + 1, upper_bound + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        # Calculate heuristic score: number of small diffs introduced
                        small_count = sum(1 for d in new_diffs if d < small_threshold)
                        valid_candidates.append((small_count, x, new_diffs))
                        # We can stop early if we have a good candidate?
                        # Since we are sorting by small_count, we might want to see a few more
                        # to ensure we have the "best" small_count.
                        # But x is increasing, so new_diffs change.
                        # Let's collect up to a pool size.
                        if len(valid_candidates) >= 50:
                            break
                
                if not valid_candidates:
                    success = False
                    break
                
                # Sort candidates: Primary key: small_count (ascending), Secondary key: x (ascending)
                # This prioritizes candidates that "waste" fewer small differences.
                valid_candidates.sort(key=lambda item: (item[0], item[1]))
                
                # Selection Strategy:
                # To balance exploitation (best heuristic) and exploration,
                # we pick from the top few candidates.
                # The top candidate has the minimum small_count.
                # We pick randomly from the top 5 candidates that have the minimum small_count?
                # Or just pick the best one with some probability, and random otherwise.
                
                # Let's pick the best one (index 0) with 70% probability,
                # and a random one from the top 10 with 30% probability.
                if random.random() < 0.7:
                    chosen = valid_candidates[0]
                else:
                    # Pick from top 10
                    top_k = min(10, len(valid_candidates))
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
