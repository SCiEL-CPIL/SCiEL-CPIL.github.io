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
    max_restarts = 5000

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
                upper_bound = best_scope - 1 if best_scope < float('inf') else 200
                
                for x in range(last_val + 1, upper_bound + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        # Heuristic: minimize small diffs + penalize large gaps
                        small_count = sum(1 for d in new_diffs if d <= 20)
                        gap = x - last_val
                        score = small_count * 2 + gap * 0.1
                        valid_candidates.append((score, x, new_diffs))
                        if len(valid_candidates) >= 40:
                            break
                
                if not valid_candidates:
                    success = False
                    break
                
                valid_candidates.sort(key=lambda item: item[0])
                
                # Pick best with some exploration
                if random.random() < 0.7:
                    chosen = valid_candidates[0]
                else:
                    top_k = min(8, len(valid_candidates))
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
