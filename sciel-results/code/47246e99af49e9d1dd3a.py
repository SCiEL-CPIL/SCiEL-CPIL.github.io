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
    
    # Heuristic: Minimize sum of 1/d for new differences.
    # This prefers larger differences, preserving small differences for later rows.
    def heuristic_score(new_diffs):
        score = 0.0
        for d in new_diffs:
            score += 1.0 / d
        return score

    max_attempts = 1000 # Fewer attempts needed with backtracking
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        used_diffs = set()
        rows = []
        
        def dfs(row_idx, col_idx, current_row, current_used_diffs):
            if time.time() - start_time > time_limit:
                return False, None
            
            if row_idx == n_rows:
                scope = max(r[-1] for r in rows)
                return True, scope
            
            if col_idx == k:
                # Row complete
                rows.append(current_row)
                success, scope = dfs(row_idx + 1, 0, [0], current_used_diffs)
                if success:
                    return True, scope
                else:
                    rows.pop()
                    return False, None
            
            last_val = current_row[-1]
            upper_bound = best_scope - 1 if best_scope < float('inf') else 200
            
            valid_candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & current_used_diffs:
                    continue
                # Valid candidate
                valid_candidates.append((heuristic_score(new_diffs), x, new_diffs))
            
            if not valid_candidates:
                return False, None
            
            # Sort by heuristic score (ascending)
            valid_candidates.sort(key=lambda item: item[0])
            
            # Pick from top 5 with randomness to explore different structures
            top_k = min(5, len(valid_candidates))
            top_candidates = valid_candidates[:top_k]
            random.shuffle(top_candidates)
            
            for _, x, new_diffs in top_candidates:
                next_used_diffs = current_used_diffs | new_diffs
                next_row = current_row + [x]
                
                success, scope = dfs(row_idx, col_idx + 1, next_row, next_used_diffs)
                if success:
                    return True, scope
            
            return False, None
        
        success, scope = dfs(0, 0, [0], used_diffs)
        
        if success and scope < best_scope:
            best_scope = scope
            best_result = [r[:] for r in rows]
            if best_scope <= 105:
                break
        
    if best_result is None:
        # Fallback to deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last_val = row[-1]
                found = False
                for x in range(last_val + 1, 500):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    # Should not happen if scope is large enough
                    row.append(last_val + 1) 
            rows.append(row)
        return rows
        
    return best_result
