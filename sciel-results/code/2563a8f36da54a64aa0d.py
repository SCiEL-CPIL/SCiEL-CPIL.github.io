import random
import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry using constraint satisfaction with backtracking.
    """
    best_result = None
    best_scope = float('inf')
    
    start_time = time.time()
    time_limit = 105.0  # Leave buffer for evaluator
    
    # Precompute: for a row of length k+1, we need k(k+1)/2 distinct differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Upper bound on scope: counting bound is total_diffs_needed, but we need to be more careful
    # The largest difference in any row is at most scope, and there are scope possible differences
    # So we need scope >= total_diffs_needed, but also need to account for the structure
    # Start with a reasonable upper bound
    initial_upper = min(150, max(112, total_diffs_needed + 10))
    
    def count_future_slots(used_diffs, current_row, remaining_rows, k):
        """
        Estimate how many valid difference slots remain for future rows.
        A higher count means more flexibility.
        """
        # For each remaining row, we need to place k(k+1)/2 differences
        # The "degree" of constraint violation is how many small differences are already used
        # Small differences (1 to some threshold) are most constrained
        threshold = 20
        small_used = sum(1 for d in used_diffs if d <= threshold)
        total_small_available = threshold  # differences 1..threshold
        small_remaining = total_small_available - small_used
        
        # Each remaining row needs at least some small differences
        # A rough estimate: each row of size k+1 will use about k small differences
        needed_small = remaining_rows * k
        # If small_remaining < needed_small, we're in trouble
        return small_remaining - needed_small
    
    def backtrack(row_idx, used_diffs, rows, scope_limit):
        """
        Backtracking search to build rows.
        Returns True if a valid configuration is found, updating rows in place.
        """
        nonlocal best_scope, best_result
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # All rows built successfully
            current_scope = max(r[-1] for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_result = [r[:] for r in rows]
            return True
        
        row = [0]
        rows.append(row)
        
        def build_row(pos, last_val):
            """
            Recursively build the current row starting from position pos.
            Returns True if row can be completed.
            """
            if time.time() - start_time > time_limit:
                return False
            
            if pos == k:
                # Row complete
                rows[row_idx] = row[:]
                # Try next row
                return backtrack(row_idx + 1, used_diffs, rows, scope_limit)
            
            last_val = row[-1]
            candidates = []
            
            for x in range(last_val + 1, scope_limit + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used_diffs:
                    continue
                
                # Heuristic: count how many small differences this choice uses
                # and estimate remaining flexibility
                small_count = sum(1 for d in new_diffs if d <= 20)
                
                # Estimate future constraint: after using these diffs, how many
                # small slots remain for subsequent rows?
                temp_used = used_diffs | new_diffs
                future_flex = count_future_slots(temp_used, row, n_rows - row_idx - 1, k)
                
                # Score: prefer candidates that use fewer small diffs and leave more flexibility
                # Lower score is better
                score = small_count * 10 - future_flex
                
                candidates.append((score, x, new_diffs))
                
                # Limit candidates to keep search tractable
                if len(candidates) >= 30:
                    break
            
            if not candidates:
                rows.pop()
                return False
            
            # Sort by score (ascending) - best first
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates in order, with some randomness for diversification
            # Use a priority queue approach: try the best first, but allow backtracking
            for score, x, new_diffs in candidates:
                row.append(x)
                used_diffs |= new_diffs
                
                if build_row(pos + 1, x):
                    return True
                
                row.pop()
                used_diffs -= new_diffs
            
            rows.pop()
            return False
        
        success = build_row(1, 0)
        if success:
            return True
        
        rows.pop()
        return False
    
    # Try multiple restarts with different random seeds and scope limits
    random.seed(42)
    
    for restart in range(2000):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly perturb the search: use a slightly randomized order
        # by adding noise to the scoring
        noise_scale = 0.1 * (restart % 5)  # Increase noise over time
        
        rows = []
        used_diffs = set()
        
        # Use a randomized greedy approach with backtracking
        # For each row, try to build it with some randomness in candidate selection
        
        def build_row_randomized(row_idx, used_diffs, rows, scope_limit):
            """
            Build a single row with randomized candidate selection.
            Returns the completed row or None if failed.
            """
            if time.time() - start_time > time_limit:
                return None
            
            row = [0]
            
            for pos in range(k):
                last_val = row[-1]
                candidates = []
                
                for x in range(last_val + 1, scope_limit + 1):
                    new_diffs = {x - a for a in row}
                    if new_diffs & used_diffs:
                        continue
                    
                    small_count = sum(1 for d in new_diffs if d <= 20)
                    temp_used = used_diffs | new_diffs
                    future_flex = count_future_slots(temp_used, row, n_rows - row_idx - 1, k)
                    score = small_count * 10 - future_flex
                    
                    candidates.append((score, x, new_diffs))
                    
                    if len(candidates) >= 40:
                        break
                
                if not candidates:
                    return None
                
                candidates.sort(key=lambda item: item[0])
                
                # Select with some randomness: pick from top candidates
                # Weight towards better scores but allow exploration
                top_n = min(15, len(candidates))
                top_candidates = candidates[:top_n]
                
                # Use softmax-like selection: probability proportional to exp(-score/temperature)
                temperature = 2.0 + 0.5 * (restart % 4)
                import math
                weights = [math.exp(-item[0] / temperature) for item in top_candidates]
                total_weight = sum(weights)
                if total_weight == 0:
                    chosen = top_candidates[0]
                else:
                    r = random.random() * total_weight
                    cumsum = 0
                    chosen = top_candidates[-1]
                    for i, w in enumerate(weights):
                        cumsum += w
                        if r <= cumsum:
                            chosen = top_candidates[i]
                            break
                
                x = chosen[1]
                new_diffs = chosen[2]
                row.append(x)
                used_diffs |= new_diffs
            
            return row
        
        # Build all rows
        success = True
        for i in range(n_rows):
            row = build_row_randomized(i, used_diffs, rows, best_scope - 1 if best_scope < float('inf') else initial_upper)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if success:
            current_scope = max(r[-1] for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_result = [r[:] for r in rows]
                if best_scope <= 105:
                    break
    
    if best_result is None:
        # Fallback: deterministic greedy
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
