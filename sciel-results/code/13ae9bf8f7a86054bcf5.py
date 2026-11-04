import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit
    
    best_rows = None
    best_score = float('inf')
    
    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    # Precompute all valid rows up to a max scope
    # A valid row is a list of k+1 increasing integers starting at 0
    # such that its internal differences are unique
    # We also need to track the max element (scope)
    
    # Strategy: 
    # 1. Generate candidate rows greedily/backtracking with various scopes
    # 2. Build a compatibility graph (edge if diff sets are disjoint)
    # 3. Find max clique (or just a clique of size n_rows) with min max scope
    
    # Since n_rows and k are small (7, 5), we can be a bit exhaustive but smart
    
    # First, let's try to find a good solution via randomized greedy with restarts
    # but also incorporate a local search that tries to reduce the max element.
    
    def randomized_greedy(max_scope_hint=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                valids = []
                last = row[-1]
                # Cap the search for next element
                cap = max_scope_hint if max_scope_hint else 200
                if best_score < float('inf'):
                    cap = min(cap, best_score)
                
                for x in range(last + 1, cap + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        valids.append(x)
                
                if not valids:
                    return None
                
                # Weighted random choice, favoring smaller x
                if len(valids) > 1:
                    # Use 1/x weighting
                    weights = [1.0 / (v + 1) for v in valids[:20]]
                    total_w = sum(weights)
                    r = random.uniform(0, total_w)
                    acc = 0
                    chosen = valids[0]
                    for v, w in zip(valids[:20], weights):
                        acc += w
                        if r <= acc:
                            chosen = v
                            break
                else:
                    chosen = valids[0]
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows

    # Local search: try to reduce the max element by moving individual elements
    def local_search(rows):
        current_max = max(max(r) for r in rows)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try to improve each row
            for i in range(n_rows):
                row = rows[i]
                # Iterate from last element to first (excluding 0)
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    # Try smaller values
                    min_val = row[pos-1] + 1
                    max_val = current_val - 1  # We want to reduce
                    
                    if max_val < min_val:
                        continue
                        
                    # Try values from smallest to largest
                    for v in range(min_val, max_val + 1):
                        # Create a copy of the row with the change
                        candidate_row = list(row)
                        candidate_row[pos] = v
                        
                        # Check if the new row is internally valid (strictly increasing, unique diffs)
                        # And if the global set of diffs is valid
                        is_valid = True
                        all_diffs = set()
                        
                        # Check all rows
                        for r_idx, r in enumerate(rows):
                            if r_idx == i:
                                r_to_check = candidate_row
                            else:
                                r_to_check = r
                            
                            # Check strictly increasing
                            for j in range(1, len(r_to_check)):
                                if r_to_check[j] <= r_to_check[j-1]:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                                
                            # Get diffs
                            d_set = row_diffs(r_to_check)
                            if len(d_set) != len(r_to_check) * (len(r_to_check) - 1) // 2:
                                is_valid = False
                                break
                                
                            # Check for global duplicates
                            if all_diffs & d_set:
                                is_valid = False
                                break
                            
                            all_diffs.update(d_set)
                        
                        if is_valid:
                            # Accept the change
                            rows[i] = candidate_row
                            new_max = max(max(r) for r in rows)
                            if new_max < current_max:
                                current_max = new_max
                                improved = True
                            break # Break inner for v loop
                    if improved:
                        break # Break inner for pos loop
                if improved:
                    break # Break outer for i loop
        return rows, current_max

    # Main loop
    attempts = 0
    while time.time() - start_time < time_limit:
        # Try randomized greedy
        rows = randomized_greedy()
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                # Try local search
                if time.time() - start_time < time_limit - 5:
                    improved_rows, improved_score = local_search(best_rows)
                    # Validate improved_rows before accepting
                    if improved_score < best_score:
                        # Quick validation
                        all_diffs = set()
                        valid = True
                        for r in improved_rows:
                            d_set = row_diffs(r)
                            if all_diffs & d_set:
                                valid = False
                                break
                            all_diffs.update(d_set)
                            # Check internal validity (strictly increasing)
                            for j in range(1, len(r)):
                                if r[j] <= r[j-1]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            best_score = improved_score
                            best_rows = improved_rows
                if best_score <= 105:
                    break
        attempts += 1
        if attempts % 500 == 0:
            # Adjust max_scope_hint for greedy if we have a good best
            pass

    if best_rows is None:
        # Fallback to deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
