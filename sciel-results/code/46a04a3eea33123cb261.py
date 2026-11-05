import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def validate(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True
    
    def greedy_row(used_diffs, scope_bound, rng=None):
        """Generate a row using greedy with optional randomization"""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > scope_bound:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs = used_diffs | new_diffs
        return row
    
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, 1000)
            if row is None:
                break
            used = used | {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
            rows.append(row)
        return rows
    
    greedy = greedy_solution()
    best_result = [row[:] for row in greedy]
    best_scope = get_scope(greedy)
    
    def solve(scope_bound, seed, max_nodes=400000):
        if time.time() > deadline - 3:
            return None
        
        rng = random.Random(seed)
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        node_count = [0]
        
        # Try different row orderings
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        def backtrack(order_idx):
            if time.time() > deadline - 2:
                return False
            
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False
            
            if order_idx == n_rows:
                return True
            
            row_idx = row_order[order_idx]
            row = rows[row_idx]
            last = row[-1]
            
            candidates = list(range(last + 1, scope_bound + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(order_idx + 1):
                    return True
                
                row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        if backtrack(0):
            return rows
        return None
    
    # Try multiple restarts with different scope bounds
    current_scope = best_scope
    
    for attempt in range(300):
        if time.time() > deadline - 5:
            break
        
        result = solve(current_scope, seed=attempt * 7919 + 13, max_nodes=300000)
        if result is not None:
            scope = get_scope(result)
            if scope < best_scope:
                best_scope = scope
                best_result = [row[:] for row in result]
                current_scope = scope
    
    # Try to find even better with lower scopes
    for scope_try in range(best_scope - 1, max(100, 0), -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            result = solve(scope_try, seed=attempt * 31 + scope_try * 7, max_nodes=200000)
            if result is not None:
                found = True
                best_scope = scope_try
                best_result = [row[:] for row in result]
                break
        
        if found:
            continue
        else:
            break
    
    # Local improvement: try to reduce scope by regenerating rows
    if time.time() < deadline - 5:
        rng = random.Random(42)
        current = [row[:] for row in best_result]
        current_scope = best_scope
        
        for _ in range(1000):
            if time.time() > deadline - 3:
                break
            
            row_idx = rng.randint(0, n_rows - 1)
            
            # Compute used diffs excluding this row
            used_diffs = set()
            for i, row in enumerate(current):
                if i == row_idx:
                    continue
                for a in range(len(row)):
                    for b in range(a + 1, len(row)):
                        used_diffs.add(row[b] - row[a])
            
            # Try to regenerate with tighter scope
            new_row = greedy_row(used_diffs, current_scope, rng)
            if new_row is not None:
                new_scope = max(current_scope, max(new_row))
                if new_scope <= current_scope:
                    current[row_idx] = new_row
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_result = [row[:] for row in current]
    
    if validate(best_result):
        return best_result
    
    return greedy
