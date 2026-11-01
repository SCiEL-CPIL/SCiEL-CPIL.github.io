import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_scope = float('inf')
    best_solution = None
    
    def greedy_construct(max_scope_limit, beam_width=5):
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            
            for pos in range(k):
                # Compute forbidden values for the next element x
                # x is forbidden if x - a is in used_diffs for any a in row
                # i.e., x is in {d + a for d in used_diffs for a in row}
                forbidden = set()
                for a in row:
                    for d in used_diffs:
                        forbidden.add(d + a)
                
                candidates = []
                x = row[-1] + 1
                while x <= max_scope_limit:
                    if x not in forbidden:
                        candidates.append(x)
                        if len(candidates) >= beam_width:
                            break
                    x += 1
                
                if not candidates:
                    return None, None
                
                # Heuristic: Pick smallest with high probability, random otherwise
                if random.random() < 0.8:
                    chosen = candidates[0]
                else:
                    chosen = random.choice(candidates)
                
                row.append(chosen)
                
                # Update used_diffs with new differences
                for a in row[:-1]:
                    used_diffs.add(chosen - a)
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope

    # Estimate initial scope
    # Counting bound: scope * (scope+1)/2 >= n_rows * k(k+1)/2
    # scope^2 >= n_rows * k(k+1)
    est_scope = int((n_rows * k * (k+1)) ** 0.5) + 10
    max_scope_limit = est_scope + 20
    
    # Initial deterministic run
    rows, scope = greedy_construct(max_scope_limit, beam_width=1)
    if rows:
        best_solution = rows
        best_scope = scope
    
    # Refinement loop
    while time.time() < deadline:
        rows, scope = greedy_construct(best_scope, beam_width=5)
        if rows:
            if scope < best_scope:
                best_solution = rows
                best_scope = scope
        
    return best_solution
