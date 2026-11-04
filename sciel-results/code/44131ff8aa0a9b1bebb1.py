import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # D total diffs
    D = n_rows * k * (k + 1) // 2
    lower_bound = D
    
    # Greedy construction
    def pure_greedy():
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
        return rows
    
    best_rows = pure_greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Local search / simulated annealing to improve
    def local_search(rows, max_iters=50000):
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        current_scope = max(max(r) for r in rows)
        best_scope_ls = current_scope
        best_rows_ls = [r[:] for r in rows]
        
        temperature = 10.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random row and a random position
            row_idx = random.randint(0, n_rows-1)
            # We can only change positions 1..k (position 0 is always 0)
            pos = random.randint(1, k)
            
            old_val = rows[row_idx][pos]
            
            # Try to find a new value for this position
            # The new value must be > rows[row_idx][pos-1] and < rows[row_idx][pos+1] (if pos < k)
            min_val = rows[row_idx][pos-1] + 1
            max_val = rows[row_idx][pos+1] - 1 if pos < k else best_scope_ls
            
            if min_val > max_val:
                continue
            
            # Remove old differences involving this position
            old_diffs = set()
            for j in range(k+1):
                if j != pos:
                    d = abs(rows[row_idx][pos] - rows[row_idx][j])
                    old_diffs.add(d)
            
            # Try a random new value in range
            new_val = random.randint(min_val, max_val)
            
            new_diffs = set()
            for j in range(k+1):
                if j != pos:
                    d = abs(new_val - rows[row_idx][j])
                    new_diffs.add(d)
            
            # Check validity
            # The new_diffs must not conflict with any other diffs (except the ones we removed)
            # But note: new_diffs might conflict with old_diffs of the SAME row if we didn't remove them properly?
            # No, we removed all diffs involving pos.
            # So we just need to check if new_diffs intersects (used_diffs - old_diffs)
            
            conflict = False
            for d in new_diffs:
                if d in used_diffs and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                # Maybe try a few more values? Or just reject
                # Let's try to find a valid one quickly
                found = False
                # Try a few random candidates
                for _ in range(5):
                    nv = random.randint(min_val, max_val)
                    if nv == old_val:
                        continue
                    nd = set()
                    for j in range(k+1):
                        if j != pos:
                            nd.add(abs(nv - rows[row_idx][j]))
                    ok = True
                    for d in nd:
                        if d in used_diffs and d not in old_diffs:
                            ok = False
                            break
                    if ok:
                        new_val = nv
                        new_diffs = nd
                        found = True
                        break
                if not found:
                    continue
            
            # Apply move
            rows[row_idx][pos] = new_val
            for d in old_diffs:
                used_diffs.discard(d)
            for d in new_diffs:
                used_diffs.add(d)
            
            new_scope = max(max(r) for r in rows)
            
            if new_scope < current_scope:
                current_scope = new_scope
                if new_scope < best_scope_ls:
                    best_scope_ls = new_scope
                    best_rows_ls = [r[:] for r in rows]
            else:
                # Accept with probability exp(-(delta)/T)
                delta = new_scope - current_scope
                if delta > 0 and random.random() < pow(2.718, -delta/temperature):
                    current_scope = new_scope
                else:
                    # Revert
                    rows[row_idx][pos] = old_val
                    for d in new_diffs:
                        used_diffs.discard(d)
                    for d in old_diffs:
                        used_diffs.add(d)
            
            temperature *= cooling
        
        return best_rows_ls, best_scope_ls
    
    # Run local search
    ls_rows, ls_scope = local_search(best_rows)
    if ls_scope < best_scope:
        best_rows = ls_rows
        best_scope = ls_scope
    
    # Try backtracking for small M if time permits
    # Only useful if n_rows and k are small enough
    if n_rows <= 4 and k <= 7:
        def solve_for_scope(M, deadline=None):
            used_mask = 0
            rows = []
            
            def backtrack(row_idx, current_row, used_mask):
                if time.time() > deadline - 2.0:
                    return False
                
                if row_idx == n_rows:
                    return True
                
                if len(current_row) == 0:
                    start_val = 1
                    for x in range(start_val, M + 1):
                        new_diffs = {x - a for a in current_row}
                        new_mask = 0
                        valid = True
                        for d in new_diffs:
                            if d > M:
                                valid = False
                                break
                            bit = 1 << d
                            if used_mask & bit:
                                valid = False
                                break
                            new_mask |= bit
                        
                        if not valid:
                            continue
                        
                        new_row = current_row + [x]
                        new_used_mask = used_mask | new_mask
                        
                        if len(new_row) == k + 1:
                            rows.append(new_row)
                            if backtrack(row_idx + 1, [], new_used_mask):
                                return True
                            rows.pop()
                        else:
                            if backtrack(row_idx, new_row, new_used_mask):
                                return True
                
                return False
            
            return backtrack(0, [], 0), rows
        
        # Try a few M values below current best
        for M in range(lower_bound, min(best_scope, lower_bound + 10)):
            if time.time() > deadline - 10.0:
                break
            success, res_rows = solve_for_scope(M, deadline=deadline)
            if success:
                best_rows = res_rows
                best_scope = M
                break
    
    return best_rows
