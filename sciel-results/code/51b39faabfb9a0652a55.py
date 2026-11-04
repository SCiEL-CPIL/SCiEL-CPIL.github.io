import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0
    
    # Fallback deterministic greedy
    def deterministic_greedy():
        used, rows = set(), []
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

    # Randomized greedy
    def try_randomized_greedy():
        best_rows = deterministic_greedy()
        best_scope = max(max(r) for r in best_rows)
        
        while time.time() - start_time < deadline:
            rows, scope = try_construct_randomized(n_rows, k, best_scope)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if best_scope <= 105: # Theoretical lower bound
                    break
        return best_rows

    def try_construct_randomized(n_rows, k, current_best_scope):
        used = set()
        rows = []
        
        for i in range(n_rows):
            row = [0]
            last = 0
            while len(row) < k + 1:
                candidates = []
                search_limit = last + 100
                x = last
                while x < search_limit and len(candidates) < 10:
                    x += 1
                    if x >= current_best_scope:
                        return None, float('inf')
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append((x, new_diffs))
                
                if not candidates:
                    return None, float('inf')
                
                if random.random() < 0.3:
                    x, new_diffs = candidates[0]
                else:
                    pool = candidates[:5]
                    x, new_diffs = random.choice(pool)
                
                row.append(x)
                used.update(new_diffs)
                last = x
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope

    # Exact Search
    def find_clique_solution(M):
        rulers = []
        
        def gen_ruler(current_row, current_diffs, last_val):
            if len(current_row) == k + 1:
                rulers.append((current_row, frozenset(current_diffs)))
                return
            
            # Pruning
            if last_val + (k + 1 - len(current_row)) > M:
                return
            
            for x in range(last_val + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & current_diffs:
                    continue
                
                gen_ruler(current_row + [x], current_diffs | new_diffs, x)
            
        gen_ruler([0], set(), 0)
        
        if not rulers:
            return None
        
        n_rulers = len(rulers)
        
        # If n_rulers is very large, we might have issues, but for this task it's small.
        
        found_clique = [False]
        result_rows = [None]
        
        def dfs_clique(clique, available):
            if found_clique[0]:
                return
            if len(clique) == n_rows:
                found_clique[0] = True
                result_rows[0] = [rulers[i][0] for i in clique]
                return
            
            if len(available) < n_rows - len(clique):
                return
            
            # Heuristic: sort available by degree in the remaining graph?
            # Or just iterate.
            # To speed up, we can pick the most constrained node first.
            # But for n_rows=7, simple iteration is fine.
            
            for i in list(available):
                if found_clique[0]:
                    return
                
                new_clique = clique + [i]
                
                new_available = set()
                for j in available:
                    if j == i:
                        continue
                    if not (rulers[i][1] & rulers[j][1]):
                        new_available.add(j)
                
                if len(new_available) < n_rows - len(new_clique):
                    continue
                
                dfs_clique(new_clique, new_available)
                
                if found_clique[0]:
                    return
        
        for start_i in range(n_rulers):
            if found_clique[0]:
                break
            
            available = set()
            for j in range(n_rulers):
                if j == start_i:
                    continue
                if not (rulers[start_i][1] & rulers[j][1]):
                    available.add(j)
            
            if len(available) < n_rows - 1:
                continue
                
            dfs_clique([start_i], available)
            
            if found_clique[0]:
                break
        
        if found_clique[0]:
            return result_rows[0]
        return None

    best_rows = None
    best_scope = float('inf')
    
    # Search M from 100 to 112
    for M in range(100, 113):
        if time.time() - start_time > deadline - 15:
            break
        
        rows = find_clique_solution(M)
        if rows:
            best_rows = rows
            best_scope = M
            break
    
    if best_rows is None:
        best_rows = try_randomized_greedy()
    
    return best_rows
