import heapq
import itertools
import time
from math import comb

def construct(n_rows, k):
    # Baseline: Greedy to get an initial upper bound
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    D = comb(k + 1, 2)
    
    def get_heuristic(i, row, used_mask):
        # i: index of current row
        # row: current partial row
        # used_mask: bitmask of used diffs
        
        # Diffs in current row so far
        diffs_cur = comb(len(row), 2)
        # Diffs needed to finish current row
        diffs_cur_rem = D - diffs_cur
        # Diffs needed for remaining full rows
        rows_rem = n_rows - i - 1
        diffs_full_rem = rows_rem * D
        
        need = diffs_cur_rem + diffs_full_rem
        
        if need == 0:
            return 0
            
        # Find 'need' smallest positive integers not in used_mask
        found = 0
        max_new = 0
        d = 1
        while found < need:
            if not (used_mask & (1 << d)):
                found += 1
                max_new = d
            d += 1
        return max_new

    start_time = time.time()
    timeout = 115.0
    
    # State: (f, g, i, row, used_mask, path)
    # f = max(g, h)
    # g = max element in path + row
    # i = current row index
    # row = current partial row
    # used_mask = bitmask of used diffs
    # path = list of completed rows
    
    start_h = get_heuristic(0, [0], 0)
    start_state = (start_h, 0, 0, [0], 0, [])
    
    open_list = [start_state]
    visited = set()
    
    while open_list:
        if time.time() - start_time > timeout:
            break
            
        f, g, i, row, used_mask, path = heapq.heappop(open_list)
        
        state_key = (i, tuple(row), used_mask)
        if state_key in visited:
            continue
        visited.add(state_key)
        
        # If we found a complete solution
        if i == n_rows - 1 and len(row) == k + 1:
            # This is a goal state
            # Since A* pops in order of f, and f=g for goals, this is optimal
            return path
        
        # Expand state
        
        # Case 1: Current row is not full
        if len(row) < k + 1:
            last = row[-1]
            # Try next x
            # We can limit x to best_scope to prune
            # But we want to find the optimal, so we should explore all valid x
            # However, if x >= best_scope, we can't improve on the greedy solution?
            # Actually, if x >= best_scope, g' >= best_scope.
            # f' = max(g', h') >= best_scope.
            # If we are looking for a solution better than best_scope, we can prune.
            # But we don't know if a better solution exists yet.
            # The greedy solution is just an upper bound.
            # We should allow x to go up to best_scope - 1 to potentially find a better one.
            # If no better one is found, we return the greedy one.
            # So limit x to best_scope - 1.
            
            for x in range(last + 1, best_scope):
                new_diffs = [x - a for a in row]
                valid = True
                new_mask = 0
                for d in new_diffs:
                    if used_mask & (1 << d):
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if valid:
                    new_row = row + [x]
                    new_used = used_mask | new_mask
                    new_g = max(g, x)
                    
                    # Prune if new_g >= best_scope
                    if new_g >= best_scope:
                        continue
                        
                    new_h = get_heuristic(i, new_row, new_used)
                    new_f = max(new_g, new_h)
                    
                    # Prune if new_f >= best_scope
                    if new_f >= best_scope:
                        continue
                        
                    child_state = (new_f, new_g, i, new_row, new_used, path)
                    heapq.heappush(open_list, child_state)
        
        # Case 2: Current row is full
        if len(row) == k + 1 and i < n_rows - 1:
            # Move to next row
            new_i = i + 1
            new_row = [0]
            new_used = used_mask
            new_path = path + [row]
            
            new_g = g # g doesn't change, max of path is same
            
            new_h = get_heuristic(new_i, new_row, new_used)
            new_f = max(new_g, new_h)
            
            # Prune if new_f >= best_scope
            if new_f >= best_scope:
                continue
                
            child_state = (new_f, new_g, new_i, new_row, new_used, new_path)
            heapq.heappush(open_list, child_state)
            
    return best_rows
