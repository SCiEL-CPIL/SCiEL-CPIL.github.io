# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        return len(all_diffs) == len(set(all_diffs))

    def scope_of(rows):
        return max(max(r) for r in rows)

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < k + 1:
                found = False
                for x in range(last + 1, 2000):
                    new_diffs = [x - a for a in row]
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    for d in new_diffs:
                        used.add(d)
                    last = x
                    found = True
                    break
                if not found:
                    break
            rows.append(row)
        return rows

    def local_search(rows, time_limit):
        rows = [r[:] for r in rows]
        n = n_rows
        L = k + 1
        
        used = set()
        for r in rows:
            for d in get_diffs(r):
                used.add(d)
        
        best_scope = scope_of(rows)
        best_rows = [r[:] for r in rows]
        current_scope = best_scope
        
        while time.time() < time_limit:
            max_val = current_scope
            max_indices = []
            for r in range(n):
                for j in range(1, L):
                    if rows[r][j] == max_val:
                        max_indices.append((r, j))
            
            if max_indices and random.random() < 0.8:
                r, j = random.choice(max_indices)
            else:
                r = random.randrange(n)
                j = random.randrange(1, L)
            
            old_val = rows[r][j]
            lower_bound = rows[r][j-1] + 1
            upper_bound = (rows[r][j+1] - 1) if j + 1 < L else max_val
            
            # Generate candidates
            cands = set()
            # Try to decrease
            for d in range(1, 10):
                if old_val - d >= lower_bound:
                    cands.add(old_val - d)
            # Try to increase slightly (for exploration, if not max)
            if rows[r][j] < max_val:
                for d in range(1, 5):
                    if old_val + d <= upper_bound:
                        cands.add(old_val + d)
            
            # Randoms
            for _ in range(5):
                cands.add(random.randint(lower_bound, upper_bound))
            
            cands.discard(old_val)
            if not cands:
                continue
            
            cands_list = list(cands)
            # Prioritize smaller values if we are trying to reduce scope
            # Sort by value
            cands_list.sort()
            
            for new_val in cands_list:
                if new_val < lower_bound or new_val > upper_bound:
                    continue
                
                # Compute old diffs involving j
                old_diffs_set = set()
                for i in range(L):
                    if i != j:
                        if i < j:
                            d = old_val - rows[r][i]
                        else:
                            d = rows[r][i] - old_val
                        old_diffs_set.add(d)
                
                # Compute new diffs involving j
                new_diffs = []
                for i in range(L):
                    if i != j:
                        if i < j:
                            d = new_val - rows[r][i]
                        else:
                            d = rows[r][i] - new_val
                        new_diffs.append(d)
                        
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                conflict = False
                for d in new_diffs:
                    if d in used and d not in old_diffs_set:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Apply
                rows[r][j] = new_val
                for d in old_diffs_set:
                    used.discard(d)
                for d in new_diffs:
                    used.add(d)
                
                new_scope = scope_of(rows)
                
                if new_scope < current_scope:
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                elif new_scope == current_scope:
                    if random.random() < 0.3:
                        # Revert to stay on best path? Or explore?
                        # Let's revert to avoid drifting
                        rows[r][j] = old_val
                        for d in new_diffs:
                            used.discard(d)
                        for d in old_diffs_set:
                            used.add(d)
                        continue
                else:
                    # Revert
                    rows[r][j] = old_val
                    for d in new_diffs:
                        used.discard(d)
                    for d in old_diffs_set:
                        used.add(d)
                    continue
                
                break
            
        return best_rows, best_scope

    rows = greedy_construct()
    best_rows = [r[:] for r in rows]
    best_scope = scope_of(rows)
    
    improved_rows, sc = local_search(rows, time_limit=deadline - 5.0)
    if sc < best_scope:
        best_rows = improved_rows
        best_scope = sc
        
    if not is_valid(best_rows):
        best_rows = greedy_construct()
        
    return best_rows
# EVOLVE-BLOCK-END
