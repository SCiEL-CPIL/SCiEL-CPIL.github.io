import time
import random

def construct(n_rows, k):
    start = time.time()
    deadline = start + 140  # Safety margin

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None, randomize=False):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    def dfs_row(row, used_set, cap, row_deadline, randomize=False):
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None
        
        candidates = list(range(last + 1, upper + 1))
        if randomize:
            random.shuffle(candidates)
        else:
            # Try smaller first to find compact rows, but maybe mix?
            # For SA, we want diversity.
            pass 

        for x in candidates:
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline, randomize)
            if res is not None:
                return res
        return None

    # Phase 1: Greedy Baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        rows.append(row)
        used |= row_diffs(row)
    
    current_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = current_max

    # Phase 2: Simulated Annealing
    # We will try to replace rows with new ones.
    # Temperature schedule
    T = 10.0
    cooling = 0.995
    
    # We need a function to generate a candidate row for index i
    # that avoids all other rows' differences.
    def generate_candidate(i, randomize=True):
        # Calculate used differences excluding row i
        used_without_i = set(used)
        used_without_i.difference_update(row_diffs(rows[i]))
        
        # Try to find a row. We want it to be "good" (small) but diverse.
        # We don't know the target cap. We can try to find *any* valid row first, 
        # or try to find one with a specific cap.
        # To allow SA to explore, we should allow rows that are slightly larger 
        # if they help the global structure? 
        # Actually, the objective is max(last). 
        # If we pick a row with last > current_max, the objective increases.
        # So we should constrain the cap to be <= current_max (or slightly higher if we want to jump).
        
        # Let's try to find a row with cap = current_max.
        # If found, it might be different and allow others to shrink.
        
        row_deadline = min(deadline - 5, time.time() + 0.5) # 500ms per candidate is generous? 
        # n_rows is small (7), k is small (6). 
        # DFS should be fast.
        
        cand = dfs_row([0], used_without_i, current_max, row_deadline, randomize=randomize)
        return cand

    # Main SA Loop
    while time.time() < deadline - 10:
        if T < 0.1:
            break
            
        # Pick a random row to perturb
        i = random.randint(0, n_rows - 1)
        
        cand = generate_candidate(i, randomize=True)
        
        if cand is not None:
            # Calculate new max if we replace rows[i] with cand
            # We don't need to rebuild all rows, just check the max of the set
            # Note: This move is "independent" in the sense that we only change row i.
            # The validity is guaranteed because cand avoids all other rows.
            
            new_last = cand[-1]
            # The new global max is max(current_max_of_other_rows, new_last)
            # current_max_of_other_rows is max(rows[j][-1] for j != i)
            
            # Optimization: Precompute max excluding i? 
            # n_rows is small, just compute it.
            max_others = 0
            for j in range(n_rows):
                if j != i:
                    if rows[j][-1] > max_others:
                        max_others = rows[j][-1]
            
            new_global_max = max(max_others, new_last)
            
            delta = new_global_max - current_max
            
            # Acceptance probability
            if delta < 0:
                accept = True
            else:
                # Probability of accepting worse solution
                # If T is high, accept.
                prob = 1.0 / (1.0 + random.random() * 1000) # Simple random check? 
                # Better: exp(-delta / T)
                import math
                try:
                    prob = math.exp(-delta / T)
                except:
                    prob = 0.0
                    
                accept = random.random() < prob
            
            if accept:
                # Update state
                used.difference_update(row_diffs(rows[i]))
                rows[i] = cand
                used |= row_diffs(cand)
                
                current_max = new_global_max
                
                if current_max < best_max:
                    best_max = current_max
                    best_rows = rows[:]
        
        T *= cooling

    # Phase 3: Final Polish (Greedy refinement on best_rows)
    # Reconstruct from best_rows to ensure validity and try to shrink further with deterministic greedy
    # Actually, best_rows is already valid.
    # Let's try a quick deterministic improvement on best_rows
    final_rows = best_rows[:]
    final_used = set()
    for r in final_rows:
        final_used |= row_diffs(r)
        
    # Try to shrink each row in final_rows using DFS (deterministic, smallest first)
    for i in range(n_rows):
        if time.time() > deadline - 2:
            break
        cur_last = final_rows[i][-1]
        if cur_last == 0: continue
        
        used_without = set(final_used)
        used_without.difference_update(row_diffs(final_rows[i]))
        
        # Try to find a smaller row
        found = None
        # Try caps from cur_last - 1 down to 0
        # Limit search depth/time
        row_deadline = min(deadline - 2, time.time() + 1.0)
        
        # We can just call dfs_row with cap = cur_last - 1
        # If it finds one, it's smaller.
        cand = dfs_row([0], used_without, cur_last - 1, row_deadline, randomize=False)
        if cand is not None:
            final_used.difference_update(row_diffs(final_rows[i]))
            final_rows[i] = cand
            final_used |= row_diffs(cand)

    return final_rows
