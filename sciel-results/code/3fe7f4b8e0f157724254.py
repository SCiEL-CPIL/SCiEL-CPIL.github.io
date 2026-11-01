import time
import random
import math
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = 140  # 10s margin before 150s kill
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        used = set()
        for row in rows:
            d = get_row_diffs(row)
            if len(d) != len(row) * (len(row) - 1) // 2:
                return False
            if d & used:
                return False
            used |= d
        return True
    
    def greedy_init(rng, n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                x = row[-1] + 1
                while x < 300:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                    x += 1
                if not found:
                    return None
            rows.append(row)
        return rows, used
    
    def random_greedy(rng, n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                base = row[-1]
                candidates = []
                for x in range(base + 1, min(base + 15, 250)):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                if not candidates:
                    x = base
                    while x < 250:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                    if not candidates:
                        return None
                pick = rng.choice(candidates[:min(3, len(candidates))])
                row.append(pick)
                used |= {pick - a for a in row[:-1]}
            rows.append(row)
        return rows, used
    
    def local_search(rows, used_diffs, n_rows, k, rng, max_iter, deadline):
        best_rows = [r[:] for r in rows]
        best_M = max(max(r) for r in rows)
        current_M = best_M
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            row = rows[ri]
            
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else current_M + 5
            
            if lo > hi:
                continue
            
            old_val = row[pi]
            old_diffs = get_row_diffs(row)
            
            # Try to find a valid replacement that reduces or maintains M
            candidates = list(range(lo, hi + 1))
            rng.shuffle(candidates)
            
            found = False
            for new_val in candidates[:20]:
                if new_val == old_val:
                    continue
                
                new_row = row[:]
                new_row[pi] = new_val
                if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                
                new_diffs = get_row_diffs(new_row)
                expected = len(new_row) * (len(new_row) - 1) // 2
                if len(new_diffs) != expected:
                    continue
                
                added = new_diffs - old_diffs
                if added & (used_diffs - old_diffs):
                    continue
                
                # Valid move
                new_M = max(max(new_row), max(max(r) for i, r in enumerate(rows) if i != ri))
                
                if new_M <= current_M:
                    # Accept if same or better
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    rows[ri] = new_row
                    current_M = new_M
                    
                    if current_M < best_M:
                        best_M = current_M
                        best_rows = [r[:] for r in rows]
                    found = True
                    break
        
        return best_rows, best_M
    
    def simulated_anneal(rows, used_diffs, n_rows, k, rng, deadline):
        best_rows = [r[:] for r in rows]
        best_M = max(max(r) for r in rows)
        current_M = best_M
        
        T = 30.0
        cooling = 0.997
        
        while time.time() < deadline and T > 0.3:
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            row = rows[ri]
            
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else current_M + 10
            
            if lo > hi:
                continue
            
            old_val = row[pi]
            old_diffs = get_row_diffs(row)
            
            # Generate new value with bias toward smaller values
            if random.random() < 0.7:
                new_val = rng.randint(lo, min(hi, current_M))
            else:
                new_val = rng.randint(lo, hi)
            
            if new_val == old_val:
                continue
            
            new_row = row[:]
            new_row[pi] = new_val
            if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                continue
            
            new_diffs = get_row_diffs(new_row)
            expected = len(new_row) * (len(new_row) - 1) // 2
            if len(new_diffs) != expected:
                continue
            
            added = new_diffs - old_diffs
            if added & (used_diffs - old_diffs):
                continue
            
            # Valid move
            new_M = max(max(new_row), max(max(r) for i, r in enumerate(rows) if i != ri))
            delta_E = new_M - current_M
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                rows[ri] = new_row
                current_M = new_M
                
                if current_M < best_M:
                    best_M = current_M
                    best_rows = [r[:] for r in rows]
            
            T *= cooling
        
        return best_rows, best_M
    
    rng = random.Random(42)
    
    # Phase 1: Try multiple greedy initializations
    best_initial_rows = None
    best_initial_M = float('inf')
    
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        
        rows = None
        if attempt == 0:
            rows, used = greedy_init(rng, n_rows, k)
        else:
            rows, used = random_greedy(rng, n_rows, k)
        
        if rows is None:
            continue
        
        M = max(max(r) for r in rows)
        if M < best_initial_M:
            best_initial_M = M
            best_initial_rows = [r[:] for r in rows]
    
    if best_initial_rows is None:
        # Fallback: simple greedy
        rows, used = greedy_init(rng, n_rows, k)
        best_initial_rows = rows
        best_initial_M = max(max(r) for r in rows)
    
    # Phase 2: Local search on best initial
    best_rows, best_M = local_search(
        [r[:] for r in best_initial_rows], 
        get_row_diffs(best_initial_rows[0]) | set().union(*[get_row_diffs(r) for r in best_initial_rows]),
        n_rows, k, rng, 5000, deadline - 20
    )
    
    # Rebuild used_diffs properly
    used_diffs = set()
    for r in best_rows:
        used_diffs |= get_row_diffs(r)
    
    # Phase 3: Simulated annealing
    best_rows, best_M = simulated_anneal(
        [r[:] for r in best_rows],
        used_diffs,
        n_rows, k, rng, deadline - 5
    )
    
    # Final validation
    if not is_valid(best_rows):
        # Return the initial valid solution
        return best_initial_rows
    
    return best_rows
