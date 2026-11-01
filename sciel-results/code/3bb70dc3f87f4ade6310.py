import random
import time
from collections import Counter

def construct(n_rows, k):
    if n_rows == 4 and k == 7:
        return _solve_4_7()
    else:
        # Fallback to greedy
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

def _solve_4_7():
    n_rows = 4
    k = 7 # number of gaps
    
    # 1. Generate initial greedy solution
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
    
    # Convert rows to gaps
    gaps = []
    for r in rows:
        g = [r[i] - r[i-1] for i in range(1, len(r))]
        gaps.append(g)
    
    # State
    current_gaps = [g[:] for g in gaps]
    
    def get_entries(g):
        e = [0]
        s = 0
        for gap in g:
            s += gap
            e.append(s)
        return e
    
    def get_diffs(e):
        d = set()
        n = len(e)
        for i in range(n):
            for j in range(i+1, n):
                d.add(e[j] - e[i])
        return d

    def calc_state(gaps_list):
        all_diffs = []
        max_span = 0
        for g in gaps_list:
            e = get_entries(g)
            max_span = max(max_span, e[-1])
            d = get_diffs(e)
            all_diffs.append(d)
        
        all_diffs_flat = []
        for d in all_diffs:
            all_diffs_flat.extend(d)
        
        counts = Counter(all_diffs_flat)
        
        conflicts = 0
        for c in counts.values():
            if c > 1:
                conflicts += c - 1
                
        return max_span, conflicts

    # Initialize
    current_span, current_conflicts = calc_state(current_gaps)
    best_span = current_span
    best_gaps = [g[:] for g in current_gaps]
    
    start_time = time.time()
    time_limit = 145
    
    T = 20.0
    cooling_rate = 0.99999
    
    while time.time() - start_time < time_limit:
        r = random.randint(0, n_rows - 1)
        i = random.randint(0, k - 1)
        
        # Generate delta
        # Mix of small and large steps
        if random.random() < 0.8:
            delta = random.randint(-2, 2)
        else:
            delta = random.randint(-10, 10)
            
        new_g = current_gaps[r][i] + delta
        if new_g < 1:
            continue
            
        old_g = current_gaps[r][i]
        current_gaps[r][i] = new_g
        
        new_span, new_conflicts = calc_state(current_gaps)
        
        old_cost = current_span + 100 * current_conflicts
        new_cost = new_span + 100 * new_conflicts
        
        delta_cost = new_cost - old_cost
        
        # Acceptance
        if delta_cost < 0:
            accept = True
        else:
            # Avoid division by zero or very small T
            if T < 0.1:
                accept = False
            else:
                accept = random.random() < pow(2.71828, -delta_cost / T)
        
        if accept:
            current_span = new_span
            current_conflicts = new_conflicts
            
            if new_conflicts == 0 and new_span < best_span:
                best_span = new_span
                best_gaps = [g[:] for g in current_gaps]
        else:
            current_gaps[r][i] = old_g
            
        T *= cooling_rate
        
    # Return best found
    best_entries = []
    for g in best_gaps:
        best_entries.append(get_entries(g))
        
    return best_entries
