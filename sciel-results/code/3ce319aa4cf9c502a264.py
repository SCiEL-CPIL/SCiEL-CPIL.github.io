import random
import time
import math
from collections import defaultdict

def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 140.0
    
    def greedy():
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
    
    rows = greedy()
    best = [r[:] for r in rows]
    best_scope = max(r[-1] for r in best)
    
    current = [r[:] for r in rows]
    diff_counts = defaultdict(int)
    for row in current:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diff_counts[row[j] - row[i]] += 1
    
    def get_conflicts(dc):
        return sum(v - 1 for v in dc.values() if v > 1)
    
    current_conflicts = get_conflicts(diff_counts)
    current_scope = max(r[-1] for r in current)
    current_energy = current_conflicts * 10000 + current_scope
    
    T = 500.0
    T_min = 0.5
    cooling = 0.9999
    
    while time.time() < deadline:
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        old_val = current[r][pos]
        
        if current_conflicts == 0:
            delta = random.randint(-15, 5)
        else:
            delta = random.randint(-10, 10)
        
        new_val = old_val + delta
        
        if new_val <= current[r][pos-1] or (pos < k and new_val >= current[r][pos+1]) or new_val < 1:
            continue
        
        row = current[r]
        
        affected = {}
        for i in range(k+1):
            if i != pos:
                if i < pos:
                    old_d = old_val - row[i]
                    new_d = new_val - row[i]
                else:
                    old_d = row[i] - old_val
                    new_d = row[i] - new_val
                affected[old_d] = affected.get(old_d, 0) - 1
                affected[new_d] = affected.get(new_d, 0) + 1
        
        conflict_delta = 0
        for d, cnt in affected.items():
            c_old = diff_counts.get(d, 0)
            c_new = c_old + cnt
            conflict_delta += max(0, c_new - 1) - max(0, c_old - 1)
        
        for d, cnt in affected.items():
            diff_counts[d] = diff_counts.get(d, 0) + cnt
            if diff_counts[d] == 0:
                del diff_counts[d]
        
        row[pos] = new_val
        
        new_scope = max(row2[-1] for row2 in current)
        new_conflicts = current_conflicts + conflict_delta
        new_energy = new_conflicts * 10000 + new_scope
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_conflicts = new_conflicts
            current_scope = new_scope
            if new_conflicts == 0 and new_scope < best_scope:
                best = [row2[:] for row2 in current]
                best_scope = new_scope
        else:
            p = math.exp(-(new_energy - current_energy) / T)
            if random.random() < p:
                current_energy = new_energy
                current_conflicts = new_conflicts
                current_scope = new_scope
            else:
                for d, cnt in affected.items():
                    diff_counts[d] = diff_counts.get(d, 0) - cnt
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                row[pos] = old_val
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best
