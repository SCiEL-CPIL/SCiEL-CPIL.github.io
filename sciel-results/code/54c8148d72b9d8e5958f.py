import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflicts
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i].add(j)
                conflict[j].add(i)
    
    def reciprocal_sum(indices):
        return sum(1.0 / numbers[i] for i in indices)
    
    # Greedy: try multiple orderings
    best_selected = set()
    best_sum = 0.0
    
    # Ordering 1: ascending a
    order1 = sorted(range(n), key=lambda i: numbers[i])
    selected = set()
    for i in order1:
        if not (conflict[i] & selected):
            selected.add(i)
    s = reciprocal_sum(selected)
    if s > best_sum:
        best_sum = s
        best_selected = set(selected)
    
    # Ordering 2: descending a (start with large numbers, fewer conflicts)
    order2 = sorted(range(n), key=lambda i: -numbers[i])
    selected = set()
    for i in order2:
        if not (conflict[i] & selected):
            selected.add(i)
    s = reciprocal_sum(selected)
    if s > best_sum:
        best_sum = s
        best_selected = set(selected)
    
    # Ordering 3: by 1/a / degree (weight per conflict)
    order3 = sorted(range(n), key=lambda i: (1.0 / numbers[i]) / (len(conflict[i]) + 1), reverse=True)
    selected = set()
    for i in order3:
        if not (conflict[i] & selected):
            selected.add(i)
    s = reciprocal_sum(selected)
    if s > best_sum:
        best_sum = s
        best_selected = set(selected)
    
    selected = best_selected
    
    # Local search
    def elapsed():
        return time.time() - start_time
    
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        # Try single swap: remove i, add j
        for i in list(selected):
            new_sel = selected - {i}
            for j in range(n):
                if j in new_sel:
                    continue
                if not (conflict[j] & new_sel):
                    cand = new_sel | {j}
                    s = reciprocal_sum(cand)
                    if s > best_sum + 1e-12:
                        selected = cand
                        best_sum = s
                        best_selected = set(selected)
                        improved = True
                        break
        
        # Try multi-add: remove i, add multiple
        if not improved and elapsed() < DEADLINE:
            for i in list(selected):
                new_sel = selected - {i}
                # Add all compatible in descending 1/a order
                candidates = [j for j in range(n) if j not in new_sel and not (conflict[j] & new_sel)]
                candidates.sort(key=lambda j: numbers[j])
                for j in candidates:
                    if not (conflict[j] & new_sel):
                        new_sel.add(j)
                s = reciprocal_sum(new_sel)
                if s > best_sum + 1e-12:
                    selected = new_sel
                    best_sum = s
                    best_selected = set(selected)
                    improved = True
    
    return [numbers[i] for i in best_selected]
