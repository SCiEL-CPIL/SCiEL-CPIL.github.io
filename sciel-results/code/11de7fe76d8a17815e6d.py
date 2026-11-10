import numpy as np
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # Compute forbidden differences (nonzero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    # Build conflict graph: conflict[a][b] = True if a and b cannot both be selected
    conflict = [[False] * m for _ in range(m)]
    for a in range(m):
        for b in range(a + 1, m):
            d1 = (a - b) % m
            d2 = (b - a) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[a][b] = True
                conflict[b][a] = True
    
    def is_compatible(candidate, selected_set):
        for s in selected_set:
            if conflict[candidate][s]:
                return False
        return True
    
    best_set = set()
    
    # Multiple random greedy starts
    for trial in range(200):
        random.seed(trial)
        order = list(range(m))
        random.shuffle(order)
        
        current = set()
        for candidate in order:
            if is_compatible(candidate, current):
                current.add(candidate)
        
        if len(current) > len(best_set):
            best_set = set(current)
    
    # Local search: try to improve
    improved = True
    while improved:
        improved = False
        current = set(best_set)
        
        # Try adding one element
        for candidate in range(m):
            if candidate not in current:
                if is_compatible(candidate, current):
                    current.add(candidate)
                    improved = True
                    break
        
        # Try swapping: remove one, add two
        if not improved:
            for remove in list(current):
                temp = current - {remove}
                # Try to add two elements
                added = 0
                new_elements = []
                for candidate in range(m):
                    if candidate not in temp and added < 2:
                        if is_compatible(candidate, temp | set(new_elements)):
                            new_elements.append(candidate)
                            added += 1
                if added == 2:
                    current = temp | set(new_elements)
                    improved = True
                    break
        
        if improved:
            best_set = set(current)
    
    return np.array(sorted(best_set), dtype=np.int64)
