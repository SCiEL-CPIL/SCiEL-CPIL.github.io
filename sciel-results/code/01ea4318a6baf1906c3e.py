import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    best_set = None
    best_size = float('inf')
    
    candidates = list(range(1, N + 1))
    
    while time.time() < deadline:
        random.shuffle(candidates)
        
        selected = []
        used_sums = set()
        
        for x in candidates:
            # Check if x can be added without conflict
            # New sums: 2*x and x+y for y in selected
            conflict = False
            if 2 * x in used_sums:
                conflict = True
            else:
                for y in selected:
                    if x + y in used_sums:
                        conflict = True
                        break
            
            if not conflict:
                # Add x to selected and update sums
                used_sums.add(2 * x)
                for y in selected:
                    used_sums.add(x + y)
                selected.append(x)
                
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = list(selected)
            
    if best_set is None:
        # Fallback
        return [1, 2, 4, 8, 16, 32, 64, 128]
        
    return best_set
