import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    fixed = [1, 2, 5, 11]
    current_set = set(fixed)
    diffs = set()
    for i in range(len(fixed)):
        for j in range(i+1, len(fixed)):
            diffs.add(abs(fixed[i] - fixed[j]))
            
    candidates = [x for x in range(1, 285) if x not in current_set]
    
    best_size = len(fixed)
    best_set = list(fixed)
    
    def search(current_size, target_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return False
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            
        if current_size == target_size:
            return True
            
        valid = []
        for x in candidates:
            if x in current_set:
                continue
            conflict = False
            for y in current_set:
                d = abs(x - y)
                if d in diffs:
                    conflict = True
                    break
            if not conflict:
                valid.append(x)
                
        if not valid:
            return False
            
        random.shuffle(valid)
        
        for x in valid:
            # Add x
            current_set.add(x)
            new_diffs_to_add = []
            for y in current_set:
                if y != x:
                    d = abs(x - y)
                    diffs.add(d)
                    new_diffs_to_add.append(d)
                    
            if search(current_size + 1, target_size):
                return True
                
            # Undo
            current_set.remove(x)
            for d in new_diffs_to_add:
                diffs.remove(d)
                
        return False

    search(len(fixed), 20)
    
    return best_set
