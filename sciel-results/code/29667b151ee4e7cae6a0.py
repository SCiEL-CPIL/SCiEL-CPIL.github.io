import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    N = 32
    MAX_VAL = 512
    
    def get_score(lst, present):
        diffs = set()
        n = len(lst)
        # Iterate pairs
        for i in range(n):
            ai = lst[i]
            for j in range(i+1, n):
                aj = lst[j]
                target = 2 * aj - ai
                if target <= MAX_VAL and present[target]:
                    diffs.add(aj - ai)
        return len(diffs)
        
    def random_set():
        return set(random.sample(range(1, MAX_VAL + 1), N))
        
    best_set = set()
    best_score = 0
    
    # 1. Find a good starting point
    # Try a few random sets and the interval
    candidates = []
    candidates.append(set(range(1, N+1)))
    for _ in range(20):
        candidates.append(random_set())
        
    for s in candidates:
        lst = sorted(s)
        present = [0] * (MAX_VAL + 1)
        for x in s:
            present[x] = 1
        sc = get_score(lst, present)
        if sc > best_score:
            best_score = sc
            best_set = s
            
    # 2. Local Search
    current_set = best_set
    current_score = best_score
    current_lst = sorted(current_set)
    current_present = [0] * (MAX_VAL + 1)
    for x in current_set:
        current_present[x] = 1
        
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Generate move
        idx_remove = random.randint(0, N - 1)
        elem_remove = current_lst[idx_remove]
        
        elem_add = random.randint(1, MAX_VAL)
        if elem_add in current_set:
            continue
            
        # Apply move
        new_set = set(current_set)
        new_set.remove(elem_remove)
        new_set.add(elem_add)
        
        new_lst = sorted(new_set)
        # Update present array efficiently? 
        # It's small, just rebuild or update in place?
        # Rebuilding is 512 ops. Updating is 2 ops.
        # Let's update in place to save time.
        # But we need a 'new_present' to evaluate the new state.
        # If we modify current_present, we lose the old state if we need to rollback.
        # Since we only keep the best, we can just modify and if we don't accept, rollback.
        
        # Modify current_present
        current_present[elem_remove] = 0
        current_present[elem_add] = 1
        
        new_score = get_score(new_lst, current_present)
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            current_lst = new_lst
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_set = current_set
        else:
            # Rollback
            current_present[elem_remove] = 1
            current_present[elem_add] = 0
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Restart logic
            # Try a new random set
            s = random_set()
            lst = sorted(s)
            temp_present = [0] * (MAX_VAL + 1)
            for x in s:
                temp_present[x] = 1
            sc = get_score(lst, temp_present)
            
            if sc > current_score:
                current_set = s
                current_score = sc
                current_lst = lst
                current_present = temp_present
            else:
                # If not better, just reset counter
                pass
            no_improve_count = 0

    return list(best_set)
