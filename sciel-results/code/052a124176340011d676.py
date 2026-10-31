import time
import random
import math

def calculate_score(U):
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a+b)
            diffset.add(a-b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 5],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 4, 8],
        [0, 1, 2, 4, 7],
        [0, 1, 2, 5, 8]
    ]
    
    best_score = 0
    best_set = [0, 1, 3]
    
    for c in candidates:
        s = calculate_score(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    top_sets = {tuple(sorted(best_set)): best_score}
    
    # Add other candidates to top_sets if they are good
    for c in candidates:
        s = calculate_score(c)
        top_sets[tuple(sorted(c))] = s
        
    # Keep top 20
    sorted_items = sorted(top_sets.items(), key=lambda x: x[1], reverse=True)
    top_sets = dict(sorted_items[:20])
    
    while time.time() < deadline:
        current_set = random.choice(list(top_sets.keys()))
        current_set = list(current_set)
        
        action = random.choice(['add', 'remove', 'replace'])
        
        if action == 'add':
            new_elem = random.randint(0, 100)
            if new_elem not in current_set:
                new_set = current_set + [new_elem]
            else:
                continue
        elif action == 'remove':
            if len(current_set) <= 2:
                continue
            idx = random.randint(0, len(current_set)-1)
            if current_set[idx] == 0:
                continue
            new_set = current_set[:idx] + current_set[idx+1:]
        else:
            if len(current_set) < 2:
                continue
            idx = random.randint(0, len(current_set)-1)
            if current_set[idx] == 0:
                continue
            new_elem = random.randint(0, 100)
            if new_elem in current_set:
                continue
            new_set = current_set[:]
            new_set[idx] = new_elem
            
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            continue
            
        new_score = calculate_score(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best_set = new_set
            
        top_sets[tuple(new_set)] = new_score
        
        if len(top_sets) > 50:
            sorted_items = sorted(top_sets.items(), key=lambda x: x[1], reverse=True)
            top_sets = dict(sorted_items[:50])
            
    return best_set
