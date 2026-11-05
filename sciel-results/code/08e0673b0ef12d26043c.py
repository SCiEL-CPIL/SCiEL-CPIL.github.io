import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build conflict graph
    conflict = [set() for _ in range(N)]
    self_conflict = [False] * N
    
    for i in range(N):
        a = i + 1
        for s in squares:
            b = s - a
            if 1 <= b <= N:
                j = b - 1
                if i == j:
                    self_conflict[i] = True
                else:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    
    degrees = [len(conflict[i]) for i in range(N)]
    
    def is_valid_addition(candidate, current_set):
        if self_conflict[candidate]:
            return False
        for neighbor in conflict[candidate]:
            if neighbor in current_set:
                return False
        return True
        
    def greedy_construct(order):
        s = set()
        for node in order:
            if self_conflict[node]:
                continue
            valid = True
            for neighbor in conflict[node]:
                if neighbor in s:
                    valid = False
                    break
            if valid:
                s.add(node)
        return s
        
    best_set = set()
    best_size = 0
    
    # 1. Degree-based greedy
    order_deg = sorted(range(N), key=lambda x: (self_conflict[x], degrees[x]))
    s_deg = greedy_construct(order_deg)
    best_set = s_deg
    best_size = len(s_deg)
    
    # 2. Randomized Greedy
    max_restarts = 2000
    for _ in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        random_order = list(range(N))
        random.shuffle(random_order)
        s = greedy_construct(random_order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # 3. Local Search (1-opt: remove one, add all valid)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_nodes = list(best_set)
        random.shuffle(current_nodes)
        
        for u in current_nodes:
            if time.time() > deadline - 5.0:
                break
            trial_set = set(best_set)
            trial_set.remove(u)
            
            # Add all valid
            for v in range(N):
                if v in trial_set:
                    continue
                if self_conflict[v]:
                    continue
                if is_valid_addition(v, trial_set):
                    trial_set.add(v)
                    
            if len(trial_set) > len(best_set):
                best_set = trial_set
                improved = True
                
    # 4. Optional: 2-opt if time permits
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try a limited number of random pairs
            nodes_list = list(best_set)
            random.shuffle(nodes_list)
            pairs_tried = 0
            max_pairs = 500
            for i in range(len(nodes_list)):
                if pairs_tried >= max_pairs:
                    break
                for j in range(i + 1, len(nodes_list)):
                    if pairs_tried >= max_pairs or time.time() > deadline - 1.0:
                        break
                    pairs_tried += 1
                    u = nodes_list[i]
                    v = nodes_list[j]
                    
                    trial_set = set(best_set)
                    trial_set.remove(u)
                    trial_set.remove(v)
                    
                    # Add all valid
                    for w in range(N):
                        if w in trial_set:
                            continue
                        if self_conflict[w]:
                            continue
                        if is_valid_addition(w, trial_set):
                            trial_set.add(w)
                            
                    if len(trial_set) > len(best_set):
                        best_set = trial_set
                        improved = True
                        break # Restart the outer loop? Or just continue?
                        # Breaking inner, but outer continues. 
                        # Actually, if we improved, we should probably restart the search from the new set.
                        # The `improved=True` will restart the while loop.
                        
    return list(best_set)
