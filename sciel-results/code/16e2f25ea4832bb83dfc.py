import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    all_vals = list(range(1, max_val + 1))
    
    def init_state(S):
        """Initialize sum set and sum-to-pair mapping."""
        Sums = set()
        sum_map = {}  # sum -> (a, b)
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                Sums.add(s)
                sum_map[s] = (S[i], S[j])
        return Sums, sum_map
    
    def can_add(x, S, Sums):
        """Check if adding x creates duplicate sums."""
        if 2 * x in Sums:
            return False
        for y in S:
            if x + y in Sums:
                return False
        return True
    
    def add_element(x, S, Sums, sum_map):
        Sums.add(2 * x)
        sum_map[2 * x] = (x, x)
        for y in S:
            Sums.add(x + y)
            sum_map[x + y] = (x, y)
        S.append(x)
        S.sort()
    
    def remove_element(x, S, Sums, sum_map):
        S.remove(x)
        Sums.discard(2 * x)
        if 2 * x in sum_map and sum_map[2 * x] == (x, x):
            del sum_map[2 * x]
        for y in S:
            s = x + y
            Sums.discard(s)
            if s in sum_map and sum_map[s] in ((x, y), (y, x)):
                del sum_map[s]

    def try_add(x, S, Sums, sum_map):
        """Try to add x; return True if successful."""
        if 2 * x in Sums:
            return False
        for y in S:
            if x + y in Sums:
                return False
        add_element(x, S, Sums, sum_map)
        return True
    
    def greedy_fill(S, Sums, sum_map, order='ascending'):
        """Fill S greedily."""
        S_set = set(S)
        if order == 'ascending':
            candidates = list(range(1, max_val + 1))
        elif order == 'descending':
            candidates = list(range(max_val, 0, -1))
        else:
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
        
        for x in candidates:
            if x in S_set:
                continue
            if can_add(x, S, Sums):
                add_element(x, S, Sums, sum_map)
                S_set.add(x)
        return S, Sums, sum_map
    
    def local_search(S, Sums, sum_map, max_iters=300, aggressive=False):
        """Local search with adds, swaps, and double swaps."""
        S_set = set(S)
        improved = False
        iters = 0
        stuck_count = 0
        
        while iters < max_iters and time.time() < deadline:
            iters += 1
            
            # Try to add an element
            candidates = [x for x in all_vals if x not in S_set]
            if not candidates:
                break
            
            random.shuffle(candidates)
            
            added = False
            for x in candidates[:min(80, len(candidates))]:
                if try_add(x, S, Sums, sum_map):
                    S_set.add(x)
                    improved = True
                    added = True
                    stuck_count = 0
                    break
            
            if added:
                continue
            
            stuck_count += 1
            
            # If stuck, try swaps
            removable = [x for x in S if x not in prescribed_set]
            if not removable:
                break
            
            random.shuffle(removable)
            swap_done = False
            
            # Try single swaps
            for r in removable[:min(20, len(removable))]:
                remove_element(r, S, Sums, sum_map)
                S_set.discard(r)
                
                candidates = [x for x in all_vals if x not in S_set]
                random.shuffle(candidates)
                
                for x in candidates[:min(80, len(candidates))]:
                    if try_add(x, S, Sums, sum_map):
                        S_set.add(x)
                        swap_done = True
                        improved = True
                        stuck_count = 0
                        break
                
                if swap_done:
                    break
                else:
                    # Restore r
                    add_element(r, S, Sums, sum_map)
                    S_set.add(r)
            
            if swap_done:
                continue
            
            # Try double swaps if aggressive
            if aggressive and len(removable) >= 2:
                random.shuffle(removable)
                r1, r2 = removable[0], removable[1]
                
                remove_element(r1, S, Sums, sum_map)
                S_set.discard(r1)
                remove_element(r2, S, Sums, sum_map)
                S_set.discard(r2)
                
                candidates = [x for x in all_vals if x not in S_set]
                random.shuffle(candidates)
                
                added_count = 0
                added_elems = []
                for x in candidates[:min(120, len(candidates))]:
                    if try_add(x, S, Sums, sum_map):
                        S_set.add(x)
                        added_elems.append(x)
                        added_count += 1
                        if added_count >= 3:
                            break
                
                if added_count >= 2:
                    improved = True
                    stuck_count = 0
                else:
                    for x in added_elems:
                        remove_element(x, S, Sums, sum_map)
                        S_set.discard(x)
                    add_element(r1, S, Sums, sum_map)
                    S_set.add(r1)
                    add_element(r2, S, Sums, sum_map)
                    S_set.add(r2)
            
            if stuck_count > 80:
                break
        
        return S, Sums, sum_map, improved
    
    def hill_climbing_polish(S, Sums, sum_map):
        """Final polish: try all possible single moves to improve."""
        S_set = set(S)
        improved = True
        polish_iters = 0
        
        while improved and polish_iters < 200 and time.time() < deadline:
            improved = False
            polish_iters += 1
            
            # Try all adds
            for x in all_vals:
                if x in S_set:
                    continue
                if try_add(x, S, Sums, sum_map):
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try all single swaps
            removable = [x for x in S if x not in prescribed_set]
            for r in removable:
                remove_element(r, S, Sums, sum_map)
                S_set.discard(r)
                
                for x in all_vals:
                    if x in S_set:
                        continue
                    if try_add(x, S, Sums, sum_map):
                        S_set.add(x)
                        improved = True
                        break
                
                if improved:
                    break
                else:
                    add_element(r, S, Sums, sum_map)
                    S_set.add(r)
        
        return S, Sums, sum_map
    
    def exhaustive_double_swap_polish(S, Sums, sum_map):
        """Try all pairs of removals and greedy additions."""
        S_set = set(S)
        improved = False
        removable = [x for x in S if x not in prescribed_set]
        
        for i in range(len(removable)):
            for j in range(i + 1, len(removable)):
                if time.time() > deadline - 5:
                    return S, Sums, sum_map, improved
                r1, r2 = removable[i], removable[j]
                
                remove_element(r1, S, Sums, sum_map)
                S_set.discard(r1)
                remove_element(r2, S, Sums, sum_map)
                S_set.discard(r2)
                
                candidates = [x for x in all_vals if x not in S_set]
                added_elems = []
                for x in candidates:
                    if try_add(x, S, Sums, sum_map):
                        S_set.add(x)
                        added_elems.append(x)
                
                if len(added_elems) >= 2:
                    improved = True
                    return S, Sums, sum_map, improved
                else:
                    for x in added_elems:
                        remove_element(x, S, Sums, sum_map)
                        S_set.discard(x)
                    add_element(r1, S, Sums, sum_map)
                    S_set.add(r1)
                    add_element(r2, S, Sums, sum_map)
                    S_set.add(r2)
        
        return S, Sums, sum_map, improved
    
    best_size = 0
    best_S = []
    
    restart_count = 0
    
    while time.time() < deadline - 20:
        restart_count += 1
        
        if restart_count % 3 == 1:
            order = 'ascending'
        elif restart_count % 3 == 2:
            order = 'descending'
        else:
            order = 'random'
        
        S = list(prescribed)
        S.sort()
        Sums, sum_map = init_state(S)
        S, Sums, sum_map = greedy_fill(S, Sums, sum_map, order=order)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        if best_size >= 20:
            break
        
        aggressive = restart_count % 2 == 0
        S, Sums, sum_map, improved = local_search(S, Sums, sum_map, max_iters=300, aggressive=aggressive)
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        if best_size >= 20:
            break
    
    # Final polish on best solution
    if best_S and time.time() < deadline - 15:
        S = best_S.copy()
        Sums, sum_map = init_state(S)
        S, Sums, sum_map = hill_climbing_polish(S, Sums, sum_map)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    # Exhaustive double-swap polish if time remains
    if best_S and time.time() < deadline - 5:
        S = best_S.copy()
        Sums, sum_map = init_state(S)
        S, Sums, sum_map, improved = exhaustive_double_swap_polish(S, Sums, sum_map)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    return best_S
