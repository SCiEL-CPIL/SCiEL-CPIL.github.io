import time
import random

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0
    
    def check_and_add(chosen, counts, x):
        """Check if x can be added. Return True if added."""
        for y in chosen:
            if counts[x + y] + 1 > 2:
                return False
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)
        return True
        
    def remove_from_set(chosen, counts, x):
        """Remove x from chosen set."""
        idx = chosen.index(x)
        for i, y in enumerate(chosen):
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen.pop(idx)

    def run_greedy(order='random'):
        if order == 'random':
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        
        chosen = []
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            check_and_add(chosen, counts, x)
                
        return chosen, counts
    
    def local_search(chosen, counts):
        """
        Efficient local search:
        1. Try to add any element that is currently valid.
        2. If no element can be added, try to remove one element and see if 
           that allows adding another.
        """
        chosen_set = set(chosen)
        improved = True
        max_iters = 100
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add an element
            candidates = [x for x in range(1, N + 1) if x not in chosen_set]
            if not candidates:
                break
            random.shuffle(candidates)
            
            # Check first 50 candidates to save time
            for x in candidates[:50]:
                can_add = True
                for y in chosen:
                    if counts[x + y] + 1 > 2:
                        can_add = False
                        break
                if can_add and counts[2 * x] + 1 > 2:
                    can_add = False
                if can_add:
                    check_and_add(chosen, counts, x)
                    chosen_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. If no add, try a swap
            if chosen:
                x_to_remove = random.choice(chosen)
                remove_from_set(chosen, counts, x_to_remove)
                chosen_set.discard(x_to_remove)
                
                candidates = [x for x in range(1, N + 1) if x not in chosen_set]
                if candidates:
                    random.shuffle(candidates)
                    added_after_remove = False
                    for x in candidates[:50]:
                        can_add = True
                        for y in chosen:
                            if counts[x + y] + 1 > 2:
                                can_add = False
                                break
                        if can_add and counts[2 * x] + 1 > 2:
                            can_add = False
                        if can_add:
                            check_and_add(chosen, counts, x)
                            chosen_set.add(x)
                            added_after_remove = True
                            improved = True
                            break
                            
                    if not added_after_remove:
                        check_and_add(chosen, counts, x_to_remove)
                        chosen_set.add(x_to_remove)

        return chosen, counts

    start_time = time.time()
    timeout = 115 # Safety margin
    
    orders = ['random', 'ascending', 'descending']
    
    while time.time() - start_time < timeout:
        order = random.choice(orders)
        current_set, current_counts = run_greedy(order)
        current_set, current_counts = local_search(current_set, current_counts)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    return best_set
