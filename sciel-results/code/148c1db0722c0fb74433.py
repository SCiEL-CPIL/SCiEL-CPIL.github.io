import random
import time
from itertools import combinations

def construct():
    n = 192
    max_elements = 48
    
    def is_valid_addition(selected, candidate):
        if len(selected) >= 3:
            for a, b, c in combinations(selected, 3):
                if (a + b + c) % candidate == 0:
                    return False
        for a in selected:
            others = [x for x in selected if x != a]
            if len(others) >= 2:
                for b, c in combinations(others, 2):
                    if (b + c + candidate) % a == 0:
                        return False
        return True

    def is_valid(selected):
        for a in selected:
            others = [x for x in selected if x != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    def greedy_large_first():
        selected = []
        for x in range(n, 0, -1):
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= max_elements:
                    break
        return selected

    def local_search(selected):
        selected = list(selected)
        start_time = time.time()
        max_time = 50.0
        
        current_set = set(selected)
        best_set = set(selected)
        best_size = len(selected)
        all_nums = list(range(1, n + 1))
        
        stuck_counter = 0
        
        while len(current_set) < max_elements and (time.time() - start_time) < max_time:
            improved = False
            
            # 1. Try to add a valid element
            if len(current_set) < max_elements:
                candidates_to_add = [x for x in all_nums if x not in current_set]
                random.shuffle(candidates_to_add)
                # Try a few random candidates, then largest?
                # Let's try largest valid to maintain "safety"
                candidates_to_add.sort(reverse=True)
                for x in candidates_to_add:
                    if is_valid_addition(list(current_set), x):
                        current_set.add(x)
                        improved = True
                        stuck_counter = 0
                        if len(current_set) > best_size:
                            best_set = set(current_set)
                            best_size = len(current_set)
                        break
            
            # 2. If no addition, try a random swap
            if not improved:
                stuck_counter += 1
                if stuck_counter > 100:
                    break
                
                # Pick random i to remove
                i = random.choice(list(current_set))
                temp_set = set(current_set)
                temp_set.remove(i)
                
                # Pick random j to add
                candidates_to_add = [x for x in all_nums if x not in temp_set]
                j = random.choice(candidates_to_add)
                
                if is_valid_addition(list(temp_set), j):
                    current_set = temp_set
                    current_set.add(j)
                    # Don't reset stuck_counter on swap? 
                    # Swaps are exploration. If we keep swapping without growing, we might loop.
                    # But we break after 100 stuck steps.

        return list(best_set)

    best = greedy_large_first()
    
    for trial in range(10):
        random.seed(4000 + trial)
        selected = []
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_addition(selected, x):
                selected.append(x)
                if len(selected) >= max_elements:
                    break
        
        s = local_search(selected)
        if len(s) > len(best):
            best = s
            if len(best) >= max_elements:
                break
    
    if not is_valid(best):
        best = greedy_large_first()
    
    return best[:max_elements]
