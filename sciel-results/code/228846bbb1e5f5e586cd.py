import time
import itertools
import math
import random

def calculate_score(U):
    if not U:
        return 0.0
    if 0 not in U:
        return 0.0
    U_set = set(U)
    sum_set = {a + b for a in U_set for b in U_set}
    diff_set = {a - b for a in U_set for b in U_set}
    max_u = max(U_set)
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    return 1 + math.log(len(diff_set) / len(sum_set)) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_set = [0, 1, 3]
    best_score = calculate_score(best_set)
    
    # Local search
    current_set = set([0, 1, 4, 6])
    current_score = calculate_score(current_set)
    if current_score > best_score:
        best_set = list(current_set)
        best_score = current_score
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding an element
        max_u = max(current_set)
        # Search range for new elements
        search_range = range(0, max_u + 15)
        for x in search_range:
            if x in current_set:
                continue
            new_set = current_set | {x}
            new_score = calculate_score(new_set)
            if new_score > current_score + 1e-9:
                current_set = new_set
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_set = list(current_set)
                    best_score = current_score
                break # Restart search with new set
        
        if time.time() > deadline:
            break
            
        # Try removing an element (except 0)
        for x in list(current_set):
            if x == 0:
                continue
            new_set = current_set - {x}
            if not new_set:
                continue
            new_score = calculate_score(new_set)
            if new_score > current_score + 1e-9:
                current_set = new_set
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_set = list(current_set)
                    best_score = current_score
                break

        if time.time() > deadline:
            break

        # Try swapping an element
        # Pick a random element to remove and try to replace it
        if current_set and len(current_set) > 1:
            elements = [x for x in current_set if x != 0]
            if elements:
                x_remove = random.choice(elements)
                for x_add in range(0, max(current_set) + 15):
                    if x_add in current_set:
                        continue
                    new_set = (current_set - {x_remove}) | {x_add}
                    new_score = calculate_score(new_set)
                    if new_score > current_score + 1e-9:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_set = list(current_set)
                            best_score = current_score
                        break

    return best_set
