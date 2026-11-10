import time
import random
from itertools import combinations

def is_sidon_subset(subset):
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def get_counts(ambient):
    """
    Returns (count_5, count_4)
    count_5: number of Sidon subsets of size 5
    count_4: number of Sidon subsets of size 4
    """
    c5 = 0
    c4 = 0
    for sub in combinations(ambient, 5):
        if is_sidon_subset(sub):
            c5 += 1
    for sub in combinations(ambient, 4):
        if is_sidon_subset(sub):
            c4 += 1
    return c5, c4

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_score = (float('inf'), float('inf'))
    
    current_set = random.sample(range(129), 12)
    current_score = get_counts(current_set)
    
    if current_score < best_score:
        best_score = current_score
        best_set = current_set[:]
        
    no_improve_iterations = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.8:
            # Single swap
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current_set:
                continue
            old_val = current_set[i]
            current_set[i] = new_val
            
            new_score = get_counts(current_set)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_iterations = 0
                if new_score < best_score:
                    best_score = new_score
                    best_set = current_set[:]
            else:
                current_set[i] = old_val
                no_improve_iterations += 1
        else:
            # Double swap
            i, j = random.sample(range(12), 2)
            new_val1 = random.randint(0, 128)
            new_val2 = random.randint(0, 128)
            
            if new_val1 == new_val2:
                continue
            if new_val1 in current_set or new_val2 in current_set:
                continue
                
            old_val1 = current_set[i]
            old_val2 = current_set[j]
            current_set[i] = new_val1
            current_set[j] = new_val2
            
            new_score = get_counts(current_set)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_iterations = 0
                if new_score < best_score:
                    best_score = new_score
                    best_set = current_set[:]
            else:
                current_set[i] = old_val1
                current_set[j] = old_val2
                no_improve_iterations += 1
        
        if no_improve_iterations > max_no_improve:
            current_set = random.sample(range(129), 12)
            current_score = get_counts(current_set)
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
            no_improve_iterations = 0

    return best_set
