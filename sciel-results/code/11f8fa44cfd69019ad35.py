import time
import random

def construct():
    deadline = time.time() + 140
    
    prescribed = [1, 2, 5, 11]
    
    def get_sums(arr):
        sums = set()
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                sums.add(arr[i] + arr[j])
        return sums
    
    initial_sums = get_sums(prescribed)
    best_set = prescribed[:]
    best_size = len(prescribed)
    
    def can_add(arr_set, sums, x):
        if x in arr_set:
            return False
        if 2 * x in sums:
            return False
        for y in arr_set:
            if x + y in sums:
                return False
        return True
    
    def greedy_build(deadline):
        nonlocal best_set, best_size
        arr = prescribed[:]
        sums = initial_sums.copy()
        arr_set = set(arr)
        
        while time.time() < deadline:
            if len(arr) >= 20:
                break
            
            valid = []
            for x in range(1, 285):
                if x not in arr_set and can_add(arr_set, sums, x):
                    valid.append(x)
            
            if not valid:
                break
            
            x = random.choice(valid)
            arr.append(x)
            arr_set.add(x)
            for y in list(arr_set):
                if y != x:
                    sums.add(x + y)
            sums.add(2 * x)
        
        if len(arr) > best_size:
            best_size = len(arr)
            best_set = arr[:]
    
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        greedy_build(deadline)
    
    return best_set
