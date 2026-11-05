import time
import random
from collections import Counter

TIME_LIMIT = 80.0
START_TIME = 0.0

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_set = set()
    best_size = 0
    
    def can_add(S_list, x):
        n = len(S_list)
        
        # Check 1: x | (b+c+d) for some triple from S
        if n >= 3:
            rems = [s % x for s in S_list]
            for i in range(n):
                ri = rems[i]
                for j in range(i+1, n):
                    rj = rems[j]
                    for k in range(j+1, n):
                        if (ri + rj + rems[k]) % x == 0:
                            return False
        
        # Check 2: for each a in S, a | (x+b+c) for some b,c in S\{a}
        for i, a in enumerate(S_list):
            if n < 3:
                continue
            target = (-x) % a
            others_rems = []
            for j in range(n):
                if j != i:
                    others_rems.append(S_list[j] % a)
            counts = Counter(others_rems)
            for r, cnt in counts.items():
                need = (target - r) % a
                if need == r:
                    if cnt >= 2:
                        return False
                else:
                    if need in counts:
                        return False
        
        return True
    
    while time.time() - START_TIME < TIME_LIMIT:
        current_set = set()
        current_list = []
        
        # Phase 1: Add large numbers greedily
        large_numbers = list(range(97, 193))
        random.shuffle(large_numbers)
        for x in large_numbers:
            if can_add(current_list, x):
                current_set.add(x)
                current_list.append(x)
        
        # Phase 2: Add smaller numbers greedily
        small_numbers = [x for x in range(1, 97) if x not in current_set]
        random.shuffle(small_numbers)
        for x in small_numbers:
            if can_add(current_list, x):
                current_set.add(x)
                current_list.append(x)
        
        # Phase 3: Local search - try to swap one out for two in
        improved = True
        while improved and time.time() - START_TIME < TIME_LIMIT:
            improved = False
            for i in range(len(current_list)):
                if time.time() - START_TIME >= TIME_LIMIT:
                    break
                removed = current_list[i]
                temp_set = set(current_list)
                temp_set.remove(removed)
                temp_list = [x for x in current_list if x != removed]
                
                remaining = [x for x in range(1, 193) if x not in temp_set]
                random.shuffle(remaining)
                
                added_count = 0
                for x in remaining:
                    if can_add(temp_list, x):
                        temp_set.add(x)
                        temp_list.append(x)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 2:
                    current_set = temp_set
                    current_list = temp_list
                    improved = True
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    return list(best_set)
