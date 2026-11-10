import random
import time
from itertools import combinations

def construct():
    N = 192
    MAX_SIZE = 48
    
    def is_valid_addition(S, x):
        if len(S) >= 3:
            for t in combinations(S, 3):
                if sum(t) % x == 0:
                    return False
        if len(S) >= 2:
            for a in S:
                others = [v for v in S if v != a]
                if len(others) >= 2:
                    for b, c in combinations(others, 2):
                        if (b + c + x) % a == 0:
                            return False
        return True
    
    def is_valid(S):
        S_list = list(S)
        for a in S_list:
            others = [v for v in S_list if v != a]
            if len(others) >= 3:
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        return False
        return True
    
    start_time = time.time()
    time_limit = 58.0
    
    current_set = set(range(1, N + 1))
    removed = []
    
    while True:
        conflict = None
        S_list = list(current_set)
        for a in S_list:
            others = [v for v in S_list if v != a]
            if len(others) >= 3:
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        conflict = (a, t)
                        break
            if conflict:
                break
        if conflict is None:
            break
        a, t = conflict
        to_remove = min([a] + list(t))
        current_set.remove(to_remove)
        removed.append(to_remove)
        if time.time() - start_time > time_limit:
            break
    
    removed.sort(reverse=True)
    best_set = list(current_set)
    best_size = len(best_set)
    
    for x in removed:
        if x in current_set or len(current_set) >= MAX_SIZE:
            continue
        if is_valid_addition(list(current_set), x):
            current_set.add(x)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
        if time.time() - start_time > time_limit:
            break
    
    candidates = [x for x in range(1, N + 1) if x not in current_set]
    random.shuffle(candidates)
    
    while time.time() - start_time < time_limit:
        if len(current_set) < MAX_SIZE and candidates:
            added = False
            for x in candidates:
                if is_valid_addition(list(current_set), x):
                    current_set.add(x)
                    candidates.remove(x)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    added = True
                    break
            if added:
                continue
        if len(current_set) > 0 and candidates:
            u = random.choice(list(current_set))
            for v in candidates:
                new_set = current_set - {u}
                if is_valid_addition(list(new_set), v):
                    current_set = new_set | {v}
                    candidates.remove(v)
                    candidates.append(u)
                    break
    
    return best_set

# EVOLVE-BLOCK-END
