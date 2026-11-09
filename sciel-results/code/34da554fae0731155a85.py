import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s out of 60s

    # Known valid solution: {66, ..., 100}
    # Size 35
    base_set = list(range(66, 101))
    best_size = len(base_set)
    best_set = base_set

    def is_valid(S):
        S = sorted(S)
        if len(S) == 0: return True
        
        # Size 1
        if 200 in S: return False
        
        # Size 2
        seen = set()
        for x in S:
            if 200 - x in seen:
                return False
            seen.add(x)
            
        # Size 3
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                rem = 200 - S[i] - S[j]
                if rem in S_set:
                    if rem != S[i] and rem != S[j]:
                        return False
        return True

    def find_conflict(S):
        S_list = sorted(S)
        S_set = set(S_list)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                rem = 200 - S_list[i] - S_list[j]
                if rem in S_set:
                    if rem != S_list[i] and rem != S_list[j]:
                        return (S_list[i], S_list[j], rem)
        return None

    def greedy_solve(pool, strategy):
        S = set(pool)
        while True:
            if time.time() > deadline:
                return None
            conf = find_conflict(S)
            if not conf:
                return S
            a, b, c = conf
            if strategy == 'min':
                S.remove(min(a, b, c))
            elif strategy == 'max':
                S.remove(max(a, b, c))
            elif strategy == 'random':
                S.remove(random.choice([a, b, c]))
            else:
                S.remove(a)

    # Try different pools and strategies
    pools = [
        list(range(60, 101)),
        list(range(55, 101)),
        list(range(51, 101))
    ]
    strategies = ['min', 'max', 'random']
    
    for pool in pools:
        for strat in strategies:
            if time.time() > deadline:
                break
            res = greedy_solve(pool, strat)
            if res:
                if len(res) > best_size:
                    best_size = len(res)
                    best_set = list(res)
                    
            # Also try to see if we can add elements to the result
            # e.g. if we have a valid set of 36, can we add one more?
            # This is a local search step.
            if res and len(res) >= 36:
                # Try to add elements from pool not in res
                candidates = [x for x in pool if x not in res]
                random.shuffle(candidates)
                for cand in candidates:
                    new_set = set(res)
                    new_set.add(cand)
                    if is_valid(new_set):
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = list(new_set)
                    if time.time() > deadline:
                        break

    return best_set
