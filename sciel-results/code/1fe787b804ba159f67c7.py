import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s margin
    
    def is_valid_approx(s):
        s_list = list(s)
        n = len(s_list)
        if n <= 2:
            return True
        for i, a in enumerate(s_list):
            others = [x for j, x in enumerate(s_list) if j != i]
            m = len(others)
            # k=2
            seen = set()
            for x in others:
                if 2*a - x in seen:
                    return False
                seen.add(x)
            # k=3
            if m >= 3:
                target = 3*a
                others_sorted = sorted(others)
                for p in range(m):
                    for q in range(p+1, m):
                        rem = target - others_sorted[p] - others_sorted[q]
                        # Binary search for rem in others_sorted[q+1:]
                        lo, hi = q+1, m-1
                        found = False
                        while lo <= hi:
                            mid = (lo+hi)//2
                            if others_sorted[mid] == rem:
                                found = True
                                break
                            elif others_sorted[mid] < rem:
                                lo = mid+1
                            else:
                                hi = mid-1
                        if found:
                            return False
        return True
    
    def is_valid_full(s):
        s_list = list(s)
        n = len(s_list)
        if n <= 2:
            return True
        for i, a in enumerate(s_list):
            others = [x for j, x in enumerate(s_list) if j != i]
            m = len(others)
            for k in range(2, m+1):
                target = k*a
                if k == 2:
                    seen = set()
                    for x in others:
                        if target-x in seen:
                            return False
                        seen.add(x)
                elif k == m:
                    if sum(others) == target:
                        return False
                else:
                    for t in combinations(others, k):
                        if sum(t) == target:
                            return False
        return True
    
    # Greedy init
    selected = set()
    for x in range(1, 513):
        if time.time() > deadline - 5:
            break
        if is_valid_approx(selected | {x}):
            selected.add(x)
    
    best = set(selected)
    best_size = len(selected)
    
    random.seed(789)
    current = set(selected)
    
    while time.time() < deadline - 5:
        not_in = [x for x in range(1, 513) if x not in current]
        if not not_in:
            break
        x = random.choice(not_in)
        test = current | {x}
        if is_valid_approx(test):
            current = test
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
        else:
            if current:
                rem = random.choice(list(current))
                test2 = (current - {rem}) | {x}
                if is_valid_approx(test2):
                    current = test2
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
    
    # Full verify and fix
    result = set(best)
    for _ in range(500):
        if is_valid_full(result):
            break
        s_list = list(result)
        removed = False
        for i, a in enumerate(s_list):
            others = [x for j, x in enumerate(s_list) if j != i]
            m = len(others)
            for k in range(2, m+1):
                target = k*a
                if k == 2:
                    seen = set()
                    for x in others:
                        if target-x in seen:
                            result.discard(a)
                            removed = True
                            break
                        seen.add(x)
                    if removed:
                        break
                elif k == m:
                    if sum(others) == target:
                        result.discard(a)
                        removed = True
                        break
                else:
                    for t in combinations(others, k):
                        if sum(t) == target:
                            result.discard(a)
                            removed = True
                            break
                    if removed:
                        break
            if removed:
                break
        if not removed:
            break
    
    return sorted(result)
