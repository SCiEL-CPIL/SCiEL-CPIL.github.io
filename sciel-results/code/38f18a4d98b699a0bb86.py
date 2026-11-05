import time
from itertools import combinations

_START = time.time()
_DEADLINE = 85.0

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct():
    LCM = 232792560
    weights = [LCM // k for k in range(1, 23)]
    n = 22
    
    def is_dissociated(indices):
        if not indices:
            return True
        sums = {0}
        for i in indices:
            w = weights[i]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True
    
    def greedy(order):
        chosen = []
        sums = {0}
        for i in order:
            w = weights[i]
            valid = True
            for s in sums:
                if (s + w) in sums:
                    valid = False
                    break
            if valid:
                chosen.append(i)
                sums.update({s + w for s in sums})
        return chosen
    
    # Try multiple greedy orderings to get a good initial solution
    best = []
    orders = [
        list(range(22)),           # k ascending
        list(range(21, -1, -1)),  # k descending
        sorted(range(22), key=lambda i: -weights[i]),  # weight descending
        sorted(range(22), key=lambda i: weights[i]),   # weight ascending
    ]
    
    for order in orders:
        cand = greedy(order)
        if len(cand) > len(best):
            best = cand
    
    # Backtracking with improved pruning
    def backtrack(start, current, current_sums, best_ref):
        if not _time_ok():
            return
        if len(current) > len(best_ref):
            best_ref[:] = current[:]
        
        if len(current) + (n - start) <= len(best_ref):
            return
        
        for i in range(start, n):
            if not _time_ok():
                return
            w = weights[i]
            valid = True
            new_sums = set()
            for s in current_sums:
                ns = s + w
                if ns in current_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if valid:
                new_total_sums = current_sums | new_sums
                current.append(i)
                backtrack(i + 1, current, new_total_sums, best_ref)
                current.pop()
    
    backtrack(0, [], {0}, best)
    
    # Local search: try to add elements not in best
    if _time_ok():
        current_set = set(best)
        sums = {0}
        for i in best:
            w = weights[i]
            sums.update({s + w for s in sums})
        
        for i in range(n):
            if i in current_set:
                continue
            w = weights[i]
            valid = True
            for s in sums:
                if (s + w) in sums:
                    valid = False
                    break
            if valid:
                best.append(i)
                current_set.add(i)
                sums.update({s + w for s in sums})
    
    if not is_dissociated(best):
        best = [0, 1, 3, 7]
    
    return [i + 1 for i in best]
