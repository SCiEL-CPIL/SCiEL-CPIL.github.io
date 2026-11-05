import time
import random

_DEADLINE = 85.0
_N = 256

def _is_good_pair(a, b):
    return (a * b) % (a + b) != 0

def _build_conflict_graph():
    conflict = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a].add(b)
                conflict[b].add(a)
    return conflict

def _validate(S):
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def _greedy_build(conflict, order):
    S = set()
    for x in order:
        if not (S & conflict[x]):
            S.add(x)
    return S

def _local_search(conflict, S, time_limit):
    S = set(S)
    best = set(S)
    best_size = len(S)
    start = time.time()
    
    while time.time() - start < time_limit:
        excluded = [x for x in range(1, _N + 1) if x not in S]
        if not excluded:
            break
        random.shuffle(excluded)
        
        improved = False
        # Try additions
        for x in excluded:
            if not (S & conflict[x]):
                S.add(x)
                improved = True
                break
        if improved:
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
            continue
        
        # Try 1-for-1 swaps
        for _ in range(300):
            if time.time() - start > time_limit:
                break
            x = random.choice(excluded)
            conflicts = S & conflict[x]
            if not conflicts:
                S.add(x)
                improved = True
                if len(S) > best_size:
                    best = set(S)
                    best_size = len(S)
                break
            if len(conflicts) <= 2:
                for y in conflicts:
                    S.remove(y)
                    if not (S & conflict[x]):
                        S.add(x)
                        improved = True
                        if len(S) > best_size:
                            best = set(S)
                            best_size = len(S)
                        break
                    else:
                        S.add(y)
        if improved:
            continue
        
        # Try 2-for-2 swaps
        for _ in range(100):
            if time.time() - start > time_limit:
                break
            x1, x2 = random.sample(excluded, 2)
            c1 = S & conflict[x1]
            c2 = S & conflict[x2]
            if not c1 and not c2:
                S.add(x1)
                S.add(x2)
                improved = True
                if len(S) > best_size:
                    best = set(S)
                    best_size = len(S)
                break
            if c1 and c2 and len(c1) <= 2 and len(c2) <= 2:
                for y1 in c1:
                    for y2 in c2:
                        if y1 == y2:
                            S.discard(y1)
                            if not (S & conflict[x1]) and not (S & conflict[x2]):
                                S.add(x1)
                                S.add(x2)
                                improved = True
                                if len(S) > best_size:
                                    best = set(S)
                                    best_size = len(S)
                            else:
                                S.add(y1)
                            break
                        else:
                            S.discard(y1)
                            S.discard(y2)
                            if not (S & conflict[x1]) and not (S & conflict[x2]):
                                S.add(x1)
                                S.add(x2)
                                improved = True
                                if len(S) > best_size:
                                    best = set(S)
                                    best_size = len(S)
                            else:
                                S.add(y1)
                                S.add(y2)
                            break
                    if improved:
                        break
        if improved:
            continue
        
        break
    
    return best

def construct():
    start = time.time()
    conflict = _build_conflict_graph()
    
    # Start with all odds (known valid, size 128)
    S = set(range(1, _N + 1, 2))
    
    # Try to add evens
    evens = list(range(2, _N + 1, 2))
    random.shuffle(evens)
    for e in evens:
        if time.time() - start > _DEADLINE - 5:
            break
        if not (S & conflict[e]):
            S.add(e)
    
    # Local search
    remaining = _DEADLINE - 5 - (time.time() - start)
    if remaining > 1:
        S = _local_search(conflict, S, remaining)
    
    result = sorted(S)
    if not result:
        return [1]
    return result
