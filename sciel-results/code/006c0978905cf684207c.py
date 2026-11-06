import time
import random

def construct():
    start_time = time.time()
    deadline = 75.0
    
    n = 256
    elements = list(range(1, n + 1))
    
    # Build conflict adjacency sets
    conflict = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                conflict[i].add(j)
                conflict[j].add(i)
    
    def greedy(order):
        selected = []
        sel_set = set()
        for v in order:
            ok = True
            for u in selected:
                if conflict[v][u]:
                    ok = False
                    break
            if ok:
                selected.append(v)
                sel_set.add(v)
        return selected
    
    best_set = []
    best_size = 0
    
    # Multiple greedy orderings
    orderings = [
        elements[:],
        elements[::-1],
        sorted(elements, key=lambda x: len(conflict[x])),
        sorted(elements, key=lambda x: -len(conflict[x])),
        sorted(elements, key=lambda x: x / (1 + len(conflict[x]))),
        sorted(elements, key=lambda x: -x / (1 + len(conflict[x]))),
    ]
    
    for order in orderings:
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
    
    # Local search: try to add elements, then try swaps
    def local_search(s, max_rounds=50):
        s_set = set(s)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            # Phase 1: try to add any compatible element
            added_any = False
            for v in elements:
                if v not in s_set:
                    if not any(conflict[v][u] for u in s_set):
                        s_set.add(v)
                        added_any = True
                        break
            if added_any:
                continue
            # Phase 2: try swap (remove 1, add 2)
            swapped = False
            for v in list(s_set):
                if time.time() > deadline:
                    break
                s_set.remove(v)
                count = 0
                for w in elements:
                    if w not in s_set and w != v:
                        if not any(conflict[w][u] for u in s_set):
                            s_set.add(w)
                            count += 1
                            if count >= 2:
                                break
                if count >= 2:
                    swapped = True
                    break
                else:
                    # Revert: try to refill
                    for w in elements:
                        if w not in s_set:
                            if not any(conflict[w][u] for u in s_set):
                                s_set.add(w)
                                break
                    if len(s_set) < len(s):
                        # Failed, revert
                        for w in list(s_set):
                            if w not in set(s):
                                s_set.remove(w)
                        s_set.add(v)
            if not swapped:
                break
        return list(s_set)
    
    if time.time() < deadline:
        ls = local_search(best_set)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls
    
    # Random restarts
    random.seed(42)
    while time.time() < deadline:
        order = elements[:]
        random.shuffle(order)
        s = greedy(order)
        s_set = set(s)
        # Quick add phase
        for v in elements:
            if time.time() > deadline:
                break
            if v not in s_set and not any(conflict[v][u] for u in s_set):
                s_set.add(v)
        if len(s_set) > best_size:
            best_size = len(s_set)
            best_set = list(s_set)
    
    return sorted(best_set)
