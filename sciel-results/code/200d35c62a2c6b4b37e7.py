import math
import random

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute conflict graph
    adj = {v: set() for v in vertices}
    for i in range(len(vertices)):
        u = vertices[i]
        for j in range(i + 1, len(vertices)):
            v = vertices[j]
            if is_square(u + v):
                adj[u].add(v)
                adj[v].add(u)
    
    def can_add(v, S):
        if is_square(2 * v):
            return False
        for u in S:
            if u in adj[v]:
                return False
        return True

    def saturate(S):
        changed = True
        while changed:
            changed = False
            for v in vertices:
                if v not in S and can_add(v, S):
                    S.add(v)
                    changed = True
        return S

    def greedy_init_from_class(base_class, compatible_class):
        S = set()
        for v in base_class:
            if can_add(v, S):
                S.add(v)
        for v in compatible_class:
            if can_add(v, S):
                S.add(v)
        for v in vertices:
            if can_add(v, S):
                S.add(v)
        return saturate(S)

    R0 = [i for i in vertices if i % 4 == 0]
    R1 = [i for i in vertices if i % 4 == 1]
    R2 = [i for i in vertices if i % 4 == 2]
    R3 = [i for i in vertices if i % 4 == 3]
    
    cand1 = greedy_init_from_class(R1, R2)
    cand2 = greedy_init_from_class(R3, R0)
    
    best_S = cand1 if len(cand1) >= len(cand2) else cand2
    
    improved = True
    max_iter = 50
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        current_S = set(best_S)
        saturate(current_S)
        if len(current_S) > len(best_S):
            best_S = current_S
            improved = True
            continue
            
        found_swap = False
        S_list = list(best_S)
        random.shuffle(S_list)
        
        for u in S_list:
            new_S = set(best_S)
            new_S.discard(u)
            
            addable = [v for v in vertices if v not in new_S and can_add(v, new_S)]
            
            if len(addable) >= 2:
                best_pair = None
                for i in range(len(addable)):
                    a = addable[i]
                    for j in range(i + 1, len(addable)):
                        b = addable[j]
                        if b not in adj[a]:
                            best_pair = (a, b)
                            break
                    if best_pair:
                        break
                
                if best_pair:
                    a, b = best_pair
                    new_S.add(a)
                    new_S.add(b)
                    if len(new_S) > len(best_S):
                        best_S = new_S
                        improved = True
                        found_swap = True
                        break
        
        if found_swap:
            continue
            
        if iter_count % 5 == 0:
            found_swap2 = False
            S_list2 = list(best_S)
            random.shuffle(S_list2)
            
            max_pairs = 100
            pairs_checked = 0
            
            for i in range(len(S_list2)):
                u = S_list2[i]
                for j in range(i + 1, len(S_list2)):
                    v = S_list2[j]
                    if pairs_checked >= max_pairs:
                        break
                    pairs_checked += 1
                    
                    new_S = set(best_S)
                    new_S.discard(u)
                    new_S.discard(v)
                    
                    addable = [x for x in vertices if x not in new_S and can_add(x, new_S)]
                    
                    if len(addable) >= 3:
                        best_triple = None
                        addable_copy = addable[:]
                        random.shuffle(addable_copy)
                        
                        for a in addable_copy:
                            for b in addable_copy:
                                if b == a: continue
                                if b in adj[a]: continue
                                for c in addable_copy:
                                    if c == a or c == b: continue
                                    if c in adj[a] or c in adj[b]: continue
                                    best_triple = (a, b, c)
                                    break
                                if best_triple:
                                    break
                            if best_triple:
                                break
                            
                        if best_triple:
                            a, b, c = best_triple
                            new_S.add(a)
                            new_S.add(b)
                            new_S.add(c)
                            if len(new_S) > len(best_S):
                                best_S = new_S
                                improved = True
                                found_swap2 = True
                                break
                if found_swap2:
                    break
            
            if found_swap2:
                continue

    return list(best_S)
