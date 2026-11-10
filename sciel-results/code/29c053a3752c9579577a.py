import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 180
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, 2 * a):
            diff = b - a
            if (a * b) % diff == 0:
                c = (a * b) // diff
                if c > b and c <= N:
                    triples.append((a, b, c))
    
    triples_by_elem = {i: [] for i in range(1, N + 1)}
    for t in triples:
        for x in t:
            triples_by_elem[x].append(t)
            
    def greedy_hit_set(triples, N, deadline, randomize=False):
        current_set = set(range(1, N + 1))
        active_triples = list(triples)
        
        while active_triples:
            if time.time() > deadline:
                break
            
            deg = {i: 0 for i in range(1, N + 1)}
            for t in active_triples:
                for x in t:
                    if x in current_set:
                        deg[x] += 1
            
            if not any(deg[x] > 0 for x in current_set):
                break
                
            candidates = [x for x in current_set if deg[x] > 0]
            
            if randomize:
                # Pick from top 10% or random
                candidates.sort(key=lambda x: -deg[x])
                top_k = max(1, len(candidates) // 10)
                to_remove = random.choice(candidates[:top_k])
            else:
                to_remove = max(candidates, key=lambda x: deg[x])
            
            current_set.remove(to_remove)
            active_triples = [t for t in active_triples if to_remove not in t]
            
        return current_set

    def try_add_back(s, triples_by_elem, N, deadline):
        s = set(s)
        removed = [i for i in range(1, N + 1) if i not in s]
        random.shuffle(removed)
        
        for x in removed:
            if time.time() > deadline:
                break
            valid = True
            for t in triples_by_elem[x]:
                others = [y for y in t if y != x]
                if all(y in s for y in others):
                    valid = False
                    break
            if valid:
                s.add(x)
        return s

    # Seed solution
    best_set = set(range(91, 181))
    
    for i in range(1000):
        if time.time() > deadline:
            break
        
        rand = (i % 2 == 0)
        s = greedy_hit_set(triples, N, deadline, randomize=rand)
        s = try_add_back(s, triples_by_elem, N, deadline)
        
        if len(s) > len(best_set):
            best_set = s
            
    return list(best_set)
