import itertools
import time

def construct():
    n_points = 16
    all_triples = list(itertools.combinations(range(n_points), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    forbidden_sets = []
    seen_forbidden = set()
    
    def get_matchings(nodes):
        if len(nodes) == 0:
            yield []
            return
        first = nodes[0]
        rest = nodes[1:]
        for i in range(len(rest)):
            second = rest[i]
            new_rest = rest[:i] + rest[i+1:]
            for rest_matchings in get_matchings(new_rest):
                yield ((first, second),) + rest_matchings

    for combo in itertools.combinations(range(n_points), 6):
        for matching in get_matchings(combo):
            p1 = set(matching[0])
            p2 = set(matching[1])
            p3 = set(matching[2])
            
            transversals = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        t = tuple(sorted((a, b, c)))
                        transversals.append(t)
            
            key = frozenset(transversals)
            if key not in seen_forbidden:
                seen_forbidden.add(key)
                forbidden_sets.append(key)
                
    num_forbidden = len(forbidden_sets)
    
    triple_to_fids = [[] for _ in range(num_triples)]
    for fid, fset in enumerate(forbidden_sets):
        for t in fset:
            idx = triple_to_idx[t]
            triple_to_fids[idx].append(fid)
            
    freqs = [len(fids) for fids in triple_to_fids]
    order = sorted(range(num_triples), key=lambda i: freqs[i], reverse=True)
    ordered_fids = [triple_to_fids[order[i]] for i in range(num_triples)]
    
    counts = [0] * num_forbidden
    best_count = 105
    best_solution = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    selected_pos = []
    start_time = time.time()
    timeout = 115.0
    
    def search(pos, count):
        nonlocal best_count, best_solution
        
        if time.time() - start_time > timeout:
            return
            
        if count + (num_triples - pos) <= best_count:
            return
            
        if pos == num_triples:
            if count > best_count:
                best_count = count
                sol = [list(all_triples[order[i]]) for i in selected_pos]
                best_solution = sol
            return
        
        t_idx = order[pos]
        fids = ordered_fids[pos]
        
        can_include = True
        for fid in fids:
            if counts[fid] == 7:
                can_include = False
                break
                
        if can_include:
            for fid in fids:
                counts[fid] += 1
            selected_pos.append(pos)
            
            search(pos + 1, count + 1)
            
            selected_pos.pop()
            for fid in fids:
                counts[fid] -= 1
        
        search(pos + 1, count)

    search(0, 0)
    
    return best_solution
