import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            edges.add((u, v))
            
    A_q3 = [0, 3, 5, 6]
    B_q3 = [1, 2, 4, 7]
    idx_a = {a: i for i, a in enumerate(A_q3)}
    
    neighbors_b_to_a = {
        1: [0, 3, 5],
        2: [0, 3, 6],
        4: [0, 5, 6],
        7: [3, 5, 6]
    }

    all_subsets = list(combinations(range(n), 8))
    random.shuffle(all_subsets)
    
    def find_q3_edge(subset):
        s = list(subset)
        
        for a_idx in combinations(range(8), 4):
            a_s = [s[i] for i in a_idx]
            b_s = [s[i] for i in range(8) if i not in a_idx]
            
            for perm_a in permutations(a_s):
                # perm_a[i] is vertex for A_q3[i]
                
                adj = {b: [] for b in B_q3}
                for b in B_q3:
                    for v in b_s:
                        valid = True
                        for a in neighbors_b_to_a[b]:
                            i = idx_a[a]
                            u_mapped = perm_a[i]
                            if (u_mapped, v) not in edges and (v, u_mapped) not in edges:
                                valid = False
                                break
                        if valid:
                            adj[b].append(v)
                
                # Matching
                # We can use a simple backtracking
                # Since it's 4x4, it's very fast
                # We only need to find ONE matching
                def match(b_idx, used_v, mapping):
                    if b_idx == 4:
                        return True
                    b = B_q3[b_idx]
                    for v in adj[b]:
                        if v not in used_v:
                            mapping[b] = v
                            if match(b_idx + 1, used_v | {v}, mapping):
                                return True
                            del mapping[b]
                    return False
                
                mapping_b = {}
                if match(0, set(), mapping_b):
                    u = perm_a[0]
                    v = mapping_b[1]
                    return (min(u, v), max(u, v))
        
        return None

    # Main loop
    while True:
        if time.time() > deadline:
            break
            
        found_edge = None
        for subset in all_subsets:
            e = find_q3_edge(subset)
            if e:
                found_edge = e
                break
        
        if found_edge:
            edges.remove(found_edge)
        else:
            break

    return [[u, v] for u, v in edges]
