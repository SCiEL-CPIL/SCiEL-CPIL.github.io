import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    
    n = 16
    points = list(range(n))
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    K_to_triples = []
    triple_to_Ks = [[] for _ in range(num_triples)]
    
    # Generate partitions of 6 elements into 3 pairs
    partitions = []
    def gen_partitions(i, used, current):
        if i == 6:
            partitions.append(tuple(sorted(current)))
            return
        u = -1
        for j in range(6):
            if not used[j]:
                u = j
                break
        for v in range(u+1, 6):
            if not used[v]:
                used[v] = True
                current.append((u, v))
                gen_partitions(i+2, used, current)
                current.pop()
                used[v] = False
                
    used = [False]*6
    gen_partitions(0, used, [])
    
    for verts in combinations(range(n), 6):
        for part in partitions:
            p1 = (verts[part[0][0]], verts[part[0][1]])
            p2 = (verts[part[1][0]], verts[part[1][1]])
            p3 = (verts[part[2][0]], verts[part[2][1]])
            
            k_triples = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        t = tuple(sorted((a, b, c)))
                        idx = triple_to_idx[t]
                        k_triples.append(idx)
                        
            k_idx = len(K_to_triples)
            K_to_triples.append(k_triples)
            for t_idx in k_triples:
                triple_to_Ks[t_idx].append(k_idx)
                
    num_Ks = len(K_to_triples)
    
    best_score = 0
    best_set = set()
    
    while time.time() < deadline:
        present = set(range(num_triples))
        bad_Ks_list = list(range(num_Ks))
        pos = {k: i for i, k in enumerate(bad_Ks_list)}
        
        deg = [len(triple_to_Ks[t]) for t in range(num_triples)]
        
        while bad_Ks_list:
            idx = random.randint(0, len(bad_Ks_list) - 1)
            k = bad_Ks_list[idx]
            
            last_k = bad_Ks_list[-1]
            bad_Ks_list[idx] = last_k
            pos[last_k] = idx
            bad_Ks_list.pop()
            del pos[k]
            
            k_trips = K_to_triples[k]
            
            max_deg = -1
            best_t = -1
            # Randomized tie-breaking: collect candidates
            # For speed, just pick first max, but random K selection helps
            for t in k_trips:
                if deg[t] > max_deg:
                    max_deg = deg[t]
                    best_t = t
            
            present.remove(best_t)
            
            for k_prime in triple_to_Ks[best_t]:
                if k_prime in pos:
                    kp_idx = pos[k_prime]
                    last_kp = bad_Ks_list[-1]
                    bad_Ks_list[kp_idx] = last_kp
                    pos[last_kp] = kp_idx
                    bad_Ks_list.pop()
                    del pos[k_prime]
                    
                    for t_prime in K_to_triples[k_prime]:
                        deg[t_prime] -= 1
                        
        score = len(present)
        if score > best_score:
            best_score = score
            best_set = set(present)
            
    result = [list(all_triples[i]) for i in best_set]
    return result
