import time
import itertools
import random

def construct():
    deadline = time.time() + 140
    n = 16
    
    all_4sets = list(itertools.combinations(range(n), 4))
    all_4sets_set = [set(s) for s in all_4sets]
    
    # Precompute disjoint pairs
    disjoint_pairs = []
    for i in range(len(all_4sets)):
        for j in range(i+1, len(all_4sets)):
            if not (all_4sets_set[i] & all_4sets_set[j]):
                disjoint_pairs.append((i, j))
    
    def is_q3_free(adj):
        for i, j in disjoint_pairs:
            A = all_4sets[i]
            B = all_4sets[j]
            count = 0
            for u in A:
                for v in B:
                    if not (adj[u] & (1 << v)):
                        count += 1
                        if count > 4:
                            break
                if count > 4:
                    break
            if count > 4:
                continue
            used_u = set()
            used_v = set()
            is_match = True
            for u in A:
                for v in B:
                    if not (adj[u] & (1 << v)):
                        if u in used_u or v in used_v:
                            is_match = False
                            break
                        used_u.add(u)
                        used_v.add(v)
                if not is_match:
                    break
            if is_match:
                return False
        return True
    
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    best_adj = adj[:]
    best_count = 39
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing_edges = [(i, j) for i, j in all_edges if not (adj[i] & (1 << j))]
    
    random.seed(42)
    random.shuffle(missing_edges)
    
    for i, j in missing_edges:
        if time.time() > deadline:
            break
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
        if is_q3_free(adj):
            new_count = sum(bin(a).count('1') for a in adj) // 2
            if new_count > best_count:
                best_count = new_count
                best_adj = adj[:]
        else:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
