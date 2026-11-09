import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    adj = [0] * n
    edges = set()

    def add_edge(u, v):
        if u == v:
            return
        if adj[u] & (1 << v):
            return
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.add((min(u, v), max(u, v)))

    def remove_edge(u, v):
        if not (adj[u] & (1 << v)):
            return
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edges.discard((min(u, v), max(u, v)))

    # Precompute splits
    all_splits = []
    seen = set()
    for A in combinations(range(8), 4):
        B = tuple(i for i in range(8) if i not in A)
        key = frozenset([frozenset(A), frozenset(B)])
        if key not in seen:
            seen.add(key)
            mask_a = 0
            mask_b = 0
            for i in A:
                mask_a |= (1 << i)
            for i in B:
                mask_b |= (1 << i)
            all_splits.append((mask_a, mask_b, A, B))

    # Precompute 8-sets for each pair
    # pair_to_sets[(u,v)] = list of tuples of 8 vertices
    pair_to_sets = {}
    for u in range(n):
        for v in range(u + 1, n):
            others = [i for i in range(n) if i != u and i != v]
            sets = []
            for combo in combinations(others, 6):
                # Sort to canonicalize? No, just store as is.
                # But we need to map the local indices 0..7 to the vertices.
                # The order matters for the split masks.
                # We will sort the 8 vertices to have a canonical order?
                # If we sort, we need to map the split masks (which are on 0..7) to the sorted vertices.
                # This is consistent.
                S = tuple(sorted((u, v) + tuple(combo)))
                sets.append(S)
            pair_to_sets[(u, v)] = sets

    def has_q3_in_subset(S, adj):
        # S is sorted tuple of 8
        la = [0] * 8
        for i in range(8):
            vi = S[i]
            m = 0
            for j in range(8):
                if i != j and (adj[vi] & (1 << S[j])):
                    m |= (1 << j)
            la[i] = m
        
        for mask_a, mask_b, A_idx, B_idx in all_splits:
            used_b = 0
            ok = True
            for i in A_idx:
                m = mask_b & ~la[i]
                if m & (m - 1):
                    ok = False
                    break
                if m:
                    if used_b & m:
                        ok = False
                        break
                    used_b |= m
            if ok:
                return True
        return False

    best_score = 0
    best_edge_set = set()

    for restart in range(5):
        if time.time() > deadline: break
        adj = [0] * n
        edges = set()
        
        # Random init
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.7:
                    add_edge(u, v)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = [(u,v) for u in range(n) for v in range(u+1,n) if not (adj[u] & (1<<v))]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline: break
                add_edge(u, v)
                
                found = False
                for S in pair_to_sets[(u,v)]:
                    if has_q3_in_subset(S, adj):
                        found = True
                        break
                
                if found:
                    remove_edge(u, v)
                else:
                    improved = True
                    if len(edges) > best_score:
                        best_score = len(edges)
                        best_edge_set = set(edges)

    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))
