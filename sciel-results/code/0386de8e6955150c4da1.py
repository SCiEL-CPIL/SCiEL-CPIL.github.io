import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # Precompute forbidden triples
    # A triple (a,b,c) is forbidden if 1/a = 1/b + 1/c for some permutation.
    # This is equivalent to checking if any element is the "sum" of the reciprocals of the other two.
    # Let the set be {u,v,w}. It is forbidden if:
    # 1. 1/u = 1/v + 1/w  => vw = u(v+w)
    # 2. 1/v = 1/u + 1/w  => uw = v(u+w)
    # 3. 1/w = 1/u + 1/v  => uv = w(u+v)
    
    # We store conflicts for each node i: list of pairs (j, k) such that {i, j, k} is forbidden
    conflicts = [set() for _ in range(N)]
    
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            for w in range(v + 1, N + 1):
                # Check if u is the 'a' in 1/a = 1/b + 1/c with b,c being v,w
                # 1/u = 1/v + 1/w  <=> v*w = u*(v+w)
                if v * w == u * (v + w):
                    # Triple (u, v, w) is forbidden
                    conflicts[u-1].add((v-1, w-1))
                    conflicts[v-1].add((u-1, w-1))
                    conflicts[w-1].add((u-1, v-1))
                    continue
                
                # Check if v is the 'a' in 1/a = 1/b + 1/c with b,c being u,w
                # 1/v = 1/u + 1/w  <=> u*w = v*(u+w)
                if u * w == v * (u + w):
                    # Triple (u, v, w) is forbidden
                    conflicts[u-1].add((v-1, w-1))
                    conflicts[v-1].add((u-1, w-1))
                    conflicts[w-1].add((u-1, v-1))
                    continue
                    
                # Check if w is the 'a' in 1/a = 1/b + 1/c with b,c being u,v
                # 1/w = 1/u + 1/v  <=> u*v = w*(u+v)
                if u * v == w * (u + v):
                    # Triple (u, v, w) is forbidden
                    conflicts[u-1].add((v-1, w-1))
                    conflicts[v-1].add((u-1, w-1))
                    conflicts[w-1].add((u-1, v-1))
    
    # If no forbidden triples, return all
    if not any(conflicts):
        return universe

    # Heuristic: Greedy + Local Search with Bitsets
    # S is a set of 0-based indices
    # mask is an integer bitmask of S
    
    def can_add(S_mask, i):
        # Check if i can be added to S_mask without forming a forbidden triple
        # For each pair (j, k) in conflicts[i], check if j and k are in S_mask
        for j, k in conflicts[i]:
            if (S_mask & (1 << j)) and (S_mask & (1 << k)):
                return False
        return True

    def greedy_construct(order):
        S = set()
        S_mask = 0
        for i in order:
            if can_add(S_mask, i):
                S.add(i)
                S_mask |= (1 << i)
        return S, S_mask

    def local_search(S, S_mask):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # 1. Try to add any element
            for i in range(N):
                if i not in S and can_add(S_mask, i):
                    S.add(i)
                    S_mask |= (1 << i)
                    improved = True
                    break
            
            if not improved:
                # 2. Try to swap: remove j, add i
                # Optimization: Only try swaps if it might increase size? No, we want max size.
                # Swaps maintain size. We need to find a sequence of swaps that allows an add later?
                # Or just do random restarts.
                # Let's try simple swaps to escape local optima?
                # Actually, if we can't add, we are stuck at this size.
                # To improve, we need to remove some elements to allow others.
                # This is complex. Let's rely on restarts.
                pass
                
        return S, S_mask

    best_size = 0
    best_set = set()
    
    # Try different orderings
    # 1. Ascending
    # 2. Descending
    # 3. Random
    # 4. By degree (number of conflicts)
    
    degrees = [len(conflicts[i]) for i in range(N)]
    
    orderings = [
        sorted(range(N)),
        sorted(range(N), reverse=True),
        sorted(range(N), key=lambda x: degrees[x], reverse=True), # High degree first? Or low?
        sorted(range(N), key=lambda x: degrees[x]), # Low degree first
    ]
    
    for order in orderings:
        S, S_mask = greedy_construct(order)
        S, S_mask = local_search(S, S_mask)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Random restarts
    while time.time() < deadline - 5:
        order = list(range(N))
        random.shuffle(order)
        S, S_mask = greedy_construct(order)
        S, S_mask = local_search(S, S_mask)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Final polish: try to add any remaining
    for i in range(N):
        if i not in best_set:
            # Reconstruct mask for best_set
            mask = 0
            for x in best_set:
                mask |= (1 << x)
            if can_add(mask, i):
                best_set.add(i)
                mask |= (1 << i)

    # Convert back to 1-based
    return [x + 1 for x in best_set]
