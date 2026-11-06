import numpy as np
import time
import random

def construct():
    n = 40
    deadline = time.time() + 110
    
    adj = np.zeros((n, n), dtype=np.uint8)
    edge_set = set()
    edges = []
    
    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        edge_set.add((u, v))
        edges.append([u, v])
    
    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        edge_set.discard((u, v))
        edges.remove([u, v])
    
    # Start with 8 copies of K5 (80 edges)
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                add_edge(5*k+a, 5*k+b)
    
    best_count = len(edges)
    rng = random.Random(777)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def can_add(u, v):
        """O(n^2) check: does adding (u,v) create a C6?"""
        # C6-free iff every neighborhood is C4-free.
        # Adding (u,v) only can create C4s in N(w) for w in {u,v} ∪ (N(u)∩N(v)).
        
        # For w=u: v enters N(u). C4 in N(u) through v: 
        #   exists a,b,c in N(u) with adj[v,a], adj[a,b], adj[b,c], adj[c,v]
        #   i.e., a,c in N(u)∩N(v), b in N(u), adj[a,b]
        S = np.where(adj[u] & adj[v])[0]
        if len(S) >= 2:
            # Build adjacency among S within N(u)
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    a, c = int(S[i]), int(S[j])
                    # Check if a,c have common neighbor in N(u)
                    if np.any(adj[u] & adj[a] & adj[c]):
                        return False
        
        # For w=v: symmetric
        if len(S) >= 2:
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    a, c = int(S[i]), int(S[j])
                    if np.any(adj[v] & adj[a] & adj[c]):
                        return False
        
        # For w in S (common neighbors of u,v): edge (u,v) is now in N(w)
        # C4 in N(w) using edge (u,v): u-v-x-y-u
        #   x in N(w)∩N(v)\{u,w}, y in N(w)∩N(u)\{v,w}, adj[x,y]
        for w in S:
            w = int(w)
            # x candidates: neighbors of w and v, excluding u and w
            x_cand = np.where(adj[w] & adj[v])[0]
            x_cand = x_cand[(x_cand != u) & (x_cand != w)]
            # y candidates: neighbors of w and u, excluding v and w
            y_cand = np.where(adj[w] & adj[u])[0]
            y_cand = y_cand[(y_cand != v) & (y_cand != w)]
            if len(x_cand) > 0 and len(y_cand) > 0:
                if np.any(adj[np.ix_(x_cand, y_cand)]):
                    return False
        
        return True
    
    # Hill climbing with random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Randomly shuffle and try to add edges greedily
        shuffled = all_pairs[:]
        rng.shuffle(shuffled)
        
        for u, v in shuffled:
            if time.time() > deadline:
                break
            if not adj[u, v]:
                if can_add(u, v):
                    add_edge(u, v)
                    if len(edges) > best_count:
                        best_count = len(edges)
        
        # If we improved, keep going with more passes
        if len(edges) == best_count:
            # Try some random removals and re-additions
            for _ in range(50):
                if time.time() > deadline:
                    break
                if edges:
                    idx = rng.randrange(len(edges))
                    u, v = edges[idx]
                    remove_edge(u, v)
                    edges.pop(idx)
                # Try to add back
                shuffled2 = all_pairs[:]
                rng.shuffle(shuffled2)
                for u2, v2 in shuffled2:
                    if not adj[u2, v2] and can_add(u2, v2):
                        add_edge(u2, v2)
                        break
    
    return edges
