"""Construct a K3,3-free graph on 20 vertices maximizing edges."""
import random

def construct():
    n = 20
    
    def is_valid_addition(edges, u, v):
        """Check if adding edge (u,v) keeps the graph K3,3-free."""
        if u == v or u > v:
            return False
        if (u, v) in edges:
            return True
        
        # Case 1: u,v in same part of K3,3
        # S1 = common neighbors of u and v
        S1 = []
        for w in range(n):
            if w == u or w == v:
                continue
            if (min(u,w), max(u,w)) in edges and (min(v,w), max(v,w)) in edges:
                S1.append(w)
        
        if len(S1) >= 3:
            for x in range(n):
                if x == u or x == v:
                    continue
                count = 0
                for b in S1:
                    if (min(x,b), max(x,b)) in edges:
                        count += 1
                if count >= 3:
                    return False
        
        # Case 2: u in one part, v in other
        for a1 in range(n):
            if a1 == u or a1 == v:
                continue
            for a2 in range(a1+1, n):
                if a2 == u or a2 == v:
                    continue
                count = 0
                for w in range(n):
                    if w in (u, v, a1, a2):
                        continue
                    if ((min(u,w), max(u,w)) in edges and 
                        (min(v,w), max(v,w)) in edges and 
                        (min(a1,w), max(a1,w)) in edges and 
                        (min(a2,w), max(a2,w)) in edges):
                        count += 1
                        if count >= 2:
                            return False
        
        return True
    
    def try_greedy(seed):
        random.seed(seed)
        edges = set()
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if is_valid_addition(edges, u, v):
                edges.add((u, v))
        
        # Second pass: try again in different order
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if (u, v) not in edges and is_valid_addition(edges, u, v):
                edges.add((u, v))
        
        return edges
    
    best = set()
    for seed in range(20):
        candidate = try_greedy(seed)
        if len(candidate) > len(best):
            best = candidate
    
    return [[u, v] for u, v in best]
