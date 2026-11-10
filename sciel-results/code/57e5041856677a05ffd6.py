import math
import time
import sys

def construct():
    N = 255
    deadline = time.time() + 55.0
    
    # 1. Generate candidates
    candidates = [i for i in range(1, N + 1) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n_cand = len(candidates)
    
    # 2. Build adjacency matrix for gcd > 1 graph (G)
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Find Max Clique in G using Tomita's algorithm (Branch and Bound)
    # This is robust and fast for N=127
    
    best_clique = []
    
    def expand(P, X, clique):
        nonlocal best_clique
        if not P and not X:
            if len(clique) > len(best_clique):
                best_clique = list(clique)
            return
        
        # Simple bound: len(clique) + len(P) <= len(best_clique) -> prune
        if len(clique) + len(P) <= len(best_clique):
            return
            
        # Select pivot u in P union X with max degree in P
        # For simplicity, just pick first one or one with max intersection
        # Better: pick u in P with max |P ∩ N(u)|
        if P:
            # Find pivot u in P that maximizes |P ∩ N(u)|
            # If P is small, this is fast. If P is large, we can approximate.
            # For N=127, we can afford a simple loop.
            max_deg = -1
            pivot = -1
            for u in P:
                deg = len(P & adj[u])
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            # Order P by color or just use P \ N(pivot)
            # Tomita uses coloring, but a simple B&B with pivot is often sufficient for N<200
            # Let's use a simpler recursive B&B with pivot
            if pivot != -1:
                # Process vertices in P that are NOT adjacent to pivot
                # This is a valid branch decomposition
                for v in list(P - adj[pivot]):
                    if time.time() > deadline:
                        return
                    expand(P & adj[v], X | {v}, clique + [v])
                    P.remove(v)
                    X.add(v)
            else:
                for v in list(P):
                    if time.time() > deadline:
                        return
                    expand(P & adj[v], X | {v}, clique + [v])
                    P.remove(v)
                    X.add(v)
        else:
            # P is empty, just record clique if X is empty? 
            # Actually if P is empty, we have a maximal clique.
            if not X:
                if len(clique) > len(best_clique):
                    best_clique = list(clique)
            
    # Initial call
    # To speed up, we can order vertices by degree
    # But Tomita's standard implementation works on the set P.
    # We need to be careful with set operations performance.
    # For N=127, sets are fine.
    
    # Heuristic initialization: Greedy clique
    greedy_clique = []
    P = set(range(n_cand))
    while P:
        # Pick vertex with max degree in P
        v = max(P, key=lambda x: len(adj[x] & P))
        greedy_clique.append(v)
        P = P & adj[v]
        
    best_clique = list(greedy_clique)
    
    # Run Tomita
    # We need to pass a set for P and X
    # Note: Python recursion limit might be an issue, but depth is max 127.
    # sys.setrecursionlimit(10000)
    
    # To avoid recursion depth issues and overhead, we can use an iterative version or just accept it.
    # Let's try the recursive one first.
    
    # One issue: The `expand` function above modifies P and X in place.
    # We must pass copies or handle state carefully.
    # Standard Tomita:
    
    def tomita_max_clique(P, X, clique):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P:
            if not X:
                if len(clique) > len(best_clique):
                    best_clique = list(clique)
            return
        
        if len(clique) + len(P) <= len(best_clique):
            return
            
        # Pivot
        # Choose u in P ∪ X with max |P ∩ N(u)|
        # If X is empty, u in P. If X not empty, u in X usually better?
        # Tomita says: choose u in P ∪ X.
        # If P is not empty, we can just pick u in P.
        # Actually, if we pick u in X, P \ N(u) is a subset of P \ N(u for u in P)? No.
        # Standard: pick u in P ∪ X.
        
        # For simplicity and speed, let's just pick u in P with max degree in P.
        if P:
            u = max(P, key=lambda x: len(P & adj[x]))
            # Branch on P \ N(u)
            for v in list(P - adj[u]):
                tomita_max_clique(P & adj[v], X | {v}, clique + [v])
                P.remove(v)
                X.add(v)
        else:
            # P empty
            if not X:
                if len(clique) > len(best_clique):
                    best_clique = list(clique)
                
    tomita_max_clique(set(range(n_cand)), set(), [])
    
    result = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in result. 
    # Since 255 is connected to all multiples of 3,5,17, and all candidates are multiples of 3,5,17,
    # 255 is connected to ALL candidates.
    # So any clique can include 255 if it doesn't conflict. It doesn't conflict.
    # So we can just add 255 if it's not there.
    if 255 not in result:
        result.append(255)
        
    return result
