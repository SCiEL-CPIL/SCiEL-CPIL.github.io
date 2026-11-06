import time
import random
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin for 90s limit
    
    # 1. Precompute forbidden triples
    # A triple (a, b, c) is forbidden if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Note: a, b, c must be distinct.
    
    n = 128
    vertices = list(range(1, n + 1))
    
    # Precompute LCMs? Or just check on the fly?
    # Number of triples is ~340k. Checking LCM is fast.
    # But we need to know for each vertex, which pairs it conflicts with.
    
    # conflicts[v] will be a set of pairs (u, w) such that {v, u, w} is a forbidden triple.
    # Actually, for greedy/local search, we just need to check if adding x to a set S is valid.
    # is_valid_addition(x, S):
    #   for all pairs (a, b) in combinations(S, 2):
    #     if lcm(a,b) == lcm(a,x) == lcm(b,x): return False
    #   return True
    
    # This is O(|S|^2) per addition. |S| can be up to ~100. 100^2 = 10k ops.
    # 128 vertices * 10k = 1.2M ops per full pass. Very fast.
    
    def is_valid(x, S):
        # S is a set of integers
        # Check if adding x creates a forbidden triple with any pair in S
        # We need to check all pairs (a, b) in S
        # If lcm(a,b) == lcm(a,x) == lcm(b,x), then it's invalid.
        # Optimization: Precompute lcm(a, x) for all a in S?
        # Or just iterate.
        
        # Further optimization:
        # If we maintain a list of S, we can iterate.
        # For each a in S:
        #   lcm_ax = lcm(a, x)
        #   For each b in S (b != a):
        #     lcm_ab = lcm(a, b)
        #     lcm_bx = lcm(b, x)
        #     if lcm_ab == lcm_ax == lcm_bx: return False
        # This is O(|S|^2).
        
        # Better:
        # Group elements in S by their LCM with x?
        # If lcm(a, x) == L, then for any b with lcm(b, x) == L, we need lcm(a, b) == L.
        
        # Let's just use the O(|S|^2) check, it's likely fast enough for N=128.
        
        s_list = list(S)
        n_s = len(s_list)
        for i in range(n_s):
            a = s_list[i]
            lcm_ax = lcm(a, x)
            for j in range(i + 1, n_s):
                b = s_list[j]
                lcm_ab = lcm(a, b)
                if lcm_ab == lcm_ax:
                    # Check lcm(b, x)
                    lcm_bx = lcm(b, x)
                    if lcm_bx == lcm_ax:
                        return False
        return True

    def greedy(order):
        S = set()
        for x in order:
            if is_valid(x, S):
                S.add(x)
        return S

    def local_search(S):
        # Try to improve S by swapping or adding
        # 1. Try to add any vertex not in S
        # 2. If adding x fails, try to remove a small number of vertices to make it possible
        #    and see if the net gain is positive.
        
        S_list = list(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each vertex not in S
            for x in range(1, n + 1):
                if x in S:
                    continue
                if is_valid(x, S):
                    S.add(x)
                    improved = True
                    break # Restart loop with new S
            
            # If no direct addition, try swaps (remove 1, add 1) or (remove 2, add 3)
            # This is complex. Let's stick to simple additions first.
            # If stuck, try random restarts or different greedy orders.
            
    best_set = set()
    
    # Strategy 1: Natural order
    S1 = greedy(vertices)
    best_set = S1
    
    # Strategy 2: Reverse order
    S2 = greedy(vertices[::-1])
    if len(S2) > len(best_set):
        best_set = S2
        
    # Strategy 3: Sorted by degree (number of forbidden triples involving v)
    # Calculate degree
    degrees = [0] * (n + 1)
    for a in vertices:
        for b in vertices:
            if a >= b: continue
            for c in vertices:
                if c <= b: continue
                if lcm(a,b) == lcm(a,c) == lcm(b,c):
                    degrees[a] += 1
                    degrees[b] += 1
                    degrees[c] += 1
                    
    # Sort by degree ascending (pick "easy" vertices first)
    order_asc = sorted(vertices, key=lambda x: degrees[x])
    S3 = greedy(order_asc)
    if len(S3) > len(best_set):
        best_set = S3
        
    # Sort by degree descending
    order_desc = sorted(vertices, key=lambda x: -degrees[x])
    S4 = greedy(order_desc)
    if len(S4) > len(best_set):
        best_set = S4
        
    # Run local search on the best set
    # Simple local search: try to add vertices
    # Since greedy is maximal (no single vertex can be added), local search must do swaps.
    # We'll skip complex swaps for now and rely on multiple greedy passes.
    
    # Try random shuffles
    for _ in range(100):
        if time.time() > deadline: break
        random.shuffle(vertices)
        Sr = greedy(vertices)
        if len(Sr) > len(best_set):
            best_set = Sr

    # Final check: ensure best_set is valid
    # (It should be, by construction)
    
    return list(best_set)
