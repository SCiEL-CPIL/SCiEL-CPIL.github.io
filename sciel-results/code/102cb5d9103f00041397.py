import math
import random
import time

def _get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    n = 256
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Weights: weight of vertex v is v
    weights = list(range(n + 1))
    
    # Precompute conflict matrix
    conflict = [[False] * (n + 1) for _ in range(n + 1)]
    for i in range(2, n + 1):
        for j in range(i + 1, n + 1):
            if math.gcd(i, j) > 1:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Vertices sorted by weight descending
    vertices = list(range(n, 0, -1))
    m = len(vertices)
    
    # Adjacency in vertex index space (0-based)
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if conflict[vertices[i]][vertices[j]]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy coloring for upper bound
    def greedy_color_bound(cand):
        if not cand:
            return 0
        color = {}
        max_color = -1
        for v in cand:
            used = set()
            for u in adj[v]:
                if u in color:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            max_color = max(max_color, c)
        
        # Sum of max weight per color class
        bound = 0
        for c in range(max_color + 1):
            cls = [v for v in cand if color[v] == c]
            if cls:
                bound += max(weights[vertices[v]] for v in cls)
        return bound
    
    best_sum = 0
    best_set = []
    
    def bb(pos, cur_sum, cur_set, cand):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if not cand:
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_set = cur_set[:]
            return
        
        ub = cur_sum + greedy_color_bound(cand)
        if ub <= best_sum:
            return
        
        # Branch on first candidate
        v = cand[0]
        rest = cand[1:]
        
        # Exclude v
        bb(pos + 1, cur_sum, cur_set, rest)
        
        if time.time() > deadline:
            return
        
        # Include v: remove v and its neighbors from rest
        new_cand = [u for u in rest if u not in adj[v]]
        cur_set.append(vertices[v])
        bb(pos + 1, cur_sum + weights[vertices[v]], cur_set, new_cand)
        cur_set.pop()
    
    bb(0, 0, [], list(range(m)))
    
    # Ensure 1 is in the set
    if 1 not in best_set:
        best_set.append(1)
    
    return best_set
