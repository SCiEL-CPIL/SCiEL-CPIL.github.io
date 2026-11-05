import time
from math import gcd
from itertools import combinations

N = 512
DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Precompute compatibility: comp[a][b] = True if lcm(a,b) <= N
    # We'll use a set-based approach for the graph
    # Nodes are 1..N
    # Edge between a and b if lcm(a,b) <= N
    
    # For efficiency, precompute for each a, the set of b that are compatible
    # Actually, let's think about this differently.
    # 
    # Key insight: if a divides b, then lcm(a,b) = b <= 512, so they're compatible.
    # More generally, lcm(a,b) = a*b/gcd(a,b) <= 512.
    
    # Let's build the graph of compatible pairs.
    # For each pair (a,b) with a < b, check if lcm(a,b) <= 512.
    
    # We want the maximum clique in this graph.
    # Maximum clique is NP-hard, but N=512 is small enough for branch and bound.
    
    # First, let's build the adjacency list.
    adj = [set() for _ in range(N+1)]
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            l = a * b // gcd(a, b)
            if l <= N:
                adj[a].add(b)
                adj[b].add(a)
    
    # Now find maximum clique using branch and bound.
    # Use the classic Bron-Kerbosch with pivot, or a simpler branch and bound.
    
    # Let's use a greedy initialization first, then local search / branch and bound.
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Greedy: sort by degree descending, add if compatible with all current
    degrees = [len(adj[i]) for i in range(1, N+1)]
    order = sorted(range(1, N+1), key=lambda x: -degrees[x])
    
    greedy_clique = []
    for v in order:
        if all(gcd(v, u) * u // gcd(v, u) <= N for u in greedy_clique):
            # Wait, lcm(v,u) = v*u//gcd(v,u)
            if all(v * u // gcd(v, u) <= N for u in greedy_clique):
                greedy_clique.append(v)
    
    best_clique = list(greedy_clique)
    best_size = len(best_clique)
    
    # Now try to improve via local search: try adding vertices, or swapping.
    # Also try branch and bound for maximum clique.
    
    # Let's implement a simple branch and bound for maximum clique.
    # State: current clique C, candidate set P (vertices compatible with all in C)
    # Bound: |C| + |P|
    
    import sys
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(C, P):
        nonlocal best_clique, best_size
        if time_remaining() < 0.5:
            return
        
        if not P:
            if len(C) > best_size:
                best_clique = list(C)
                best_size = len(C)
            return
        
        # Bound
        if len(C) + len(P) <= best_size:
            return
        
        # Pick pivot u from P to reduce branching
        # Use the vertex with max degree in P as pivot
        if P:
            pivot = max(P, key=lambda v: len(adj[v] & P))
            # Branch on P \ N(pivot)
            candidates = P - adj[pivot]
        else:
            candidates = P
        
        for v in list(candidates):
            if time_remaining() < 0.5:
                return
            if len(C) + 1 + len(P & adj[v]) <= best_size:
                continue
            C.append(v)
            branch_and_bound(C, P & adj[v])
            C.pop()
            P.discard(v)
    
    # Start branch and bound from empty set
    # But this might be too slow. Let's start from the greedy clique and try to extend.
    
    # Actually, let's try a different approach: start with greedy, then try to add/remove.
    
    # Local search: try to add a vertex not in clique that is compatible with all in clique
    current = set(best_clique)
    improved = True
    while improved:
        improved = False
        for v in range(1, N+1):
            if v in current:
                continue
            if all(v * u // gcd(v, u) <= N for u in current):
                current.add(v)
                improved = True
                if time_remaining() < 0.5:
                    break
        if time_remaining() < 0.5:
            break
    
    best_clique = list(current)
    best_size = len(best_clique)
    
    # Now try swapping: remove one vertex, add one or more
    for _ in range(50):  # limited iterations
        if time_remaining() < 1.0:
            break
        improved = False
        for i in range(len(best_clique)):
            if time_remaining() < 0.5:
                break
            removed = best_clique[i]
            temp = best_clique[:i] + best_clique[i+1:]
            temp_set = set(temp)
            # Try to add any vertex not in temp_set
            for v in range(1, N+1):
                if v in temp_set:
                    continue
                if all(v * u // gcd(v, u) <= N for u in temp_set):
                    new_clique = temp + [v]
                    if len(new_clique) > best_size:
                        best_clique = new_clique
                        best_size = len(best_clique)
                        improved = True
                    break
        if not improved:
            break
    
    # Try branch and bound from scratch with a time limit
    if time_remaining() > 5.0:
        # Use branch and bound starting from empty
        P = set(range(1, N+1))
        C = []
        branch_and_bound(C, P)
    
    # Final validation
    result = list(best_clique)
    # Ensure non-empty
    if not result:
        result = [1]
    
    return result
