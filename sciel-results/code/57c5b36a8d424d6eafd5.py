import math
import time

def construct():
    N = 255
    start_time = time.time()
    
    # Candidates must share a factor with 255
    # Primes of 255: 3, 5, 17
    candidates = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    
    # We must include 255.
    # Find index of 255
    idx_255 = candidates.index(255)
    
    # We want max clique in G[candidates] containing 255.
    # This is equivalent to max clique in G[candidates \ {255}] + {255}.
    # Let's build graph on candidates \ {255}.
    
    # Remove 255 from list for graph construction to save a tiny bit of time/memory
    # But indices need to be consistent.
    # Let's just build on all candidates, and force 255 in the clique.
    # Or better, build on V' = candidates \ {255}.
    
    V = [c for c in candidates if c != 255]
    n = len(V)
    
    # Build adjacency list
    # adj[i] is set of j such that gcd(V[i], V[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        vi = V[i]
        for j in range(i + 1, n):
            vj = V[j]
            if math.gcd(vi, vj) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Bron-Kerbosch with pivot to find max clique
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        # Time check
        if time.time() - start_time > 80:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
            
        # Choose pivot
        # u from P U X
        # We want to minimize |P \ N(u)|, so maximize |P \cap N(u)|
        
        # If P is empty, we can't choose pivot from P, but X might not be empty.
        # However, if P is empty, the loop below won't execute anyway.
        # So we only need pivot if P is not empty.
        
        if not P:
            return
            
        # Find best pivot
        # To speed up, we can just pick a node from P with high degree in P
        # or iterate all in P U X.
        # n is small (~126), so iterating is fine.
        
        max_inter = -1
        pivot = -1
        for u in list(P) + list(X):
            # Intersection of P and adj[u]
            # adj[u] contains indices in V
            inter = P.intersection(adj[u])
            if len(inter) > max_inter:
                max_inter = len(inter)
                pivot = u
                
        if pivot == -1:
            # Should not happen if P is not empty
            return
            
        # Iterate over P \ N(pivot)
        # N(pivot) is adj[pivot]
        for v in list(P - adj[pivot]):
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
            
        # If time is running out, stop
        if time.time() - start_time > 85:
            return

    # Initialize
    P = set(range(n))
    X = set()
    R = set()
    
    bron_kerbosch(R, P, X)
    
    # Convert indices to values
    clique_vals = [V[i] for i in best_clique]
    clique_vals.append(255)
    
    return clique_vals
