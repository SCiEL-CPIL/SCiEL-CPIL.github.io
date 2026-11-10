import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    p = 41
    
    # 1. Compute NRS mod 41
    nrs = set()
    for x in range(1, p):
        if pow(x, (p - 1) // 2, p) == p - 1:
            nrs.add(x)
            
    # 2. Build graph where edge exists if diff is NR
    # We want max clique in this graph.
    adj = [set() for _ in range(p)]
    for u in range(p):
        for v in range(u + 1, p):
            if (u - v) % p in nrs:
                adj[u].add(v)
                adj[v].add(u)
                
    # 3. Max Clique Search
    best_clique = []
    
    def expand(R, P):
        global best_clique
        if len(R) + len(P) <= len(best_clique):
            return
        if not P:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Order P by degree to find large cliques faster
        P_list = sorted(P, key=lambda x: len(adj[x] & P), reverse=True)
        
        for v in P_list:
            new_R = R | {v}
            new_P = P & adj[v]
            expand(new_R, new_P)
            P.remove(v)
            if len(R) + len(P) <= len(best_clique):
                return
                
    expand(set(), set(range(p)))
    
    A = set(best_clique)
    
    # 4. Construct S
    inv5 = 33 # 5 * 33 = 165 = 1 mod 41
    S = []
    for a in A:
        # Residue 0 mod 5
        k0 = (a * inv5) % p
        S.append(5 * k0)
        
        # Residue 2 mod 5
        k2 = ((a - 2) * inv5) % p
        S.append(5 * k2 + 2)
        
    S = list(set(S))
    
    # 5. Verify
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if (S[i] - S[j]) % m in forbidden or (S[j] - S[i]) % m in forbidden:
                S = [0]
                break
        else:
            continue
        break
        
    return np.array(S, dtype=np.int64)
