import math
import random

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    N = 256
    nodes = list(range(1, N + 1))
    
    # 1. Build Graph
    adj = {u: set() for u in nodes}
    for i in range(len(nodes)):
        u = nodes[i]
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            if is_square(u + v):
                adj[u].add(v)
                adj[v].add(u)
                
    def run_local_search(initial_S):
        S = set(initial_S)
        in_S = [False] * (N + 1)
        for u in S:
            in_S[u] = True
            
        # Compute initial conflict counts
        conflict_count = [0] * (N + 1)
        for u in S:
            for v in adj[u]:
                if not in_S[v]:
                    conflict_count[v] += 1
        
        def update_add():
            added = 0
            for u in nodes:
                if not in_S[u] and conflict_count[u] == 0:
                    in_S[u] = True
                    S.add(u)
                    added += 1
                    for v in adj[u]:
                        if not in_S[v]:
                            conflict_count[v] += 1
            return added
        
        update_add()
        best_size = len(S)
        best_S = set(S)
        
        no_improve_count = 0
        
        for _ in range(500):
            if update_add() > 0:
                no_improve_count = 0
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                continue
                
            # No adds. Try swaps.
            candidates = []
            for u in nodes:
                if in_S[u]:
                    continue
                if conflict_count[u] == 1:
                    v = None
                    for w in adj[u]:
                        if in_S[w]:
                            v = w
                            break
                    if v is not None:
                        gain = 0
                        for w in adj[v]:
                            if not in_S[w]:
                                if conflict_count[w] == 1:
                                    gain += 1
                        for w in adj[u]:
                            if not in_S[w]:
                                if conflict_count[w] == 0:
                                    gain -= 1
                        candidates.append((gain, v, u))
                        
            if not candidates:
                break
                
            candidates.sort(key=lambda x: x[0], reverse=True)
            gain, v, u = candidates[0]
            
            # Perform swap
            in_S[v] = False
            S.remove(v)
            for w in adj[v]:
                if not in_S[w]:
                    conflict_count[w] -= 1
                    
            in_S[u] = True
            S.add(u)
            for w in adj[u]:
                if not in_S[w]:
                    conflict_count[w] += 1
                    
            if gain <= 0:
                no_improve_count += 1
                if no_improve_count > 20:
                    break
            else:
                no_improve_count = 0
                
        return list(best_S)
        
    # Try different initializations
    # 1. R1 (1 mod 4)
    init1 = [i for i in nodes if i % 4 == 1]
    res1 = run_local_search(init1)
    
    # 2. R3 (3 mod 4)
    init2 = [i for i in nodes if i % 4 == 3]
    res2 = run_local_search(init2)
    
    # 3. R1 + R2 subset? No, just R1 and R3 are the pure independent sets.
    # Maybe try R0 subset?
    # Let's try R1 and R3.
    
    if len(res1) >= len(res2):
        return res1
    else:
        return res2

