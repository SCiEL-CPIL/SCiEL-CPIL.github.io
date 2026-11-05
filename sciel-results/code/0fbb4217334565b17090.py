import time
import random

def construct():
    start_time = time.time()
    limit = 55.0
    
    N = 256
    squares = {i*i for i in range(1, 20)}
    
    adj = [set() for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if u + v in squares:
                adj[u].add(v)
                adj[v].add(u)
                
    valid_nodes = [u for u in range(1, N + 1) if 2*u not in squares]
    
    best_set = set()
    best_size = 0
    
    max_restarts = 2000
    for r in range(max_restarts):
        if time.time() - start_time > limit:
            break
            
        random.shuffle(valid_nodes)
        
        S = set()
        C = set(valid_nodes)
        
        for u in valid_nodes:
            if u not in C:
                continue
            if not (adj[u] & S):
                S.add(u)
                C.remove(u)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        neighbors_in_S = {}
        for v in C:
            neighbors_in_S[v] = adj[v] & S
            
        addable = set(v for v in C if len(neighbors_in_S[v]) == 0)
        
        stuck_count = 0
        max_iter = 50000
        
        for i in range(max_iter):
            if time.time() - start_time > limit:
                break
                
            if addable:
                v = random.choice(list(addable))
                
                C.remove(v)
                S.add(v)
                addable.remove(v)
                
                # Update neighbors_in_S for w in C adjacent to v
                for w in adj[v]:
                    if w in C:
                        if len(neighbors_in_S[w]) == 0:
                            # w was addable, now it's not
                            addable.discard(w)
                        neighbors_in_S[w].add(v)
                        
            else:
                if not S:
                    break
                    
                # Remove u from S with min degree in S
                min_deg = 1000
                best_u = None
                max_deg_C = -1
                for u in S:
                    deg_s = len(adj[u] & S)
                    deg_c = len(adj[u] & C)
                    if deg_s < min_deg or (deg_s == min_deg and deg_c > max_deg_C):
                        min_deg = deg_s
                        max_deg_C = deg_c
                        best_u = u
                        
                if best_u is None:
                    break
                
                u = best_u
                S.remove(u)
                C.add(u)
                
                # Update neighbors_in_S for w in C adjacent to u
                for w in adj[u]:
                    if w in C:
                        neighbors_in_S[w].remove(u)
                        if len(neighbors_in_S[w]) == 0:
                            addable.add(w)
                            
                stuck_count += 1
                if stuck_count > 2000:
                    break
                    
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
                
    return list(best_set)
