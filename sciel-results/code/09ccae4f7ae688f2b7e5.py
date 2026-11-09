import time
import random

def construct():
    n = 15
    start = time.time()
    timeout = 55.0
    
    def make_adj(edges):
        adj = [set() for _ in range(n)]
        for u,v in edges:
            adj[u].add(v); adj[v].add(u)
        return adj
        
    def is_valid(edges):
        adj = make_adj(edges)
        deg = [len(adj[i]) for i in range(n)]
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v: continue
                    for t in adj[v]:
                        if t == u or t == s: continue
                        if deg[s] == deg[t]:
                            return False
        return True

    all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    
    best = set()
    best_count = 0
    
    random.seed(42)
    
    for restart in range(1000):
        if time.time() - start > timeout: break
        
        cur_edges = set()
        
        while time.time() - start < timeout:
            missing = [e for e in all_pairs if e not in cur_edges]
            if not missing:
                break
            
            success = False
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts and not success:
                e = random.choice(missing)
                if e in cur_edges:
                    attempts += 1
                    continue
                trial = cur_edges | {e}
                if is_valid(trial):
                    cur_edges = trial
                    success = True
                else:
                    attempts += 1
            
            if success:
                if len(cur_edges) > best_count:
                    best_count = len(cur_edges)
                    best = set(cur_edges)
            else:
                if cur_edges:
                    e_to_remove = random.choice(list(cur_edges))
                    cur_edges.remove(e_to_remove)
                else:
                    break
                    
    return [[u,v] for u,v in best]
