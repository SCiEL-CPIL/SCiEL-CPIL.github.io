import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 15
    
    def find_violation(adj_list, deg):
        for u in range(n):
            for v in adj_list[u]:
                if v <= u:
                    continue
                for s in adj_list[u]:
                    if s == v:
                        continue
                    for t in adj_list[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return (s, u, v, t)
        return None
    
    # Start with complete graph
    adj_list = [set(range(n)) - {i} for i in range(n)]
    deg = [n-1] * n
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        viol = find_violation(adj_list, deg)
        if viol is None:
            break
        
        s, u, v, t = viol
        # Remove middle edge (u,v)
        adj_list[u].remove(v)
        adj_list[v].remove(u)
        deg[u] -= 1
        deg[v] -= 1
    
    result = []
    for u in range(n):
        for v in adj_list[u]:
            if u < v:
                result.append([int(u), int(v)])
    
    return result
