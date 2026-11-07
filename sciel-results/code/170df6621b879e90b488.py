import time
import random
from collections import deque

def construct():
    deadline = time.time() + 55
    
    # Best known solution: two K_{3,3} components
    best_edges = []
    for u in range(3):
        for v in range(3, 6):
            best_edges.append([u, v])
    for u in range(6, 9):
        for v in range(9, 12):
            best_edges.append([u, v])
    best_value = 2
    
    def count_cycle_lengths(adj):
        n = len(adj)
        adj_set = [set(s) for s in adj]
        lengths = set()
        for L in range(4, 13, 2):
            found = False
            for start in range(n):
                stack = [(start, start, 1, 1 << start)]
                while stack and not found:
                    cur, prev, depth, mask = stack.pop()
                    if depth == L:
                        if start in adj_set[cur]:
                            found = True
                            break
                        continue
                    for nb in adj_set[cur]:
                        if nb != prev and not (mask & (1 << nb)):
                            stack.append((nb, cur, depth + 1, mask | (1 << nb)))
            if found:
                lengths.add(L)
        return lengths
    
    def is_valid(adj):
        n = len(adj)
        for i in range(n):
            if len(adj[i]) < 3:
                return False
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                for k in adj[i]:
                    if k > j and k in adj[j]:
                        return False
        visited = [False] * n
        for s in range(n):
            if not visited[s]:
                visited[s] = True
                q = deque([s])
                nv = 0
                e = 0
                while q:
                    u = q.popleft()
                    nv += 1
                    for v in adj[u]:
                        e += 1
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
                if e // 2 >= nv:
                    return True
        return False
    
    random.seed(42)
    for _ in range(30000):
        if time.time() > deadline - 5:
            break
        b_stubs = [b for b in range(6, 12) for _ in range(3)]
        random.shuffle(b_stubs)
        adj = [[] for _ in range(12)]
        for a in range(6):
            adj[a] = b_stubs[a*3:(a+1)*3]
            for b in adj[a]:
                adj[b].append(a)
        
        if not is_valid(adj):
            continue
        
        lens = count_cycle_lengths(adj)
        if len(lens) < best_value:
            best_value = len(lens)
            best_edges = [[u, v] for u in range(12) for v in adj[u] if u < v]
            if best_value == 1:
                break
    
    return [list(e) for e in best_edges]
