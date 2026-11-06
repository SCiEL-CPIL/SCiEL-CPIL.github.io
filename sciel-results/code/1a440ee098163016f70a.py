import time
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    N = 256
    # Precompute squares
    squares = set()
    k = 2
    while k*k <= 2*N:
        squares.add(k*k)
        k += 1
        
    # Precompute conflicts
    forbidden = set()
    conflicts = [set() for _ in range(N+1)]
    for i in range(1, N+1):
        if 2*i in squares:
            forbidden.add(i)
        for j in range(i+1, N+1):
            if i+j in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    valid_nodes = [i for i in range(1, N+1) if i not in forbidden]
    
    # Find connected components
    visited = [False] * (N+1)
    components = []
    for i in range(1, N+1):
        if i in forbidden:
            continue
        if not visited[i]:
            queue = deque([i])
            visited[i] = True
            comp = []
            while queue:
                u = queue.popleft()
                comp.append(u)
                for v in conflicts[u]:
                    if not visited[v] and v not in forbidden:
                        visited[v] = True
                        queue.append(v)
            components.append(comp)

    def solve_component(comp):
        # If component is small, use exact B&B or exhaustive search
        # Otherwise, use local search
        if len(comp) <= 25:
            # Exhaustive search with pruning
            best = set()
            comp_sorted = sorted(comp, key=lambda x: len(conflicts[x]))
            def backtrack(idx, current):
                nonlocal best
                if idx == len(comp_sorted):
                    if len(current) > len(best):
                        best = set(current)
                    return
                u = comp_sorted[idx]
                # Try including u
                if all(u not in conflicts[v] for v in current):
                    current.append(u)
                    backtrack(idx+1, current)
                    current.pop()
                # Try excluding u
                backtrack(idx+1, current)
            backtrack(0, [])
            return best
        else:
            # Local search with restarts
            best_S = set()
            restarts = 0
            while time.time() < deadline - 2:
                restarts += 1
                nodes = comp[:]
                random.shuffle(nodes)
                S = set()
                for u in nodes:
                    ok = True
                    for v in S:
                        if u in conflicts[v]:
                            ok = False
                            break
                    if ok:
                        S.add(u)
                        
                improved = True
                while improved and time.time() < deadline - 2:
                    improved = False
                    # Try add
                    for x in comp:
                        if x in S:
                            continue
                        ok = True
                        for y in S:
                            if x in conflicts[y]:
                                ok = False
                                break
                        if ok:
                            S.add(x)
                            improved = True
                            break
                    if improved:
                        continue
                        
                    # Try 2-swap
                    for y in list(S):
                        S_temp = S - {y}
                        candidates = []
                        for x in comp:
                            if x in S:
                                continue
                            ok = True
                            for z in S_temp:
                                if x in conflicts[z]:
                                    ok = False
                                    break
                            if ok:
                                candidates.append(x)
                        if len(candidates) >= 2:
                            for i in range(len(candidates)):
                                for j in range(i+1, len(candidates)):
                                    x1 = candidates[i]
                                    x2 = candidates[j]
                                    if x1 not in conflicts[x2]:
                                        S.remove(y)
                                        S.add(x1)
                                        S.add(x2)
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                            
                if len(S) > len(best_S):
                    best_S = S
                    
                if restarts > 50:
                    break
            return best_S

    best_set = set()
    
    # Solve each component
    for comp in components:
        best_comp = solve_component(comp)
        best_set.update(best_comp)
        
    return list(best_set)
