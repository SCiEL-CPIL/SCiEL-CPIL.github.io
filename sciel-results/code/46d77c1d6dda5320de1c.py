import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    if not edges:
        return True
        
    def propagate(coloring):
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    cv = coloring[v]
                    if cv == 0:
                        c0 += 1
                    elif cv == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                    
                    if c0 == 5 or c1 == 5:
                        return False
                
                if c0 == 4 and len(unassigned) == 1:
                    coloring[unassigned[0]] = 1
                    changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    coloring[unassigned[0]] = 0
                    changed = True
        return True

    def backtrack(idx):
        if idx == n:
            for e in edges:
                c0 = sum(1 for v in e if coloring[v] == 0)
                if c0 == 5:
                    return False
            return True
            
        coloring[idx] = 0
        if propagate(coloring) and backtrack(idx + 1):
            return True
            
        coloring[idx] = 1
        if propagate(coloring) and backtrack(idx + 1):
            return True
            
        coloring[idx] = -1
        return False

    coloring = [-1] * n
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_edges = None
    best_n = 9
    best_len = float('inf')
    
    def greedy_reduce(edges, n, deadline):
        current = edges[:]
        random.shuffle(current)
        
        while time.time() < deadline:
            changed = False
            i = 0
            while i < len(current):
                if time.time() > deadline:
                    return current
                e = current[i]
                new_edges = current[:i] + current[i+1:]
                if not is_2_colorable(n, new_edges):
                    current = new_edges
                    changed = True
                else:
                    i += 1
            if not changed:
                break
        return current

    # n=9 K9^5
    n = 9
    k9_edges = [tuple(e) for e in combinations(range(n), 5)]
    for i in range(30):
        if time.time() > deadline:
            break
        trial = greedy_reduce(k9_edges, n, deadline)
        if len(trial) < best_len:
            best_len = len(trial)
            best_edges = trial[:]
            best_n = n

    # n=9 Random
    for i in range(30):
        if time.time() > deadline:
            break
        rand_edges = [e for e in combinations(range(n), 5) if random.random() < 0.8]
        if not is_2_colorable(n, rand_edges):
            trial = greedy_reduce(rand_edges, n, deadline)
            if len(trial) < best_len:
                best_len = len(trial)
                best_edges = trial[:]
                best_n = n

    # n=10 Random
    n = 10
    for i in range(10):
        if time.time() > deadline:
            break
        rand_edges = [e for e in combinations(range(n), 5) if random.random() < 0.6]
        if not is_2_colorable(n, rand_edges):
            trial = greedy_reduce(rand_edges, n, deadline)
            if len(trial) < best_len:
                best_len = len(trial)
                best_edges = trial[:]
                best_n = n

    # n=11 Random
    n = 11
    for i in range(5):
        if time.time() > deadline:
            break
        rand_edges = [e for e in combinations(range(n), 5) if random.random() < 0.4]
        if not is_2_colorable(n, rand_edges):
            trial = greedy_reduce(rand_edges, n, deadline)
            if len(trial) < best_len:
                best_len = len(trial)
                best_edges = trial[:]
                best_n = n

    if best_edges is None:
        best_edges = k9_edges
        best_n = 9

    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
