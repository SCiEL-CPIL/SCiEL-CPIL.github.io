import time
import random
import sys
import itertools
from collections import defaultdict
sys.setrecursionlimit(1000000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set((min(a, b), max(a, b)) for a, b in edges)
    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find all triangles
    triangles = []
    for i in range(n):
        for j in adj[i]:
            if j <= i: continue
            for k in adj[j]:
                if k <= j: continue
                if k in adj[i]:
                    triangles.append((i, j, k))
                    
    best_cycles = []
    for _ in range(100): # Try 100 random orderings
        random.shuffle(triangles)
        used_edges = set()
        cycles = []
        for tri in triangles:
            edges_in_tri = set()
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                edges_in_tri.add((min(a, b), max(a, b)))
            if not any(e in used_edges for e in edges_in_tri):
                used_edges.update(edges_in_tri)
                cycles.append(list(tri))
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
        if len(best_cycles) == 21: # Max possible
            break
            
    if len(best_cycles) >= 3:
        return best_cycles
    else:
        cycle1 = [(i * 1) % n for i in range(n)]
        cycle2 = [(i * 2) % n for i in range(n)]
        cycle3 = [(i * 5) % n for i in range(n)]
        return [cycle1, cycle2, cycle3]
