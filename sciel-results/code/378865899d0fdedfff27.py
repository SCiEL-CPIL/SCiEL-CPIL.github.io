import math
import time
import sys
from collections import defaultdict

class Tree:
    def __init__(self, labels):
        self.labels = labels
        self.n = len(labels)
        # adjacency list
        self.adj = defaultdict(set)
        self.branch_len = {} # (u, v) -> length
        self.parent = {} # for rooting during DP
        self.children = defaultdict(list)
        
    def add_edge(self, u, v, length):
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.branch_len[(u, v)] = length
        self.branch_len[(v, u)] = length

def _jc_distance(a, b):
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def reconstruct(sequences):
    start_time = time.time()
    deadline = 240.0 # Safety margin for 240s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # 1. Compute Distance Matrix
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d
            
    # 2. Neighbor Joining
    # Implement standard NJ
    # Returns Newick string and branch lengths
    
    # ... (NJ Implementation) ...
    
    # 3. Parse Newick to Tree Object
    # ...
    
    # 4. NNI Refinement
    # ...
    
    # 5. Branch Length Optimization
    # ...
    
    # 6. Return Newick
    # ...
