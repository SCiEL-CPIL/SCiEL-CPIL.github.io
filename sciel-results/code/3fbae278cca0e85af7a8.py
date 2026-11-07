"""Rewritten program for small exact biclique partitions of a random graph."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    # Build adjacency as a set of frozensets for fast lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # Adjacency list for quick neighbor lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 85.0  # 85s deadline, 5s safety margin
    best_result = None
    best_count = float('inf')
    
    def is_valid_biclique(L, R, remaining_edges):
        """Check if (L, R) forms a valid biclique where all cross edges are in remaining_edges."""
        for u in L:
            for v in R:
                if u == v:
                    return False
                e = (min(u, v), max(u, v))
                if e not in remaining_edges:
                    return False
        return True
    
    def greedy_partition(random_seed=None):
        """Greedy biclique partition with randomization."""
        if random_seed is not None:
            random.seed(random_seed)
        
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            # Pick a random edge to start
            edge = random.choice(list(remaining))
            u, v = edge
            
            # Build initial biclique
            L = [u]
            R = [v]
            
            # Expand greedily
            changed = True
            while changed:
                changed = False
                
                # Try to add vertices to L
                for u_prime in range(n):
                    if u_prime not in L and u_prime not in R:
                        can_add = True
                        for r in R:
                            e = (min(u_prime, r), max(u_prime, r))
                            if e not in remaining:
                                can_add = False
                                break
                        if can_add:
                            L.append(u_prime)
                            changed = True
                
                # Try to add vertices to R
                for v_prime in range(n):
                    if v_prime not in R and v_prime not in L:
                        can_add = True
                        for l in L:
                            e = (min(l, v_prime), max(l, v_prime))
                            if e not in remaining:
                                can_add = False
                                break
                        if can_add:
                            R.append(v_prime)
                            changed = True
            
            # Remove all edges in this biclique from remaining
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        remaining.discard(e)
            
            bicliques.append([L, R])
        
        return bicliques
    
    # Try multiple random restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        result = greedy_partition(random_seed=i)
        
        # Validate
        covered_edges = set()
        for L, R in result:
            for u in L:
                for v in R:
                    if u != v:
                        e = (min(u, v), max(u, v))
                        covered_edges.add(e)
        
        if covered_edges == edge_set and len(covered_edges) == len(edge_set):
            count = len(result)
            if count < best_count:
                best_count = count
                best_result = [[list(L), list(R)] for L, R in result]
                if best_count <= 15:  # Good enough, stop early
                    break
    
    if best_result is None:
        # Fallback: star decomposition
        best_result = []
        for u in range(n):
            neighbors = [v for v in adj[u] if v > u]
            if neighbors:
                best_result.append([[u], neighbors])
    
    # Ensure types are correct
    final_result = []
    for L, R in best_result:
        L_clean = [int(x) for x in L]
        R_clean = [int(x) for x in R]
        final_result.append([L_clean, R_clean])
    
    return final_result
# EVOLVE-BLOCK-END
