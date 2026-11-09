"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 85  # safety margin
    
    # Build edge index map: (u,v) -> arrival index (1-based)
    edge_index = {}
    for i, (u, v) in enumerate(P['edge_order']):
        if u > v:
            u, v = v, u
        edge_index[(u, v)] = i + 1
    
    # Build adjacency list sorted by arrival index
    adj = [[] for _ in range(n)]
    for (u, v), idx in edge_index.items():
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    for i in range(n):
        adj[i].sort(key=lambda x: x[1])
    
    # Precompute edge weights for a given cycle
    def cycle_score(cycle):
        """Compute the max arrival index of edges in the cycle."""
        max_idx = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if u > v:
                u, v = v, u
            idx = edge_index[(u, v)]
            if idx > max_idx:
                max_idx = idx
        return max_idx
    
    # Greedy construction: build a Hamilton cycle using earliest edges
    def greedy_build():
        """Build a Hamilton cycle greedily using earliest available edges."""
        used_vertices = set()
        cycle = []
        
        # Start with vertex 0
        current = 0
        used_vertices.add(0)
        cycle.append(0)
        
        for _ in range(n - 1):
            # Find the earliest edge from current to an unused vertex
            best_next = None
            best_idx = float('inf')
            for (v, idx) in adj[current]:
                if v not in used_vertices and idx < best_idx:
                    best_idx = idx
                    best_next = v
            
            if best_next is None:
                # Fallback: pick any unused vertex with smallest index
                for v in range(n):
                    if v not in used_vertices:
                        best_next = v
                        break
            
            current = best_next
            used_vertices.add(current)
            cycle.append(current)
        
        return cycle
    
    # 2-opt improvement
    def two_opt(cycle, max_iterations=10000):
        """Improve cycle using 2-opt moves."""
        best_score = cycle_score(cycle)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue  # This would be the same cycle
                    # Try reversing segment i+1..j
                    new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                    new_score = cycle_score(new_cycle)
                    if new_score < best_score:
                        cycle = new_cycle
                        best_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        return cycle, best_score
    
    # Or-opt: move a segment of 1-3 vertices
    def or_opt(cycle, max_iterations=5000):
        """Improve cycle using or-opt moves."""
        best_score = cycle_score(cycle)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            
            for seg_len in [1, 2, 3]:
                if improved:
                    break
                for i in range(n):
                    if time.time() > deadline:
                        break
                    if i + seg_len >= n:
                        continue
                    # Extract segment
                    segment = cycle[i:i+seg_len]
                    # Remove segment
                    remaining = cycle[:i] + cycle[i+seg_len:]
                    # Try inserting at different positions
                    for j in range(len(remaining)):
                        if j == i or (i > 0 and j == i - 1):
                            continue  # Same position
                        new_cycle = remaining[:j] + segment + remaining[j:]
                        new_score = cycle_score(new_cycle)
                        if new_score < best_score:
                            cycle = new_cycle
                            best_score = new_score
                            improved = True
                            break
                    if improved:
                        break
        
        return cycle, best_score
    
    # Start with greedy construction
    best_cycle = greedy_build()
    best_score = cycle_score(best_cycle)
    
    # Apply 2-opt
    best_cycle, best_score = two_opt(best_cycle)
    
    # Apply or-opt
    best_cycle, best_score = or_opt(best_cycle)
    
    # Try multiple random restarts if time permits
    rng = random.Random(614)
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        # Create a random initial cycle
        random_cycle = list(range(n))
        rng.shuffle(random_cycle)
        
        # Improve it
        improved_cycle, improved_score = two_opt(random_cycle, max_iterations=5000)
        improved_cycle, improved_score = or_opt(improved_cycle, max_iterations=2000)
        
        if improved_score < best_score:
            best_cycle = improved_cycle
            best_score = improved_score
    
    # Final polish with more iterations
    best_cycle, best_score = two_opt(best_cycle, max_iterations=20000)
    best_cycle, best_score = or_opt(best_cycle, max_iterations=10000)
    
    # Return the cycle as a permutation
    return list(best_cycle)
# EVOLVE-BLOCK-END
