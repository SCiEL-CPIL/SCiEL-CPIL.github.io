import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n = 18
    # Start with K6,6,6
    # Parts: 0-5, 6-11, 12-17
    adj = np.zeros((n, n), dtype=bool)
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = adj[b][a] = True
    
    def is_k4_free(adj):
        """Check if graph is K4-free."""
        for a, b, c, d in combinations(range(n), 4):
            if adj[a][b] and adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d] and adj[c][d]:
                return False
        return True
    
    def max_triangle_free_induced(adj, timeout=5.0):
        """Exact branch and bound for max triangle-free induced subgraph size."""
        best = 0
        start = time.time()
        
        # Precompute triangle conflicts: for each pair (i,j), which vertices k form a triangle
        tri_conflict = [[set() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(n):
                    if k != i and k != j and adj[i][k] and adj[j][k]:
                        tri_conflict[i][j].add(k)
                        tri_conflict[j][i].add(k)
        
        # Branch and bound with better pruning
        def dfs(current, candidates):
            nonlocal best
            if time.time() - start > timeout:
                return
            if len(current) + len(candidates) <= best:
                return
            if not candidates:
                best = max(best, len(current))
                return
            
            # Choose vertex with maximum degree in conflict graph for better pruning
            v = candidates[0]
            remaining = candidates[1:]
            
            # Check if v can be added to current
            can_add = True
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    if v in tri_conflict[current[i]][current[j]]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                new_current = current + [v]
                # Filter remaining: remove vertices that would create triangle with any pair in new_current
                new_remaining = []
                for w in remaining:
                    ok = True
                    for u in new_current:
                        for w2 in new_current:
                            if u == w2: continue
                            if w in tri_conflict[u][w2]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        new_remaining.append(w)
                dfs(new_current, new_remaining)
            
            dfs(current, remaining)
        
        dfs([], list(range(n)))
        return best
    
    def greedy_triangle_free_size(adj, trials=50):
        """Greedy estimate of max triangle-free induced subgraph size."""
        best = 0
        for _ in range(trials):
            order = list(range(n))
            random.shuffle(order)
            current = []
            for v in order:
                can_add = True
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        if adj[v][current[i]] and adj[v][current[j]] and adj[current[i]][current[j]]:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    current.append(v)
            best = max(best, len(current))
        return best
    
    # ILP-style branch and bound: minimize max triangle-free set size
    # Variables: x_{uv} for each edge (binary), and auxiliary variables for triangle-free sets
    # This is complex, so we use a different approach:
    # Start from K6,6,6 and use local search with exact evaluation
    
    current_adj = adj.copy()
    current_score = max_triangle_free_induced(current_adj, timeout=3.0)
    best_adj = current_adj.copy()
    best_score = current_score
    
    # Local search: try flipping edges and keep improvements
    def try_edge_change(adj, u, v, add):
        """Try adding or removing edge (u,v), return new adj if K4-free, else None."""
        new_adj = adj.copy()
        if add:
            new_adj[u][v] = new_adj[v][u] = True
        else:
            new_adj[u][v] = new_adj[v][u] = False
        
        # Check K4-freeness locally
        if add:
            # Check if adding (u,v) creates a K4
            common = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    if adj[common[i]][common[j]]:
                        return None  # K4 formed
        return new_adj
    
    # Simulated annealing with exact evaluation for promising moves
    T = 2.0
    cooling = 0.995
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick random edge to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        add = not current_adj[u][v]
        new_adj = try_edge_change(current_adj, u, v, add)
        if new_adj is None:
            T *= cooling
            no_improve_count += 1
            continue
        
        # Quick greedy estimate first
        new_score_greedy = greedy_triangle_free_size(new_adj, trials=10)
        
        # If greedy suggests improvement, verify with exact
        if new_score_greedy < current_score or random.random() < np.exp(-(new_score_greedy - current_score) / max(T, 0.01)):
            new_score = max_triangle_free_induced(new_adj, timeout=2.0)
            
            if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / max(T, 0.01)):
                current_adj = new_adj
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_adj = new_adj.copy()
                    best_score = new_score
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling
        # Reset temperature if stuck
        if no_improve_count > 100:
            T = max(T, 0.5)
            no_improve_count = 0
    
    # Final verification
    final_score = max_triangle_free_induced(best_adj, timeout=10.0)
    if final_score < best_score:
        best_score = final_score
    
    # Return best
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
