import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s kill
    
    # 1. Initialize with Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i, j] = True
                adj[j, i] = True
    
    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges

    # Helper to find all K4s in a graph defined by adj_matrix and size size
    def find_k4s(adj, size):
        k4s = []
        for combo in combinations(range(size), 4):
            a, b, c, d = combo
            if (adj[a, b] and adj[a, c] and adj[a, d] and
                adj[b, c] and adj[b, d] and adj[c, d]):
                k4s.append(combo)
        return k4s

    # Helper to find all bar-K4s (independent sets of size 4)
    def find_bar_k4s(adj, size):
        bar_k4s = []
        for combo in combinations(range(size), 4):
            a, b, c, d = combo
            if (not adj[a, b] and not adj[a, c] and not adj[a, d] and
                not adj[b, c] and not adj[b, d] and not adj[c, d]):
                bar_k4s.append(combo)
        return bar_k4s

    current_n = n
    best_n = current_n
    best_adj = adj.copy()
    
    # We will try to extend the graph.
    # Strategy: 
    # 1. Precompute K4s and bar-K4s for the current graph.
    # 2. Try to add a new vertex by finding a valid neighborhood S.
    # 3. Use a local search approach minimizing violations to find S.
    
    while time.time() < deadline:
        # Precompute constraints for current graph
        k4s = find_k4s(adj, current_n)
        bar_k4s = find_bar_k4s(adj, current_n)
        
        # Precompute incidence lists for fast delta calculation
        k4s_containing = [[] for _ in range(current_n)]
        for k4 in k4s:
            for v in k4:
                k4s_containing[v].append(k4)
                
        bar_k4s_containing = [[] for _ in range(current_n)]
        for bk4 in bar_k4s:
            for v in bk4:
                bar_k4s_containing[v].append(bk4)

        found = False
        best_S = None
        
        # Local search for S
        # Start with random S
        S = set()
        for v in range(current_n):
            if random.random() < 0.5:
                S.add(v)
                
        # Calculate initial violations
        # Violation 1: K4 fully in S
        # Violation 2: BarK4 disjoint from S (fully in V\S)
        violations = 0
        for k4 in k4s:
            if all(v in S for v in k4):
                violations += 1
        for bk4 in bar_k4s:
            if not any(v in S for v in bk4):
                violations += 1
        
        if violations == 0:
            best_S = S.copy()
        else:
            # Local search to reduce violations
            max_iterations = 5000
            for _ in range(max_iterations):
                if time.time() > deadline - 10:
                    break
                    
                v = random.randint(0, current_n - 1)
                
                # Calculate delta violations if we flip v
                delta = 0
                
                if v in S:
                    # We are removing v from S
                    # Check K4s containing v: if other 3 were in S, removing v reduces violation by 1
                    for k4 in k4s_containing[v]:
                        others = [u for u in k4 if u != v]
                        if all(u in S for u in others):
                            delta -= 1
                    # Check BarK4s containing v: if other 3 were NOT in S, removing v increases violation by 1 (becomes disjoint)
                    for bk4 in bar_k4s_containing[v]:
                        others = [u for u in bk4 if u != v]
                        if not any(u in S for u in others):
                            delta += 1
                else:
                    # We are adding v to S
                    # Check K4s containing v: if other 3 were in S, adding v increases violation by 1
                    for k4 in k4s_containing[v]:
                        others = [u for u in k4 if u != v]
                        if all(u in S for u in others):
                            delta += 1
                    # Check BarK4s containing v: if other 3 were NOT in S, adding v reduces violation by 1 (no longer disjoint)
                    for bk4 in bar_k4s_containing[v]:
                        others = [u for u in bk4 if u != v]
                        if not any(u in S for u in others):
                            delta -= 1
                
                new_violations = violations + delta
                
                # Accept if improves or with small probability if neutral (to escape local minima)
                if new_violations < violations or (new_violations == violations and random.random() < 0.5):
                    # Apply flip
                    if v in S:
                        S.remove(v)
                    else:
                        S.add(v)
                    violations = new_violations
                    
                    if violations == 0:
                        best_S = S.copy()
                        break
                else:
                    # Revert (do nothing, as we didn't apply)
                    pass
                    
                # Random restart occasionally if stuck
                if _ % 500 == 499 and violations > 0:
                    S = set()
                    for vv in range(current_n):
                        if random.random() < 0.5:
                            S.add(vv)
                    violations = 0
                    for k4 in k4s:
                        if all(v in S for v in k4):
                            violations += 1
                    for bk4 in bar_k4s:
                        if not any(v in S for v in bk4):
                            violations += 1
                    if violations == 0:
                        best_S = S.copy()
                        break

        if best_S is not None:
            # Add new vertex with neighborhood best_S
            adj = np.resize(adj, (current_n + 1, current_n + 1))
            adj[current_n, :] = False
            adj[:, current_n] = False
            for v in best_S:
                adj[current_n, v] = True
                adj[v, current_n] = True
            current_n += 1
            found = True
            best_n = current_n
            best_adj = adj.copy()
        else:
            # Failed to add vertex
            # Try a few more quick random attempts
            quick_found = False
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                S = set()
                for v in range(current_n):
                    if random.random() < 0.5:
                        S.add(v)
                
                # Quick check
                valid = True
                for k4 in k4s:
                    if all(v in S for v in k4):
                        valid = False
                        break
                if valid:
                    for bk4 in bar_k4s:
                        if not any(v in S for v in bk4):
                            valid = False
                            break
                
                if valid:
                    adj = np.resize(adj, (current_n + 1, current_n + 1))
                    adj[current_n, :] = False
                    adj[:, current_n] = False
                    for v in S:
                        adj[current_n, v] = True
                        adj[v, current_n] = True
                    current_n += 1
                    quick_found = True
                    best_n = current_n
                    best_adj = adj.copy()
                    break
            
            if not quick_found:
                break

    edges = get_edges(best_adj, best_n)
    return best_n, edges

# Test run
if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
