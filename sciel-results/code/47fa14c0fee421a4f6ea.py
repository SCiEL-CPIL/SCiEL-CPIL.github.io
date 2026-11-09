import time
import random
import numpy as np

def check_validity(n, adj):
    """
    Check if the graph is K4-free and has alpha < 6.
    adj is a numpy boolean matrix of shape (n, n).
    """
    # Check K4
    # For each edge (u, v), check common neighbors for edges between them
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                common = np.where(adj[u] & adj[v])[0]
                # Check if there is an edge between any pair in common
                # This forms a K4 if there is an edge (w, x) in common
                # Wait, K4 is u, v, w, x.
                # u-v is edge.
                # w, x are neighbors of u and v.
                # If w-x is an edge, then {u, v, w, x} is a K4.
                for i in range(len(common)):
                    for j in range(i + 1, len(common)):
                        if adj[common[i], common[j]]:
                            return False
    # Check alpha < 6
    # Backtracking search for independent set of size 6
    def find_independent_set(k, available):
        if k == 0:
            return True
        if len(available) < k:
            return False
        # Heuristic: pick vertex with min degree in available subgraph
        # Or just iterate
        for i in range(len(available)):
            v = available[i]
            # New available: vertices in available that are not neighbors of v and not v itself
            new_avail = [u for u in available if u != v and not adj[v, u]]
            if find_independent_set(k - 1, new_avail):
                return True
        return False

    # To speed up, sort vertices by degree
    degrees = np.sum(adj, axis=1)
    sorted_verts = np.argsort(degures)
    if not find_independent_set(6, list(sorted_verts)):
        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    best_n = 1
    best_graph = None
    
    # Try to find a valid graph for n from 35 down to 18
    for n in range(35, 17, -1):
        if time.time() > deadline:
            break
        
        # Simulated Annealing
        # Initialize graph
        # Density heuristic: For R(4,6), graphs are roughly half-density?
        # Paley(17) has degree 8, density 8/16 = 0.5.
        # Let's start with random graph p=0.5
        
        adj = np.zeros((n, n), dtype=bool)
        # Random init
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        edges_idx = np.where(mask)
        for i in range(len(edges_idx[0])):
            if random.random() < 0.5:
                u, v = edges_idx[0][i], edges_idx[1][i]
                adj[u, v] = True
                adj[v, u] = True
        
        # Helper to count K4s
        def count_k4s(adj_mat):
            count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_mat[u, v]:
                        common = np.where(adj_mat[u] & adj_mat[v])[0]
                        for i in range(len(common)):
                            for j in range(i + 1, len(common)):
                                if adj_mat[common[i], common[j]]:
                                    count += 1
            return count
        
        k4_count = count_k4s(adj)
        
        # Greedy MIS size estimator
        def greedy_mis_size(adj_mat):
            remaining = set(range(n))
            mis = 0
            while remaining:
                # Pick vertex with min degree in remaining
                min_deg = float('inf')
                pick = None
                for v in remaining:
                    deg = sum(1 for u in remaining if u != v and adj_mat[v, u])
                    if deg < min_deg:
                        min_deg = deg
                        pick = v
                mis += 1
                # Remove pick and its neighbors
                neighbors = {u for u in remaining if adj_mat[pick, u]}
                remaining.remove(pick)
                remaining -= neighbors
            return mis
        
        # SA Loop
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        steps = 0
        while time.time() < deadline - 5.0:
            if T < T_min:
                break
            
            # Pick random edge to toggle
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Calculate delta K4
            # If adding edge (u,v):
            # New K4s = edges in N(u) & N(v)
            # If removing edge (u,v):
            # Removed K4s = edges in N(u) & N(v)
            
            common = np.where(adj[u] & adj[v])[0]
            # Count edges in common
            delta_k4 = 0
            for i in range(len(common)):
                for j in range(i + 1, len(common)):
                    if adj[common[i], common[j]]:
                        delta_k4 += 1
            
            if adj[u, v]:
                # Removing edge
                new_k4_count = k4_count - delta_k4
                delta_cost_k4 = -delta_k4
            else:
                # Adding edge
                new_k4_count = k4_count + delta_k4
                delta_cost_k4 = delta_k4
            
            # Heuristic for alpha:
            # We don't have incremental alpha.
            # We will just use the K4 count for the SA step,
            # and check validity periodically.
            
            # Cost function: Minimize K4 count.
            # If K4 count is 0, we hope alpha is small.
            # If K4 count is 0, we run exact check.
            
            # Simple acceptance:
            # If new_k4_count < k4_count, accept.
            # If new_k4_count == k4_count, accept with prob exp(-delta/T) where delta is 0?
            # We need to balance.
            
            # Let's just accept if K4 count decreases or stays same.
            # And occasionally accept increases to escape local minima.
            
            if new_k4_count < k4_count:
                accept = True
            elif new_k4_count == k4_count:
                accept = True
            else:
                # Accept increase with probability
                delta = new_k4_count - k4_count
                accept = random.random() < np.exp(-delta / T)
            
            if accept:
                adj[u, v] = not adj[u, v]
                adj[v, u] = not adj[v, u]
                k4_count = new_k4_count
            
            steps += 1
            
            # Periodic check
            if steps % 100 == 0:
                if k4_count == 0:
                    # Check alpha exactly
                    if check_validity(n, adj):
                        best_n = n
                        best_graph = adj
                        return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]}
                # Also check if greedy MIS is small
                # If greedy MIS <= 5, it's a strong candidate
                # But greedy MIS can be smaller than actual alpha.
                # If greedy MIS >= 6, it's definitely invalid.
                # So if greedy MIS <= 5, we can try exact check.
                # But exact check is expensive.
                # Let's just rely on the k4_count == 0 check.
            
            T *= cooling
        
        # If we exited loop, check best state?
        # We return immediately if found.
        
    # If no graph found for n > 17, return the seed Paley 17
    # Or try to find one for n=17?
    # The seed is valid.
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
