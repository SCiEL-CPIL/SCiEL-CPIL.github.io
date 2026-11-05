import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 31
    m = 91
    
    # 1. Construct PG(2,5) points
    points = []
    for a in range(5):
        for b in range(5):
            points.append((1, a, b))
    for b in range(5):
        points.append((0, 1, b))
    points.append((0, 0, 1))
    
    # 2. Construct adjacency matrix for polarity graph (90 edges, C4-free)
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Helper: count C4s efficiently using matrix multiplication
    def count_c4s(adj):
        # C4 count = (1/4) * sum over all pairs (u,v) of C(common_neighbors(u,v), 2)
        # common_neighbors(u,v) = (adj @ adj)[u,v] - 1 if u~v else (adj @ adj)[u,v]
        A = adj.astype(np.int32)
        A2 = A @ A
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = A2[u, v]
                if adj[u, v] == 1:
                    cn -= 1  # subtract the edge itself
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 2
    
    # Helper: compute delta in C4 count when adding edge (i,j)
    def delta_add(adj, i, j):
        # New C4s = number of length-3 paths between i and j
        # = sum over w in N(i) of |N(w) ∩ N(j)|
        # But we need to be careful: this counts each C4 once
        count = 0
        Ni = np.where(adj[i] == 1)[0]
        Nj = np.where(adj[j] == 1)[0]
        for w in Ni:
            # Count neighbors of w that are also in Nj
            count += np.sum(adj[w, Nj])
        return count // 2
    
    # Helper: compute delta in C4 count when removing edge (i,j)
    def delta_remove(adj, i, j):
        # Removing edge (i,j) destroys C4s that use this edge
        # = number of length-3 paths between i and j (excluding the edge itself)
        # Same formula as delta_add but we need to account for the edge being present
        count = 0
        Ni = np.where(adj[i] == 1)[0]
        Nj = np.where(adj[j] == 1)[0]
        # Exclude i and j from consideration
        for w in Ni:
            if w == j:
                continue
            count += np.sum(adj[w, Nj])
            # Subtract 1 if w is adjacent to j (which it is, since w in Nj)
            # Wait, we need to count paths i-w-x-j where w in N(i), x in N(j), w-x in E
            # and w != j, x != i
        # Let me redo this more carefully
        # C4s containing edge (i,j): i-j-x-y-i where x in N(j)\{i}, y in N(i)\{j}, x-y in E
        count = 0
        for x in Nj:
            if x == i:
                continue
            for y in Ni:
                if y == j:
                    continue
                if adj[x, y] == 1:
                    count += 1
        return count // 2
    
    # Helper: full delta for a swap (remove (u,v), add (i,j))
    def delta_swap(adj, u, v, i, j):
        # This is complex; let's just recompute
        old_c4s = count_c4s(adj)
        adj[u, v] = 0
        adj[v, u] = 0
        adj[i, j] = 1
        adj[j, i] = 1
        new_c4s = count_c4s(adj)
        # Revert
        adj[u, v] = 1
        adj[v, u] = 1
        adj[i, j] = 0
        adj[j, i] = 0
        return new_c4s - old_c4s, old_c4s, new_c4s
    
    # 3. Find best edge to add to get 91 edges
    best_delta = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 0:
                d = delta_add(adj, i, j)
                if d < best_delta:
                    best_delta = d
                    best_edge = (i, j)
                    
    if best_edge:
        i, j = best_edge
        adj[i, j] = 1
        adj[j, i] = 1
    
    current_c4s = count_c4s(adj)
    best_c4s = current_c4s
    best_adj = adj.copy()
    
    # Get edge and non-edge lists
    def get_edge_lists(adj):
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        return edges, non_edges
    
    edges, non_edges = get_edge_lists(adj)
    
    # 4. Simulated Annealing with 1-swap and 2-swap moves
    T = 5.0  # Initial temperature
    cooling = 0.9995
    min_T = 0.01
    
    iteration = 0
    max_iterations = 200000
    
    while iteration < max_iterations and time.time() < deadline - 3:
        iteration += 1
        
        if time.time() > deadline - 3:
            break
        
        # Choose move type: 70% 1-swap, 30% 2-swap
        if random.random() < 0.7 or len(non_edges) < 2:
            # 1-swap: remove one edge, add one non-edge
            e_idx = random.randint(0, len(edges)-1)
            u, v = edges[e_idx]
            ne_idx = random.randint(0, len(non_edges)-1)
            i, j = non_edges[ne_idx]
            
            # Apply
            adj[u, v] = 0
            adj[v, u] = 0
            adj[i, j] = 1
            adj[j, i] = 1
            
            new_c4s = count_c4s(adj)
            delta = new_c4s - current_c4s
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                current_c4s = new_c4s
                edges[e_idx] = (i, j)
                non_edges[ne_idx] = (u, v)
                if current_c4s < best_c4s:
                    best_c4s = current_c4s
                    best_adj = adj.copy()
            else:
                # Revert
                adj[i, j] = 0
                adj[j, i] = 0
                adj[u, v] = 1
                adj[v, u] = 1
            
            T *= cooling
        else:
            # 2-swap: remove two edges, add two non-edges
            e_idx1 = random.randint(0, len(edges)-1)
            e_idx2 = random.randint(0, len(edges)-1)
            if e_idx1 == e_idx2:
                e_idx2 = (e_idx2 + 1) % len(edges)
            
            u1, v1 = edges[e_idx1]
            u2, v2 = edges[e_idx2]
            
            ne_idx1 = random.randint(0, len(non_edges)-1)
            ne_idx2 = random.randint(0, len(non_edges)-1)
            if ne_idx1 == ne_idx2:
                ne_idx2 = (ne_idx2 + 1) % len(non_edges)
            
            i1, j1 = non_edges[ne_idx1]
            i2, j2 = non_edges[ne_idx2]
            
            # Check no conflicts
            if (i1, j1) == (u1, v1) or (i1, j1) == (u2, v2) or \
               (i2, j2) == (u1, v1) or (i2, j2) == (u2, v2):
                T *= cooling
                continue
            
            # Apply
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            adj[i1, j1] = 1
            adj[j1, i1] = 1
            adj[i2, j2] = 1
            adj[j2, i2] = 1
            
            new_c4s = count_c4s(adj)
            delta = new_c4s - current_c4s
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                current_c4s = new_c4s
                edges[e_idx1] = (i1, j1)
                edges[e_idx2] = (i2, j2)
                non_edges[ne_idx1] = (u1, v1)
                non_edges[ne_idx2] = (u2, v2)
                if current_c4s < best_c4s:
                    best_c4s = current_c4s
                    best_adj = adj.copy()
            else:
                # Revert
                adj[i1, j1] = 0
                adj[j1, i1] = 0
                adj[i2, j2] = 0
                adj[j2, i2] = 0
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                adj[u2, v2] = 1
                adj[v2, u2] = 1
            
            T *= cooling
        
        # Restart if stuck for too long
        if iteration % 5000 == 0 and T <= min_T:
            # Perturb best solution
            adj = best_adj.copy()
            current_c4s = count_c4s(adj)
            edges, non_edges = get_edge_lists(adj)
            T = 2.0
            
            # Randomly swap a few edges
            for _ in range(random.randint(2, 5)):
                e_idx = random.randint(0, len(edges)-1)
                ne_idx = random.randint(0, len(non_edges)-1)
                u, v = edges[e_idx]
                i, j = non_edges[ne_idx]
                adj[u, v] = 0
                adj[v, u] = 0
                adj[i, j] = 1
                adj[j, i] = 1
                edges[e_idx] = (i, j)
                non_edges[ne_idx] = (u, v)
            current_c4s = count_c4s(adj)
    
    # Use the best solution found
    adj = best_adj
    
    # Return edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 1:
                result.append([i, j])
                
    return result
