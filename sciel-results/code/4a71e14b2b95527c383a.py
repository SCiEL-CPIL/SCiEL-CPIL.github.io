import numpy as np
import itertools
import random

def _largest_tf_induced(adj):
    """Find the largest set of vertices inducing a triangle-free subgraph."""
    n = len(adj)
    vertices = list(range(n))
    best = 0
    
    # Use a simple branch and bound
    # We want to find the maximum subset with no triangles
    # Triangle-free means no three vertices all mutually connected
    
    def has_triangle(subset):
        if len(subset) < 3:
            return False
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                for k in range(j+1, len(subset)):
                    u, v, w = subset[i], subset[j], subset[k]
                    if adj[u][v] and adj[v][w] and adj[u][w]:
                        return True
        return False
    
    # Backtracking search for maximum triangle-free set
    def backtrack(remaining, current):
        nonlocal best
        if len(remaining) + len(current) <= best:
            return
        if not remaining:
            if len(current) > best:
                best = len(current)
            return
        
        # Try adding each vertex from remaining
        for i, v in enumerate(remaining):
            new_current = current + [v]
            if has_triangle(new_current):
                continue
            remaining_new = remaining[i+1:]
            backtrack(remaining_new, new_current)
        
        # Also try not adding any more (current is the answer)
        if len(current) > best:
            best = len(current)
    
    backtrack(vertices, [])
    return best

def _is_k4_free(adj):
    """Check if graph is K4-free."""
    n = len(adj)
    for combo in itertools.combinations(range(n), 4):
        u, v, w, x = combo
        if adj[u][v] and adj[u][w] and adj[u][x] and \
           adj[v][w] and adj[v][x] and adj[w][x]:
            return False
    return True

def _make_k4_free(adj):
    """Remove edges to make graph K4-free."""
    n = len(adj)
    changed = True
    while changed:
        changed = False
        for combo in itertools.combinations(range(n), 4):
            u, v, w, x = combo
            if adj[u][v] and adj[u][w] and adj[u][x] and \
               adj[v][w] and adj[v][x] and adj[w][x]:
                # Remove one random edge from this K4
                edges_in_k4 = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                edge = random.choice(edges_in_k4)
                adj[edge[0]][edge[1]] = 0
                adj[edge[1]][edge[0]] = 0
                changed = True
    return adj

def construct():
    n = 18
    random.seed(42)
    np.random.seed(42)
    
    # Spectral optimization approach
    # Represent graph as continuous adjacency matrix
    # Optimize to minimize triangle-free induced subgraph size while keeping K4-free
    
    # Start with random continuous adjacency matrix in [0, 1]
    A = np.random.rand(n, n)
    np.fill_diagonal(A, 0)
    A = (A + A.T) / 2  # Symmetrize
    
    # Gradient descent to maximize density (minimize alpha_TF)
    # while penalizing K4 subgraphs
    
    lr = 0.05
    k4_penalty = 0.1
    density_weight = 1.0
    
    for iteration in range(500):
        # Compute gradient
        grad = np.zeros_like(A)
        
        # Density term: want to maximize total edges
        # Gradient of sum of A is 1 for each off-diagonal
        for i in range(n):
            for j in range(i+1, n):
                grad[i][j] += density_weight
                grad[j][i] += density_weight
        
        # K4 penalty: for each K4, add penalty if all edges are present
        for combo in itertools.combinations(range(n), 4):
            u, v, w, x = combo
            edges_k4 = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
            # Product of edge weights - if all close to 1, penalty is high
            prod = 1.0
            for (a, b) in edges_k4:
                prod *= A[a][b]
            # Gradient of product
            for (a, b) in edges_k4:
                if A[a][b] > 1e-6:
                    g = prod / A[a][b]
                    grad[a][b] -= k4_penalty * g
                    grad[b][a] -= k4_penalty * g
        
        # Update
        A += lr * grad
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        A = np.clip(A, 0, 1)
    
    # Round to binary
    adj = (A > 0.5).astype(int)
    np.fill_diagonal(adj, 0)
    adj = (adj + adj.T) // 2
    
    # Make K4-free
    adj = _make_k4_free(adj)
    
    # Local improvement: try to add edges that don't create K4
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] == 0:
                    # Check if adding (u,v) creates a K4
                    creates_k4 = False
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[u][w] and adj[v][w]:
                            for x in range(w+1, n):
                                if x == u or x == v:
                                    continue
                                if adj[u][x] and adj[v][x] and adj[w][x]:
                                    creates_k4 = True
                                    break
                        if creates_k4:
                            break
                    if not creates_k4:
                        # Check if this improves the objective
                        old_score = _largest_tf_induced(adj)
                        adj[u][v] = 1
                        adj[v][u] = 1
                        new_score = _largest_tf_induced(adj)
                        if new_score < old_score:
                            improved = True
                        else:
                            adj[u][v] = 0
                            adj[v][u] = 0
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edges.append([u, v])
    
    return edges
