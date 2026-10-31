import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    vectors = np.array(candidates, dtype=np.int64)
    N = len(vectors)
    
    if N == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Compute norms
    norms2 = (vectors ** 2).sum(axis=1)
    max_norm2 = norms2.max()
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # dist^2(i,j) = norms2[i] + norms2[j] - 2*dots[i,j]
    dist2 = norms2[:, None] + norms2[None, :] - 2 * dots
    np.fill_diagonal(dist2, 0)
    
    min_dist2 = dist2.min()
    
    if min_dist2 >= max_norm2:
        # All vectors are valid together
        return vectors
    
    # Need to find a large subset with min dist^2 >= max_norm2
    # Conflict: dist2[i,j] < max_norm2
    conflicts = dist2 < max_norm2
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency list for efficient MIS search
    # adj[i] = set of nodes conflicting with i
    adj = [set() for _ in range(N)]
    for i in range(N):
        neighbors = np.where(conflicts[i])[0]
        adj[i] = set(neighbors.tolist())
    
    # Tomita's branch-and-bound for Maximum Independent Set
    # Using color coding for upper bound
    
    best_size = 0
    best_set = []
    
    def expand(P, R, t):
        """Branch and bound: P = candidate set, R = current independent set, t = time deadline"""
        nonlocal best_size, best_set
        
        if time.time() > t:
            return False  # timeout signal
        
        if len(P) + len(R) <= best_size:
            return True
        
        if len(P) == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_set = list(R)
            return True
        
        # Color coding for upper bound
        # Order P by degree (descending) for better pruning
        P_list = list(P)
        
        # Simple greedy coloring to get upper bound
        # Color each vertex in P with the minimum possible color
        colors = [0] * len(P_list)
        color_classes = []  # color_classes[c] = set of indices in P_list with color c
        
        # Greedy coloring: process vertices in order, assign smallest available color
        vertex_colors = {}
        next_color = 0
        for idx in range(len(P_list)):
            v = P_list[idx]
            # Find smallest color not used by neighbors of v that are already colored
            used_colors = set()
            for u_idx in range(idx):
                u = P_list[u_idx]
                if u in adj[v]:
                    used_colors.add(vertex_colors[u_idx])
            c = 0
            while c in used_colors:
                c += 1
            vertex_colors[idx] = c
            if c + 1 > len(color_classes):
                color_classes.append(set())
            color_classes[c].add(idx)
            next_color = max(next_color, c + 1)
        
        num_colors = len(color_classes)
        
        # Upper bound: len(R) + num_colors
        if len(R) + num_colors <= best_size:
            return True
        
        # Branch: try each color class in reverse order (largest color first)
        # For each color class, pick a vertex and branch
        for c in range(num_colors - 1, -1, -1):
            if time.time() > t:
                return False
            
            if len(R) + c + 1 <= best_size:
                break
            
            # Pick a vertex from color class c
            # Heuristic: pick vertex with highest degree in P
            class_vertices = list(color_classes[c])
            # Sort by degree in P (descending)
            class_vertices.sort(key=lambda idx: len(adj[P_list[idx]] & P), reverse=True)
            
            for idx in class_vertices:
                if time.time() > t:
                    return False
                
                v = P_list[idx]
                
                # New candidate set: P minus v and all neighbors of v
                new_P = set()
                for u_idx in range(len(P_list)):
                    u = P_list[u_idx]
                    if u != v and u not in adj[v]:
                        new_P.add(u)
                
                # Recurse
                timeout = expand(new_P, R | {v}, t)
                if timeout:
                    return True
                
                if len(new_P) + len(R) + 1 <= best_size:
                    break
        
        return True
    
    # Initial call with all vertices as candidates
    all_vertices = set(range(N))
    expand(all_vertices, set(), deadline)
    
    if best_set:
        result_vectors = vectors[best_set]
        return result_vectors
    
    # Fallback: return at least one vector
    return vectors[:1]
