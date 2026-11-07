import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    def get_score(values, edges):
        """Calculate the raw score: max(distinct_sums, distinct_prods)."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def spectral_clustering_init(target_adj):
        """
        Use spectral clustering to group vertices into clusters.
        Assign values such that vertices in the same cluster have similar values.
        This encourages collisions for edges within clusters.
        """
        # Compute eigenvectors of the Laplacian of the target graph
        # L = D - A
        degree = np.sum(target_adj, axis=1)
        L = np.diag(degree) - target_adj
        
        # We want the first few non-trivial eigenvectors
        # np.linalg.eigh is for symmetric matrices
        vals, vecs = np.linalg.eigh(L)
        
        # Sort by eigenvalue (eigh already does this for symmetric)
        # The first eigenvector is constant (0 eigenvalue).
        # Use the next k eigenvectors for clustering.
        k_clusters = 3 # Try to form 3 clusters
        
        # Take the first k non-zero eigenvectors (indices 1 to k)
        # Actually, for spectral clustering, we use the first k eigenvectors.
        # The first one is constant. We might use 1..k or 2..k+1 depending on connectedness.
        # Let's use the first k eigenvectors corresponding to the smallest non-zero eigenvalues.
        
        # Simple approach: use the first k eigenvectors (including the trivial one? No, usually exclude or normalize)
        # Standard spectral clustering:
        # 1. Compute k smallest eigenvectors of L.
        # 2. Form matrix U (n x k).
        # 3. Normalize rows of U.
        # 4. K-means on rows.
        
        U = vecs[:, :k_clusters]
        
        # Normalize rows
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1
        U_norm = U / norms
        
        # K-means
        # Initialize centroids randomly from U_norm
        centroids = U_norm[random.sample(range(n), k_clusters)].copy()
        
        for _ in range(100): # K-means iterations
            # Assign points to nearest centroid
            dists = np.linalg.norm(U_norm[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = centroids.copy()
            for i in range(k_clusters):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centroids[i] = np.mean(U_norm[mask], axis=0)
                else:
                    # Reinitialize empty cluster
                    new_centroids[i] = U_norm[random.randint(0, n-1)]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Assign values based on cluster
        # We want distinct values.
        # Strategy: Within a cluster, values should be very close to promote sum/product collisions.
        # Between clusters, values should be distinct to avoid "accidental" global collisions that might be hard to control, 
        # OR we might want some cross-cluster collisions.
        # Let's try: Cluster 0 gets small values, Cluster 1 gets medium, Cluster 2 gets large.
        # Or: All clusters get values from the same small range but offset?
        # If we use small ranges, sum collisions are likely.
        
        # Let's assign base values per cluster and add small noise.
        # To ensure distinctness, we must be careful.
        
        values = [0] * n
        used_values = set()
        
        # Define ranges for clusters to encourage intra-cluster collisions
        # Cluster 0: [1, 10]
        # Cluster 1: [11, 20]
        # Cluster 2: [21, 30]
        
        ranges = [(1, 10), (11, 20), (21, 30)]
        
        for i in range(n):
            c = labels[i]
            low, high = ranges[c]
            # Try to find a value in this range not used
            attempts = 0
            while attempts < 100:
                v = random.randint(low, high)
                if v not in used_values:
                    values[i] = v
                    used_values.add(v)
                    break
                attempts += 1
            if attempts >= 100:
                # Fallback: random distinct value
                while True:
                    v = random.randint(1, 10000)
                    if v not in used_values:
                        values[i] = v
                        used_values.add(v)
                        break
        
        return values, labels

    def joint_sa(initial_values, initial_edges, max_iter=5000, T_start=1.0, T_end=0.05):
        """
        Joint Simulated Annealing for values and edges.
        Enhanced with spectral-inspired initialization and moves.
        """
        current_values = list(initial_values)
        current_edges = set(initial_edges)
        
        current_score = get_score(current_values, current_edges)
        
        best_score = current_score
        best_values = list(current_values)
        best_edges = set(current_edges)
        
        edge_list = all_edges
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            T = T_start * (T_end / T_start) ** (i / max_iter)
            
            # Move Type Selection
            # 1. Edge Swap (Standard)
            # 2. Value Perturbation (Standard)
            # 3. Spectral Move: Swap values of two nodes in the same "cluster" (defined by similarity)
            #    or move values to be closer together.
            
            move_type = random.random()
            
            new_values = current_values
            new_edges = current_edges
            
            if move_type < 0.4:
                # Edge Move
                if len(current_edges) == 0:
                    continue
                e_out = random.choice(list(current_edges))
                while True:
                    e_in = random.choice(edge_list)
                    if e_in not in current_edges:
                        break
                new_edges = (current_edges - {e_out}) | {e_in}
                
            elif move_type < 0.7:
                # Value Move: Small step
                idx = random.randint(0, n - 1)
                delta = random.randint(-5, 5)
                new_val = current_values[idx] + delta
                if 1 <= new_val <= 10000 and new_val not in current_values:
                    new_values = list(current_values)
                    new_values[idx] = new_val
                else:
                    continue
            else:
                # Spectral-inspired Move: Swap values of two nodes
                # Hypothesis: Swapping values between nodes that are structurally similar
                # (e.g., both high degree or both in same cluster) might preserve or improve collisions.
                # For simplicity, let's just do a random swap, but weighted towards nodes with similar degrees?
                # Or just a random swap.
                idx1 = random.randint(0, n - 1)
                idx2 = random.randint(0, n - 1)
                if idx1 == idx2:
                    continue
                new_values = list(current_values)
                new_values[idx1], new_values[idx2] = new_values[idx2], new_values[idx1]

            new_score = get_score(new_values, new_edges)
            
            delta_score = new_score - current_score
            
            if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                current_values = new_values
                current_edges = new_edges
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current_values)
                    best_edges = set(current_edges)
                    
        return best_score, best_values, best_edges

    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None
    
    random.seed(42)
    
    # Generate Initial Configurations using Spectral Clustering
    # We need a "target" graph to cluster on.
    # Let's try a few target graphs:
    # 1. Complete graph (all nodes similar) -> 1 big cluster? No, Laplacian of complete graph has 0 then (n-1).
    #    Eigenvectors are orthogonal.
    # 2. Random graph.
    # 3. Clustered graph (3 cliques with weak connections).
    
    initial_configs = []
    
    # Target 1: 3 Cliques
    # Nodes 0-4, 5-9, 10-13
    target_adj_1 = np.zeros((n, n))
    for i in range(5):
        for j in range(i+1, 5):
            target_adj_1[i][j] = target_adj_1[j][i] = 1
    for i in range(5, 10):
        for j in range(i+1, 10):
            target_adj_1[i][j] = target_adj_1[j][i] = 1
    for i in range(10, 14):
        for j in range(i+1, 14):
            target_adj_1[i][j] = target_adj_1[j][i] = 1
    # Weak connections between clusters
    target_adj_1[0][5] = target_adj_1[5][0] = 0.1
    target_adj_1[9][10] = target_adj_1[10][9] = 0.1
    
    v1, _ = spectral_clustering_init(target_adj_1)
    e1 = set(random.sample(all_edges, m))
    initial_configs.append((v1, e1))
    
    # Target 2: Random Graph
    target_adj_2 = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.3:
                target_adj_2[i][j] = target_adj_2[j][i] = 1
                
    v2, _ = spectral_clustering_init(target_adj_2)
    e2 = set(random.sample(all_edges, m))
    initial_configs.append((v2, e2))
    
    # Target 3: Complete Graph (Spectral clustering on K14)
    target_adj_3 = np.ones((n, n)) - np.eye(n)
    v3, _ = spectral_clustering_init(target_adj_3)
    e3 = set(random.sample(all_edges, m))
    initial_configs.append((v3, e3))
    
    # Fallback: Random small values
    v4 = random.sample(range(1, 51), n)
    e4 = set(random.sample(all_edges, m))
    initial_configs.append((v4, e4))
    
    time_per_config = (deadline - time.time()) / len(initial_configs)
    
    for vals, edges in initial_configs:
        if time.time() > deadline:
            break
            
        max_iter = int(time_per_config * 1000)
        max_iter = min(max_iter, 50000)
        max_iter = max(max_iter, 1000)
        
        score, best_v, best_e = joint_sa(vals, edges, max_iter=max_iter, T_start=1.0, T_end=0.01)
        
        if score < best_global_score:
            best_global_score = score
            best_global_values = best_v
            best_global_edges = best_e
            
    # Fallback
    if best_global_values is None:
        best_global_values = list(range(1, 15))
        best_global_edges = set(all_edges[:35])
        best_global_score = get_score(best_global_values, best_global_edges)
        
    # Ensure types
    final_values = [int(v) for v in best_global_values]
    final_edges = [[int(u), int(v)] for u, v in best_global_edges]
    
    return {
        'values': final_values,
        'edges': final_edges
    }
