"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    n = 14
    m = 35
    max_val = 10000

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def spectral_partition(values, n_clusters=3):
        """
        Use spectral clustering to partition vertices into clusters.
        Vertices with similar values should be in the same cluster.
        """
        # Build a similarity matrix based on value differences
        sim = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Similar if values are close (relative)
                    diff = abs(values[i] - values[j])
                    base = max(values[i], values[j], 1)
                    sim[i][j] = 1.0 - min(diff / base, 1.0)
        
        # Build Laplacian: L = D - W
        degree = [sum(sim[i]) for i in range(n)]
        lap = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    lap[i][j] = degree[i]
                else:
                    lap[i][j] = -sim[i][j]
        
        # Power iteration to find Fiedler vector (2nd eigenvector)
        # First find the principal eigenvector (should be constant)
        v = [1.0/n] * n
        for _ in range(100):
            new_v = [0.0]*n
            for i in range(n):
                for j in range(n):
                    new_v[i] += lap[i][j] * v[j]
            # Normalize
            norm = sum(x*x for x in new_v)**0.5
            if norm > 0:
                new_v = [x/norm for x in new_v]
            v = new_v
        
        # Remove mean (deflation) to find 2nd eigenvector
        mean_v = sum(v)/n
        v2 = [v[i] - mean_v for i in range(n)]
        
        # Power iteration on deflated Laplacian
        v2 = [1.0/n] * n
        for _ in range(100):
            new_v2 = [0.0]*n
            for i in range(n):
                for j in range(n):
                    new_v2[i] += lap[i][j] * v2[j]
            # Deflate: remove component along first eigenvector
            dot = sum(v2[i]*v[i] for i in range(n))
            v2 = [v2[i] - dot*v[i] for i in range(n)]
            norm = sum(x*x for x in v2)**0.5
            if norm > 0:
                v2 = [x/norm for x in v2]
        
        # K-means style partition based on Fiedler vector
        # For n_clusters, use thresholding
        sorted_indices = sorted(range(n), key=lambda i: v2[i])
        cluster_size = n // n_clusters
        clusters = []
        for c in range(n_clusters):
            start_idx = c * cluster_size
            if c == n_clusters - 1:
                end_idx = n
            else:
                end_idx = start_idx + cluster_size
            clusters.append(sorted_indices[start_idx:end_idx])
        
        return clusters

    def generate_clustered_values(rng, n_clusters=3):
        """
        Generate values where vertices in the same cluster have very similar values,
        so that intra-cluster edges produce few distinct sums and products.
        """
        values = [0]*n
        cluster_sizes = [n // n_clusters] * n_clusters
        cluster_sizes[-1] += n % n_clusters
        
        cluster_centers = []
        for c in range(n_clusters):
            # Choose a center value for this cluster
            center = rng.randint(10, max_val // n_clusters)
            cluster_centers.append(center)
        
        # Assign values: within each cluster, values are very close
        for c in range(n_clusters):
            size = cluster_sizes[c]
            base = cluster_centers[c]
            for i in range(size):
                # Small perturbation to keep values distinct
                offset = rng.randint(1, 5)
                if i % 2 == 0:
                    val = base + offset
                else:
                    val = base - offset
                if val < 1:
                    val = base + offset + 1
                if val > max_val:
                    val = base - offset - 1
                if val < 1:
                    val = 1
                if val > max_val:
                    val = max_val
                # Ensure uniqueness
                while val in values:
                    val = rng.randint(1, max_val)
                values[i] = val
        
        # Ensure all values are distinct and valid
        values = list(set(values))
        while len(values) < n:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:n]
        
        return values

    def intra_cluster_edges(clusters):
        """
        Select edges that are primarily intra-cluster.
        This maximizes the chance of sum/product collisions.
        """
        edges = []
        # First, add all intra-cluster edges
        for cluster in clusters:
            for u, v in combinations(cluster, 2):
                if len(edges) < m:
                    edges.append([u, v])
        
        # If we need more edges, add inter-cluster edges
        if len(edges) < m:
            all_pairs = list(combinations(range(n), 2))
            edge_set = set(tuple(sorted(e)) for e in edges)
            for u, v in all_pairs:
                if len(edges) >= m:
                    break
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    edges.append([u, v])
                    edge_set.add((u, v))
        
        return edges[:m]

    def local_search_edges(values, edges, rng, max_iters=3000):
        """
        Local search: try swapping edges to reduce distinct sums and products.
        """
        edges_set = set(tuple(sorted(e)) for e in edges)
        current_edges = [list(e) for e in edges]
        
        # Count occurrences of each sum and product
        sum_counts = {}
        prod_counts = {}
        for u, v in current_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        best_score = max(distinct_sums, distinct_prods)
        
        all_pairs = list(combinations(range(n), 2))
        non_edges = [p for p in all_pairs if tuple(sorted(p)) not in edges_set]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try removing a random edge and adding a random non-edge
            idx = rng.randint(0, m - 1)
            old_u, old_v = current_edges[idx]
            old_s = values[old_u] + values[old_v]
            old_p = values[old_u] * values[old_v]
            
            # Try several candidate replacements
            candidates = rng.sample(non_edges, min(40, len(non_edges)))
            
            best_new_score = best_score
            best_new_edge = None
            best_new_sum_counts = None
            best_new_prod_counts = None
            
            for new_u, new_v in candidates:
                if (new_u, new_v) in edges_set or (new_v, new_u) in edges_set:
                    continue
                
                new_s = values[new_u] + values[new_v]
                new_p = values[new_u] * values[new_v]
                
                # Simulate removal and addition
                new_sum_counts = dict(sum_counts)
                new_prod_counts = dict(prod_counts)
                
                # Remove old edge
                new_sum_counts[old_s] -= 1
                if new_sum_counts[old_s] == 0:
                    del new_sum_counts[old_s]
                new_prod_counts[old_p] -= 1
                if new_prod_counts[old_p] == 0:
                    del new_prod_counts[old_p]
                
                # Add new edge
                new_sum_counts[new_s] = new_sum_counts.get(new_s, 0) + 1
                new_prod_counts[new_p] = new_prod_counts.get(new_p, 0) + 1
                
                new_distinct_sums = len(new_sum_counts)
                new_distinct_prods = len(new_prod_counts)
                new_score = max(new_distinct_sums, new_distinct_prods)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_edge = (new_u, new_v)
                    best_new_sum_counts = new_sum_counts
                    best_new_prod_counts = new_prod_counts
            
            if best_new_edge is not None:
                # Apply the improvement
                current_edges[idx] = list(best_new_edge)
                edges_set.discard(tuple(sorted((old_u, old_v))))
                edges_set.add(tuple(sorted(best_new_edge)))
                
                # Update non_edges list
                non_edges.remove(tuple(sorted(best_new_edge)))
                non_edges.append(tuple(sorted((old_u, old_v))))
                
                sum_counts = best_new_sum_counts
                prod_counts = best_new_prod_counts
                distinct_sums = len(sum_counts)
                distinct_prods = len(prod_counts)
                best_score = best_new_score
                improved = True
        
        return current_edges, best_score

    def optimize_values_clustered(values, edges, rng, max_iters=200):
        """
        Try adjusting values within their cluster to improve collisions.
        """
        best_values = list(values)
        best_score = evaluate(best_values, edges)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            i = rng.randint(0, n - 1)
            old_val = best_values[i]
            
            # Try small perturbations
            for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                new_val = old_val + delta
                if new_val < 1 or new_val > max_val:
                    continue
                if new_val in best_values:
                    continue
                
                new_values = list(best_values)
                new_values[i] = new_val
                
                new_score = evaluate(new_values, edges)
                if new_score < best_score:
                    best_values = new_values
                    best_score = new_score
                    break
        
        return best_values, best_score

    # Main search loop
    rng = random.Random(42)
    best_result = None
    best_score = float('inf')
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Try different numbers of clusters
        n_clusters = rng.choice([2, 3, 4])
        
        # Generate clustered values
        values = generate_clustered_values(rng, n_clusters)
        
        # Get spectral partition
        clusters = spectral_partition(values, n_clusters)
        
        # Select intra-cluster edges
        edges = intra_cluster_edges(clusters)
        
        # Local search on edges
        edges, score = local_search_edges(values, edges, rng, max_iters=1500)
        
        # Optimize values within clusters
        values, score = optimize_values_clustered(values, edges, rng, max_iters=150)
        
        # Re-do edge selection after value changes
        clusters = spectral_partition(values, n_clusters)
        edges = intra_cluster_edges(clusters)
        edges, score = local_search_edges(values, edges, rng, max_iters=1000)
        
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
            if best_score <= 8:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Final validation
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in values)
    assert len(edges) == m
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n and 0 <= e[1] < n
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
