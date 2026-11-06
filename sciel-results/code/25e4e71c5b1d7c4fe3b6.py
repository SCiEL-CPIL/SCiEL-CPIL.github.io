# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def _is_regular_subgraph(adj, subset, n):
    """Check if the induced subgraph on 'subset' is regular (all same degree within subset)."""
    k = len(subset)
    if k < 6:
        return False
    degrees = [0] * k
    subset_set = set(subset)
    for i, u in enumerate(subset):
        for v in subset:
            if v != u and adj[u][v]:
                degrees[i] += 1
    first = degrees[0]
    for d in degrees[1:]:
        if d != first:
            return False
    return True

def _check_graph_valid(adj, n):
    """Check if graph has no regular induced subgraph on 6+ vertices."""
    # Precompute all subsets of size 6
    # We can be smarter: check all 6-subsets. If any 6-subset is regular, invalid.
    # Actually if a k>6 subset is regular, every 6-subset of it is NOT necessarily regular.
    # Wait: if a k-vertex induced subgraph is regular of degree d, then every vertex has degree d within that k-set.
    # The induced subgraph on any 6 vertices of that set does NOT have to be regular.
    # So checking only 6-subsets is NOT sufficient.
    
    # We need to check all subsets of size >= 6.
    # This is expensive. Let's check sizes 6, 7, ..., n.
    # But we can prune: if a subset is not regular, its supersets might still be regular? No.
    # Actually, we just need to check all subsets of size 6 to n.
    # For n up to 22, sum_{k=6}^{22} C(22,k) is large (~2^22 - sum_{k=0}^{5} C(22,k) ~ 4M).
    # That's too slow in pure Python for many evaluations.
    
    # Better approach: check all 6-subsets. If a 6-subset is regular, it's invalid.
    # But as noted, a larger regular subgraph doesn't imply a regular 6-subset.
    # However, if there IS a regular induced subgraph on k>=6 vertices, then... 
    # Let's think again. The condition is "No set of 6 or more vertices may induce a regular subgraph".
    # So we must check all subsets of size >= 6.
    
    # Optimized check: For each subset S of size >= 6, check if regular.
    # We can iterate by size.
    # For performance, let's use a helper that checks a specific subset.
    
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            subset_list = list(subset)
            k_size = len(subset_list)
            degrees = [0] * k_size
            for i, u in enumerate(subset_list):
                for j, v in enumerate(subset_list):
                    if i != j and adj[u][v]:
                        degrees[i] += 1
            first = degrees[0]
            regular = True
            for d in degrees[1:]:
                if d != first:
                    regular = False
                    break
            if regular:
                return False
    return True

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Try to find a graph with n vertices, starting from small n and increasing.
    # We'll use a local search / randomized construction approach.
    
    best_n = 0
    best_edges = []
    
    # Start with known small valid graphs and try to extend them.
    # Or start random and try to increase n.
    
    # Strategy: 
    # 1. Try n = 6 to 12 with random graphs and local improvement.
    # 2. For each n, try to construct a valid graph.
    
    # Actually, the known result is that graphs with 20 vertices are known.
    # We should aim to find one efficiently.
    
    # Let's use a heuristic: start with a random graph on n vertices, 
    # then flip edges to eliminate regular subgraphs.
    
    # But checking all subsets is expensive. Let's focus on n=6 first.
    
    # For n=6: a 6-vertex regular subgraph means the whole graph must be non-regular.
    # So for n=6, we just need a non-regular graph. Easy.
    
    # Let's build up. Start with n=6, find a valid graph.
    # Then add a vertex and adjust edges.
    
    # Simpler: use random search with increasing n.
    
    random.seed(42)
    
    # Precompute: for a given n, how many 6-subsets? C(n,6).
    # For n=20, C(20,6) = 38760. For each, checking regularity is O(36). ~1.4M ops. Fast.
    # But we also need 7..20 subsets. Total subsets ~ 2^20 = 1M. Still manageable in Python if optimized.
    
    # Let's try a construction approach:
    # Start with empty graph on n vertices.
    # Add edges one by one, checking validity after each addition.
    # This is too slow.
    
    # Alternative: Random graph with p=0.5, then fix conflicts.
    
    # Let's try n from 15 to 22, and for each, do random restarts with local search.
    
    for target_n in range(18, 22):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_restarts = 50 if target_n <= 18 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initial graph
            adj = [[False] * target_n for _ in range(target_n)]
            edges = []
            
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i][j] = adj[j][i] = True
                        edges.append([i, j])
            
            # Check if valid
            if _check_graph_valid(adj, target_n):
                best_n = target_n
                best_edges = edges.copy()
                found = True
                break
            
            # If not valid, try local search: flip edges to eliminate regular subgraphs
            # This is complex. For now, just try more random graphs.
        
        if found:
            # Try to extend to target_n + 1
            if time.time() > deadline - 15:
                break
            
            # Try adding one vertex
            new_adj = [[False] * (target_n + 1) for _ in range(target_n + 1)]
            for i in range(target_n):
                for j in range(target_n):
                    new_adj[i][j] = adj[i][j]
            
            # Randomly connect new vertex
            for restart2 in range(100):
                if time.time() > deadline - 5:
                    break
                
                for i in range(target_n):
                    if random.random() < 0.5:
                        new_adj[i][target_n] = new_adj[target_n][i] = True
                    else:
                        new_adj[i][target_n] = new_adj[target_n][i] = False
                
                if _check_graph_valid(new_adj, target_n + 1):
                    new_edges = []
                    for i in range(target_n + 1):
                        for j in range(i + 1, target_n + 1):
                            if new_adj[i][j]:
                                new_edges.append([i, j])
                    best_n = target_n + 1
                    best_edges = new_edges
                    adj = new_adj
                    target_n = target_n + 1
                    break
    
    # If we didn't find anything, return a small valid graph
    if best_n == 0:
        # Return a simple valid graph on 6 vertices (non-regular)
        # Path graph on 6 vertices: degrees 1,2,2,2,2,1 -> not regular
        edges_6 = [[0,1], [1,2], [2,3], [3,4], [4,5]]
        return {'n': 6, 'edges': edges_6}
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
