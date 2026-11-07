"""Improved program: Randomized search for clique subdivisions with robust path finding."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees for heuristic sorting
    degrees = [len(adj[i]) for i in range(n)]
    
    def _bfs_path(start, end, forbidden, adj):
        """Find a simple path from start to end avoiding forbidden vertices (except start, end)."""
        if start == end:
            return [start]
        
        queue = deque()
        queue.append((start, [start]))
        visited = {start}
        
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            for neighbor in adj[node]:
                if neighbor in visited:
                    continue
                if neighbor in forbidden and neighbor not in (start, end):
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        return None

    def find_subdivision(branches, attempts=200):
        """
        Try to find vertex-disjoint paths for all pairs in branches.
        Returns a list of paths or None.
        """
        k = len(branches)
        if k < 2:
            return None
        
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append((i, j))
        
        # For small k, we can try all permutations? No, too many.
        # We use random shuffles.
        # Also, we can try to order pairs by "difficulty" (e.g., distance)
        
        best_paths = None
        
        for attempt in range(attempts):
            trial_paths = {}
            trial_used = set(branches)
            success = True
            
            # Create a list of pair indices
            pair_indices = list(range(len(pairs)))
            if attempt > 0:
                random.shuffle(pair_indices)
            
            for idx in pair_indices:
                i, j = pairs[idx]
                b1 = branches[i]
                b2 = branches[j]
                
                # Find path
                path = _bfs_path(b1, b2, trial_used, adj)
                if path is None:
                    success = False
                    break
                
                # Check if internal vertices are disjoint
                internal_valid = True
                for v in path[1:-1]:
                    if v in trial_used:
                        internal_valid = False
                        break
                    trial_used.add(v)
                
                if not internal_valid:
                    success = False
                    break
                
                trial_paths[(i, j)] = path
            
            if success:
                return [trial_paths[pairs[idx]] for idx in range(len(pairs))]
        
        return None

    # Strategy:
    # 1. Try all cliques of size up to 8 (they are good candidates)
    # 2. Try random sets of high-degree vertices
    # 3. Try random sets of vertices
    
    max_k = 8
    found = None
    
    # Generate candidate sets
    candidates = []
    
    # 1. All cliques
    for k in range(max_k, 1, -1):
        for subset in combinations(range(n), k):
            # Check if clique
            is_clique = True
            for i in range(k):
                for j in range(i+1, k):
                    if subset[j] not in adj[subset[i]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                candidates.append((list(subset), k))
    
    # 2. Random high-degree sets
    high_deg_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    for _ in range(1000):
        k = random.randint(3, max_k)
        # Pick k vertices, biased towards high degree
        subset = []
        available = list(range(n))
        for _ in range(k):
            # Weighted choice? Simple: pick from top 10 degrees randomly, or just random
            # Let's just pick random subset but sort by degree to prioritize
            if available:
                # Pick a random vertex from the top half of degrees?
                # Or just random
                v = random.choice(available)
                subset.append(v)
                available.remove(v)
        candidates.append((subset, k))
        
    # 3. Pure random sets
    for _ in range(1000):
        k = random.randint(3, max_k)
        subset = random.sample(range(n), k)
        candidates.append((subset, k))
        
    # Sort candidates by size descending, then by "cliqueness" or degree sum
    # We want to try larger sets first
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # To avoid redundant checks, we can limit the number of attempts per size
    # But we have 60s, so we can try many.
    
    # We will iterate through candidates.
    # To be efficient, we can group by size and stop if we find a solution for that size.
    
    # However, the order of candidates matters.
    # Let's just iterate.
    
    # Optimization: If we find a solution of size k, we can stop if k is the max possible?
    # We don't know the max possible.
    
    # Let's try sizes from max_k down to 2.
    # For each size, we have a list of candidates.
    
    # Re-organize candidates by size
    candidates_by_size = {k: [] for k in range(2, max_k+1)}
    for subset, k in candidates:
        if 2 <= k <= max_k:
            candidates_by_size[k].append(subset)
            
    # Also add all cliques to their respective size buckets (already done)
    
    # To ensure we don't miss a large clique, we should prioritize them.
    # The `candidates` list already has them.
    
    # Let's just iterate through the sorted `candidates` list.
    # But we should stop early if we find a large one.
    
    # To make it faster, we can limit the number of random candidates per size.
    # But the clique candidates are exhaustive for that size.
    
    # Let's just try them all.
    # If it's too slow, we can reduce attempts.
    
    # Timeout guard? The evaluator has 60s.
    # We can check time.
    import time
    start_time = time.time()
    timeout = 55  # Leave some buffer
    
    for subset, k in candidates:
        if time.time() - start_time > timeout:
            break
            
        if k < 2:
            continue
            
        paths = find_subdivision(subset, attempts=50)
        if paths is not None:
            # Verify
            # Build result
            branches = list(subset)
            result_paths = paths
            return {'branches': branches, 'paths': result_paths}
            
    # Fallback: find any two adjacent vertices
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
