import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Try to construct graphs from n=35 down to 18
    for n in range(35, 17, -1):
        if time.time() > deadline - 5.0:
            break
        result = local_search(n, deadline)
        if result is not None:
            return result
    
    # Fallback: Paley graph on 17 vertices
    return paley_17()

def paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                edges.append([i, j])
    return {'n': 17, 'edges': edges}

def count_k4(adj, n):
    """Count K4s in the graph using bitmasks."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                # For each pair (j, k) where k > j and k is common neighbor of i and j
                # check if j and k are adjacent
                for k in range(j + 1, n):
                    if common & (1 << k):
                        if adj[j] & (1 << k):
                            # i, j, k form a triangle. Check if any common neighbor of all three
                            # forms a K4
                            common_ijk = adj[i] & adj[j] & adj[k]
                            common_ijk &= ~((1 << i) | (1 << j) | (1 << k))
                            count += bin(common_ijk).count('1')
    return count

def has_k4(adj, n):
    """Check if graph contains a K4. Returns True if found."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            for k in range(j + 1, n):
                if not (common & (1 << k)):
                    continue
                if not (adj[j] & (1 << k)):
                    continue
                # Triangle i,j,k exists. Check for 4th vertex
                common_ijk = adj[i] & adj[j] & adj[k]
                common_ijk &= ~((1 << i) | (1 << j) | (1 << k))
                if common_ijk:
                    return True
    return False

def has_ind6(adj, n):
    """Check if graph contains an independent set of size 6."""
    # Use branch and bound
    def find_ind_k(k, candidates):
        if k == 0:
            return True
        if len(candidates) < k:
            return False
        if k == 1:
            return True
        # Pick a vertex with minimum degree in the induced subgraph (heuristic)
        # For speed, just pick the first one
        v = candidates[0]
        remaining = [u for u in candidates[1:] if not (adj[v] & (1 << u))]
        if find_ind_k(k - 1, remaining):
            return True
        if find_ind_k(k, candidates[1:]):
            return True
        return False
    
    return find_ind_k(6, list(range(n)))

def local_search(n, deadline):
    """
    Local search / coordinate descent to find a K4-free graph with no independent 6-set.
    Start from a random graph and iteratively flip edges to eliminate violations.
    """
    
    def flip_edge(adj, i, j):
        """Flip edge between i and j."""
        if adj[i] & (1 << j):
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    def violation_score(adj, n):
        """Compute a score based on violations. Lower is better."""
        # Count K4s
        k4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    for k in range(j + 1, n):
                        if common & (1 << k):
                            if adj[j] & (1 << k):
                                common_ijk = adj[i] & adj[j] & adj[k]
                                common_ijk &= ~((1 << i) | (1 << j) | (1 << k))
                                k4_count += bin(common_ijk).count('1')
        
        # Check for ind6 - expensive, so do it less frequently
        # For now, just return k4_count as proxy
        return k4_count
    
    def find_k4(adj, n):
        """Find a specific K4 in the graph. Returns the 4 vertices or None."""
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j]
                for k in range(j + 1, n):
                    if not (common & (1 << k)):
                        continue
                    if not (adj[j] & (1 << k)):
                        continue
                    common_ijk = adj[i] & adj[j] & adj[k]
                    common_ijk &= ~((1 << i) | (1 << j) | (1 << k))
                    if common_ijk:
                        # Find the lowest set bit
                        l = (common_ijk & -common_ijk).bit_length() - 1
                        return [i, j, k, l]
        return None
    
    def find_ind6(adj, n):
        """Find an independent set of size 6. Returns the 6 vertices or None."""
        def find_ind_k(k, candidates):
            if k == 0:
                return []
            if len(candidates) < k:
                return None
            if k == 1:
                return [candidates[0]]
            v = candidates[0]
            remaining = [u for u in candidates[1:] if not (adj[v] & (1 << u))]
            res = find_ind_k(k - 1, remaining)
            if res is not None:
                return [v] + res
            res = find_ind_k(k, candidates[1:])
            if res is not None:
                return res
            return None
        
        return find_ind_k(6, list(range(n)))
    
    # Try multiple random restarts
    for restart in range(30):
        if time.time() > deadline - 2.0:
            break
        
        # Initialize with a random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local search: iteratively fix violations
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Check for K4
            k4 = find_k4(adj, n)
            if k4 is not None:
                # Break the K4 by removing a random edge from it
                a, b, c, d = k4
                edges_to_remove = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                random.shuffle(edges_to_remove)
                for u, v in edges_to_remove:
                    if adj[u] & (1 << v):
                        flip_edge(adj, u, v)
                        improved = True
                        break
                continue
            
            # Check for ind6
            ind6 = find_ind6(adj, n)
            if ind6 is not None:
                # Add a random edge within the independent set to break it
                random.shuffle(ind6)
                u, v = ind6[0], ind6[1]
                if not (adj[u] & (1 << v)):
                    flip_edge(adj, u, v)
                    improved = True
                else:
                    # Shouldn't happen, but just in case
                    for i in range(len(ind6)):
                        for j in range(i+1, len(ind6)):
                            if not (adj[ind6[i]] & (1 << ind6[j])):
                                flip_edge(adj, ind6[i], ind6[j])
                                improved = True
                                break
                        if improved:
                            break
                continue
            
            # No violations found - we have a valid graph!
            if not has_k4(adj, n) and not has_ind6(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
        # If we exited the while loop without finding a valid graph,
        # try a few more random perturbations
        if time.time() < deadline - 1.0:
            for perturb in range(10):
                if time.time() > deadline - 1.0:
                    break
                # Randomly flip a few edges
                for _ in range(3):
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i != j:
                        flip_edge(adj, i, j)
                
                # Continue local search
                improved = True
                while improved and time.time() < deadline - 1.0:
                    improved = False
                    
                    k4 = find_k4(adj, n)
                    if k4 is not None:
                        a, b, c, d = k4
                        edges_to_remove = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                        random.shuffle(edges_to_remove)
                        for u, v in edges_to_remove:
                            if adj[u] & (1 << v):
                                flip_edge(adj, u, v)
                                improved = True
                                break
                        continue
                    
                    ind6 = find_ind6(adj, n)
                    if ind6 is not None:
                        random.shuffle(ind6)
                        u, v = ind6[0], ind6[1]
                        if not (adj[u] & (1 << v)):
                            flip_edge(adj, u, v)
                            improved = True
                        continue
                    
                    if not has_k4(adj, n) and not has_ind6(adj, n):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if (adj[i] >> j) & 1:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
    
    return None
