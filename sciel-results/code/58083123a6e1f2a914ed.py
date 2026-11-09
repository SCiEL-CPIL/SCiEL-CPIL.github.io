import time
import random
import itertools

_START_TIME = time.time()
_DEADLINE = 110.0

def _popcount(x):
    return bin(x).count('1')

def _lowest_bit(x):
    return (x & -x).bit_length() - 1

def _has_independent_set_10(adj, n):
    """
    Checks if the graph defined by adj (list of sets) has an independent set of size 10.
    This is equivalent to checking if the complement graph has a clique of size 10.
    """
    # Build complement adjacency masks
    # comp_mask[i] has bit j set if (i,j) is NOT an edge in G, and i != j
    comp_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp_mask[i] |= (1 << j)
                comp_mask[j] |= (1 << i)
    
    # We want to find a clique of size 10 in the graph defined by comp_mask.
    # Initial candidates: all vertices
    initial_candidates = (1 << n) - 1
    
    # Heuristic: order vertices by degree in complement?
    # For simplicity, we use the standard bitset branch and bound.
    # We can improve pruning by picking the vertex with max degree in candidates.
    
    def search(candidates, size):
        if size >= 10:
            return True
        if candidates == 0:
            return False
        # Pruning: if size + popcount(candidates) < 10, impossible
        if size + _popcount(candidates) < 10:
            return False
        
        # Pick a vertex v from candidates
        # Heuristic: pick v with max degree within candidates to reduce branching factor
        # deg(v) in candidates = popcount(candidates & comp_mask[v])
        # Finding max degree vertex takes O(n * popcount). n=40, so it's fast.
        
        # Optimization: If we just pick the lowest bit, it's slower but simpler.
        # Let's try a simple greedy pick: highest degree in candidates.
        best_v = -1
        best_deg = -1
        temp_cands = candidates
        while temp_cands:
            v = _lowest_bit(temp_cands)
            temp_cands &= ~(1 << v)
            deg = _popcount(candidates & comp_mask[v])
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            return False
            
        # Branch 1: Include best_v
        # New candidates must be neighbors of best_v in complement, and in current candidates
        new_candidates = candidates & comp_mask[best_v]
        if search(new_candidates, size + 1):
            return True
            
        # Branch 2: Exclude best_v
        candidates &= ~(1 << best_v)
        if search(candidates, size):
            return True
            
        return False

    return search(initial_candidates, 0)

def _generate_and_cleanup(n, seed):
    """
    Generate a random graph G(n, 0.5) and remove edges to make it triangle-free.
    """
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    
    # Add edges with probability 0.5
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i].add(j)
                adj[j].add(i)
                
    # Remove edges to destroy triangles
    # Loop until no triangles
    # To speed up, we can find all triangles, pick one, remove an edge.
    # Re-finding triangles is O(n^3) which is ~64k for n=40. Very fast.
    
    max_iters = 10000
    for _ in range(max_iters):
        # Find a triangle
        tri_found = False
        for i in range(n):
            # Optimization: only check j > i, k > j
            for j in adj[i]:
                if j <= i: continue
                # Check common neighbors
                common = adj[i] & adj[j]
                # We need k > j to ensure unique triangle identification? 
                # Actually, any triangle is fine.
                # But iterating common neighbors is fast.
                for k in common:
                    if k > i and k > j: # Ensure we find each triangle once? 
                        # Actually, we just need ONE triangle to break.
                        # If we break it, we restart the search.
                        # So we can just return the first one found.
                        # Remove edge (i, j)
                        adj[i].discard(j)
                        adj[j].discard(i)
                        tri_found = True
                        break
                if tri_found:
                    break
            if tri_found:
                break
        if not tri_found:
            break
            
    return adj

def _andrasfai_9():
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def construct():
    """
    Find a triangle-free graph with no independent set of 10 vertices.
    Maximize n.
    """
    # Fallback
    fallback = _andrasfai_9()
    
    # Try to find larger graphs
    # n=40 is the max possible if R(3,10) > 40.
    # If R(3,10) = 41, then n=40 is valid.
    # If R(3,10) = 40, then n=39 is valid.
    
    # We iterate n downwards.
    for n in range(40, 26, -1):
        if time.time() - _START_TIME > _DEADLINE:
            return fallback
            
        # Try multiple seeds
        # The probability of finding a valid graph might be low for n=40.
        # We allocate time based on n.
        attempts = 50
        if n > 35:
            attempts = 20
        
        for seed in range(attempts):
            if time.time() - _START_TIME > _DEADLINE:
                return fallback
                
            adj = _generate_and_cleanup(n, seed)
            
            # Check if valid
            # It is triangle-free by construction.
            # Check alpha < 10
            if not _has_independent_set_10(adj, n):
                # Valid!
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    return fallback
