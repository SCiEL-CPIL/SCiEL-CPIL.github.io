import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n = 31
    m = 91
    
    # Helper to count C4s using numpy-free approach for speed
    # We represent adjacency as a list of sets for common neighbor computation
    # Or a bitset for speed. n=31 fits in an int.
    # Let's use an integer bitmask for each vertex.
    
    def count_c4(adj):
        # adj: list of ints (bitmasks)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    count += common.bit_count()
        return count // 2
    
    # Step 1: Generate a strong initial graph
    # Strategy: Start with a C4-free graph with 90 edges, then add the "best" 91st edge.
    # How to get a C4-free graph with 90 edges?
    # Known construction: Erdos-Renyi random graph is not C4-free with high prob at this density.
    # Random greedy construction (Kleitman-Spencer):
    # Start with empty graph. Add edges randomly if it doesn't create a C4.
    # This will stop when no more edges can be added without creating a C4.
    # The size of such a maximal C4-free graph is typically close to ex(n, C4).
    # For n=31, ex(31, C4) = 90.
    
    def generate_maximal_c4_free(n, m_target=90, max_restarts=50):
        best_edges = None
        best_count = -1
        # We want a C4-free graph.
        # We will try to build one greedily.
        # Randomize the order of potential edges.
        
        for _ in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
                
            edges = set()
            adj = [0] * n
            # Potential edges in random order
            potential = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(potential)
            
            for i, j in potential:
                # Check if adding (i,j) creates a C4
                # A C4 is formed if there exists k such that (i,k) and (j,k) are in the graph.
                # common neighbors of i and j
                common = adj[i] & adj[j]
                if common == 0:
                    # Safe to add
                    edges.add((i, j))
                    edges.add((j, i)) # Wait, we store undirected, let's keep a set of frozensets or just tuples (min, max)
                    # Let's use a set of (u, v) with u < v
                    edges.discard((j, i))
                    edges.add((i, j))
                    
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
                    if len(edges) >= 90:
                        # We have a C4-free graph with at least 90 edges.
                        # It might have more, but we only need 90.
                        # Actually, the greedy process adds edges as long as they are C4-free.
                        # So the resulting graph is C4-free.
                        # If it has > 90 edges, we can just pick any 90.
                        # But we want to add the 91st edge later.
                        # So let's store this adj.
                        return list(edges)
                        
            # If we run out of potential edges, the graph is maximal C4-free.
            # It might have fewer than 90 edges (unlikely for n=31, but possible with bad luck)
            if len(edges) >= 90:
                return list(edges)
            
        # Fallback: Return a random graph with 90 edges (will have many C4s, but valid)
        # Or better, just return the last generated one even if < 90? No, must be exactly 91 total.
        # If greedy fails to get 90 C4-free edges, we start from empty and just add random edges until we have 90, ignoring C4s?
        # That's bad.
        # Let's try to find a known C4-free graph or just use random sampling with high retry count.
        
        # Better fallback: 
        # Use a specific known C4-free graph structure if possible, or just random.
        # For now, return a random set of 90 edges.
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        return all_edges[:90]

    # Generate initial C4-free set of 90 edges
    base_edges = generate_maximal_c4_free(n)
    
    # If base_edges has more than 90, trim to 90. 
    # Note: Trimming might break the "maximal" property but keeps it C4-free.
    # Actually, any subset of a C4-free graph is C4-free.
    if len(base_edges) > 90:
        random.shuffle(base_edges)
        base_edges = base_edges[:90]
    
    if len(base_edges) < 90:
        # Fallback: add random edges until we have 90 (will introduce C4s, but better than crash)
        existing = set(base_edges)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for e in all_possible:
            if len(base_edges) >= 90:
                break
            if e not in existing:
                base_edges.append(e)
                existing.add(e)
    
    # Now we have 90 edges, which is C4-free (or mostly, if fallback used).
    # We need to add 1 more edge.
    # To minimize total C4s in the final 91-edge graph:
    # Total C4s = C4s in base (0) + C4s involving the new edge.
    # A new edge (u, v) creates a C4 for every common neighbor w of u and v in the base graph.
    # So we want to find a non-edge (u, v) that minimizes the number of common neighbors in the base graph.
    # Ideally 0, which would mean the graph remains C4-free (impossible since ex(31, C4)=90, so any 91st edge creates >=1 C4).
    # So we are looking for the minimum number of common neighbors among all non-edges.
    
    # Build adjacency for base graph
    adj = [0] * n
    for u, v in base_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Find the best edge to add
    best_edge = None
    min_common = n
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    existing_set = set(base_edges)
    
    for u, v in all_pairs:
        if (u, v) in existing_set:
            continue
        common = (adj[u] & adj[v]).bit_count()
        if common < min_common:
            min_common = common
            best_edge = (u, v)
            if min_common == 1:
                # Can't get 0 if ex=90. 1 is optimal.
                # We can break early if we find 1? 
                # Yes, 1 is the theoretical minimum if ex(n,C4) = m-1.
                break
    
    # Add the best edge
    final_edges = base_edges + [list(best_edge)]
    
    # Shuffle to look random? Not required, but good practice.
    random.shuffle(final_edges)
    
    # Verify validity
    assert len(final_edges) == 91
    seen = set()
    for e in final_edges:
        u, v = e
        assert 0 <= u < 31 and 0 <= v < 31 and u != v
        key = (min(u,v), max(u,v))
        assert key not in seen
        seen.add(key)
        
    return final_edges
