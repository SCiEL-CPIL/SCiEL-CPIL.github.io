import time
import random
import numpy as np
from itertools import combinations

def construct():
    # Define the two instances
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    total_time_limit = 140.0  # Leave 10s safety margin for the evaluator
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        remaining_time = total_time_limit - elapsed
        
        # If we don't have enough time, just do the trivial solution
        if remaining_time < 1.0:
            cliques = trivial_partition(n, p, seed)
        else:
            # Allocate time proportionally to problem size, but with a minimum
            # Give at least 5s to each instance
            if idx == 0:
                instance_time_limit = min(remaining_time * 0.4, 40.0)
            else:
                instance_time_limit = min(remaining_time * 0.6, 100.0)
            
            cliques = solve_instance(n, p, seed, instance_time_limit)
            
        results.append(cliques)
        
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def generate_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def solve_instance(n, p, seed, time_limit):
    edges = generate_edges(n, p, seed)
    if not edges:
        return []
        
    # Build bitset adjacency for residual graph
    # res_adj[u] is an integer where bit v is 1 if edge (u,v) is in residual graph
    res_adj = [0] * n
    for u, v in edges:
        res_adj[u] |= (1 << v)
        res_adj[v] |= (1 << u)
        
    cliques = []
    start_time = time.time()
    
    # Helper to get list of vertices from bitset
    def bitset_to_list(bits):
        return [i for i in range(n) if (bits >> i) & 1]

    # Helper to count bits
    def popcount(bits):
        return bin(bits).count('1')

    # Main greedy loop
    while any(res_adj):
        current_time = time.time()
        if current_time - start_time > time_limit * 0.7: # Reserve 30% for local search
             # Timeout: fall back to trivial for remaining edges
             for u in range(n):
                 bits = res_adj[u]
                 while bits:
                     lsb = bits & -bits
                     v = lsb.bit_length() - 1
                     if u < v:
                         cliques.append([u, v])
                     bits -= lsb
             break
        
        # Find a large clique using bitset greedy
        clique = find_large_clique_bitset(res_adj, n)
        
        if not clique:
            # Safety: find any edge
            found = False
            for u in range(n):
                bits = res_adj[u]
                while bits:
                    lsb = bits & -bits
                    v = lsb.bit_length() - 1
                    if u < v:
                        clique = [u, v]
                        found = True
                        break
                    bits -= lsb
                if found: break
            if not found:
                break
                
        # Remove edges in clique from residual graph
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                res_adj[u] &= ~(1 << v)
                res_adj[v] &= ~(1 << u)
                
        cliques.append(clique)

    # Local Search: Merge cliques if possible
    # We want to reduce the number of cliques.
    # If C1 and C2 are cliques, and all vertices in C1 are connected to all vertices in C2,
    # then C1 U C2 is a clique.
    # The edges between C1 and C2 are currently covered by other cliques.
    # If we merge C1 and C2, we cover those cross-edges with the new clique.
    # We must remove those cross-edges from the other cliques.
    
    # To do this efficiently, we need to know which cliques cover which edges.
    # This is complex. A simpler local search:
    # Just check if any two cliques in the list can be merged into a single clique
    # such that the union is a clique, AND we don't double count edges.
    # Wait, the partition is a set of cliques. The edges are the union of all edges in these cliques.
    # If we merge C1 and C2, the new set of edges is E(C1) U E(C2) U E(C1, C2).
    # The original partition had E(C1), E(C2), and the edges in E(C1, C2) were distributed among other cliques.
    # If we merge, we add E(C1, C2) to the new clique. We must REMOVE E(C1, C2) from the other cliques.
    # This is a valid operation if the remaining edges in the other cliques still form cliques.
    
    # Given the complexity and time limits, we will implement a simplified version:
    # We will try to find two cliques C_i, C_j such that C_i U C_j is a clique.
    # If found, we merge them. We assume the "cross edges" were covered by cliques that 
    # will be handled by re-running the greedy on the residual? No, that's too slow.
    
    # Alternative Local Search:
    # Re-extract cliques from the residual graph more intelligently?
    # Actually, the bitset greedy is already quite good.
    # Let's just return the greedy result. The bitset speedup allows us to run more restarts if we want, 
    # but a single pass is often near-optimal for random graphs.
    
    # Let's try to improve the initial greedy by running it multiple times with different start vertices?
    # The current `find_large_clique_bitset` already tries multiple starts.
    
    return cliques

def find_large_clique_bitset(res_adj, n):
    """Find a large clique using bitset operations."""
    # Calculate degrees (popcount)
    degrees = [popcount(res_adj[i]) for i in range(n)]
    
    # Get non-isolated vertices
    active_vertices = [u for u in range(n) if degrees[u] > 0]
    if not active_vertices:
        return []
    
    best_clique = []
    
    # Strategy: Greedy from top-degree vertices
    sorted_by_degree = sorted(active_vertices, key=lambda x: degrees[x], reverse=True)
    
    # Try top K start vertices
    K = min(50, len(active_vertices))
    for start_v in sorted_by_degree[:K]:
        C = [start_v]
        P = res_adj[start_v]
        
        while P:
            # Pick u in P that maximizes popcount(P & res_adj[u])
            # Iterate through bits in P
            best_u = -1
            best_score = -1
            temp_P = P
            while temp_P:
                lsb = temp_P & -temp_P
                u = lsb.bit_length() - 1
                score = popcount(P & res_adj[u])
                if score > best_score:
                    best_score = score
                    best_u = u
                temp_P -= lsb
            
            if best_u == -1:
                break
                
            C.append(best_u)
            P &= res_adj[best_u]
            
        if len(C) > len(best_clique):
            best_clique = C
    
    # Random restarts
    import random as _rnd
    rng = _rnd.Random(123)
    for _ in range(20):
        if not active_vertices: break
        start_v = rng.choice(active_vertices)
        C = [start_v]
        P = res_adj[start_v]
        
        while P:
            best_u = -1
            best_score = -1
            temp_P = P
            while temp_P:
                lsb = temp_P & -temp_P
                u = lsb.bit_length() - 1
                score = popcount(P & res_adj[u])
                if score > best_score:
                    best_score = score
                    best_u = u
                temp_P -= lsb
            
            if best_u == -1:
                break
                
            C.append(best_u)
            P &= res_adj[best_u]
            
        if len(C) > len(best_clique):
            best_clique = C
            
    return best_clique
