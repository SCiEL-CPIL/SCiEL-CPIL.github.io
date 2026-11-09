import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    n = 20
    
    # Helper: Check if graph is K3,3-free
    # K3,3 exists iff there are 3 vertices with >= 3 common neighbors.
    def is_k33_free(adj_bits):
        for i in range(n):
            for j in range(i+1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    # Strategy:
    # 1. Use the structure of the polarity graph of PG(2,4).
    #    PG(2,4) has 21 points. The polarity graph has 21 vertices.
    #    It is known to be K3,3-free (or close to it).
    #    We need 20 vertices, so we can take a subgraph of 20 vertices.
    #    
    #    Construction of PG(2,4) polarity graph:
    #    Points are projective points over GF(4).
    #    Let GF(4) = {0, 1, w, w^2} with w^2 = w + 1, w^3 = 1.
    #    Projective points are 1-dim subspaces of GF(4)^3.
    #    There are (4^3 - 1)/(4-1) = 63/3 = 21 points.
    #    
    #    Let's enumerate them.
    #    We can represent points as tuples (x,y,z) not all zero, normalized such that the first non-zero component is 1.
    #    Or simpler: iterate over all non-zero vectors, identify unique lines.
    
    # GF(4) elements
    # 0, 1, w, w2
    # w = 1 + alpha, w2 = alpha, w3 = 1
    # Actually, let's just use integers 0,1,2,3 for GF(4) elements with custom arithmetic?
    # Or use a library? No, standard lib only.
    # Let's hardcode the 21 points and the adjacency based on a known polarity.
    # A standard polarity in PG(2,4) is given by the Hermitian form or a symmetric bilinear form.
    # For PG(2,4), the "polarity graph" usually refers to the graph where two points are adjacent if they are incident (one lies on the polar line of the other).
    # This graph is known to be K3,3-free? 
    # Actually, the polarity graph of PG(2, q) for q even is a known example of a K3,3-free graph? 
    # Let's check: In PG(2,4), each point is incident to its polar line (4 points). 
    # The graph has 21 vertices. Degree is roughly 4 or 5?
    # Total edges ~ 21 * 4 / 2 = 42. This is sparse. We want dense.
    
    # Alternative: The complement of the polarity graph?
    # If G is K3,3-free, is G^c K3,3-free? No.
    
    # Let's try a different known dense K3,3-free construction.
    # The graph formed by the "non-collinearity" of points in PG(2,4)?
    # Or maybe use a random graph with high density and repair?
    
    # Let's go back to the inspiration: "Stronger algorithm for dense K3,3-free graphs".
    # The seed program uses random greedy + local search.
    # Let's try a different initialization: Start with a complete graph and remove edges to break K3,3s.
    # This is a "deletion" heuristic.
    
    # 1. Start with K20 (190 edges).
    # 2. While there is a K3,3, remove an edge from one of the K3,3s.
    #    Which edge? Randomly, or the one in the most K3,3s?
    
    adj_bits = [0] * n
    # Initialize with complete graph
    for i in range(n):
        for j in range(i+1, n):
            add_edge(adj_bits, i, j)
            
    # Find all K3,3s and remove edges
    # A K3,3 is defined by two disjoint sets of 3 vertices A={a1,a2,a3} and B={b1,b2,b3}
    # such that all edges between A and B exist.
    # In a dense graph, there are many.
    
    # To be efficient, we can iterate through triples of vertices in A.
    # For each triple A, find common neighbors of A.
    # If common neighbors contain a triple B (disjoint from A), we have a K3,3.
    
    def find_and_remove_k33(adj_bits, max_iterations=10000):
        removed_any = False
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Find a K3,3
            k33_found = None
            # Iterate through all triples A
            for a1 in range(n):
                for a2 in range(a1+1, n):
                    for a3 in range(a2+1, n):
                        # Common neighbors of A
                        common = adj_bits[a1] & adj_bits[a2] & adj_bits[a3]
                        # Remove A itself from common (though they are not neighbors of each other in simple graph sense for common neighbor definition? 
                        # Wait, common neighbors of a1, a2, a3 are vertices v such that v is connected to a1, a2, a3.
                        # a1 is not connected to a1. So a1 is not in common.
                        # So common contains only vertices outside {a1, a2, a3} potentially, 
                        # BUT if a1, a2, a3 are mutually connected, they are not in their own common neighbor set.
                        
                        # We need 3 vertices B in common that are mutually connected?
                        # No, K3,3 requires all 9 edges between A and B.
                        # It does NOT require edges within A or within B.
                        # So we just need 3 distinct vertices in common that are NOT in A.
                        # And they must be distinct from each other.
                        
                        # Let's get the list of vertices in common
                        verts_in_common = []
                        mask = common
                        while mask:
                            lsb = mask & -mask
                            v = lsb.bit_length() - 1
                            verts_in_common.append(v)
                            mask ^= lsb
                        
                        # Filter out A
                        verts_in_common = [v for v in verts_in_common if v not in (a1, a2, a3)]
                        
                        if len(verts_in_common) >= 3:
                            # We have at least 3 candidates for B.
                            # Pick the first 3.
                            b1, b2, b3 = verts_in_common[0], verts_in_common[1], verts_in_common[2]
                            # This forms a K3,3 between {a1,a2,a3} and {b1,b2,b3}
                            k33_found = ((a1, a2, a3), (b1, b2, b3))
                            break
                    if k33_found:
                        break
                if k33_found:
                    break
            
            if not k33_found:
                break
            
            # Remove an edge from this K3,3
            # Which one? Randomly.
            A, B = k33_found
            a1, a2, a3 = A
            b1, b2, b3 = B
            # All 9 edges exist.
            # Pick a random edge to remove
            u, v = random.choice([(a1,b1), (a1,b2), (a1,b3), (a2,b1), (a2,b2), (a2,b3), (a3,b1), (a3,b2), (a3,b3)])
            remove_edge(adj_bits, u, v)
            removed_any = True
            
        return removed_any

    # Run deletion phase
    # This might be slow if we restart the search from scratch every time.
    # But n=20 is small.
    
    while find_and_remove_k33(adj_bits, max_iterations=100):
        if time.time() > deadline - 10:
            break

    # Now we have a K3,3-free graph.
    # Try to add edges back (local search)
    current_adj = adj_bits[:]
    
    # Local search: try to add missing edges
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] >> j) & 1]
    
    progress = True
    while progress and time.time() < deadline - 10:
        progress = False
        random.shuffle(missing)
        new_missing = []
        for u, v in missing:
            if time.time() > deadline - 10:
                new_missing.append((u, v))
                continue
            add_edge(current_adj, u, v)
            if is_k33_free(current_adj):
                progress = True
            else:
                remove_edge(current_adj, u, v)
                new_missing.append((u, v))
        missing = new_missing
        
        # If stuck, try swaps? 
        # For now, just re-shuffle and retry.
        
    # Final result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (current_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
