"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    deadline = time.time() + 140.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Bitmask representation of adjacency
        # adj_mask[i] is an integer where bit j is 1 if (i, j) is an edge
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        cliques = []
        
        # Helper to get max clique in a graph defined by masks
        # We use a greedy heuristic: 
        # 1. Pick vertex with highest degree in current subgraph
        # 2. Expand greedily to form a clique
        
        def get_max_clique(start_vertex, masks):
            # Greedy expansion
            clique = [start_vertex]
            # Candidates must be neighbors of all current clique members
            # Start with neighbors of start_vertex
            candidates = masks[start_vertex]
            
            while True:
                if candidates == 0:
                    break
                
                # Find best candidate
                # Heuristic: Pick candidate with max intersection with current clique's common neighbors?
                # Or just pick the one with highest degree in the candidate set?
                
                # For speed, let's just pick the candidate with the most connections to other candidates
                # But that's expensive. 
                # Simpler: Pick a random candidate? No, deterministic is better for reproducibility.
                # Let's pick the candidate with the highest degree within the candidate set.
                
                # Get list of candidates
                cand_list = []
                c = candidates
                while c:
                    # Extract lowest set bit
                    lsb = c & -c
                    idx = lsb.bit_length() - 1
                    cand_list.append(idx)
                    c ^= lsb
                
                if not cand_list:
                    break
                
                # Score candidates: degree within cand_list
                best_cand = None
                best_score = -1
                for w in cand_list:
                    # Count how many other candidates are neighbors of w
                    # neighbors of w in original graph: masks[w]
                    # intersection with candidates
                    inter = masks[w] & candidates
                    # popcount
                    score = bin(inter).count('1')
                    if score > best_score:
                        best_score = score
                        best_cand = w
                
                if best_cand is None:
                    break
                
                clique.append(best_cand)
                
                # Update candidates: must be neighbor of best_cand
                candidates &= masks[best_cand]
                # Remove best_cand from candidates (it's in clique now)
                candidates &= ~(1 << best_cand)
                
            return clique
        
        # Main loop
        # We need to track remaining edges.
        # Since we remove edges, we can just modify adj_mask.
        
        # To pick the next clique, we find the vertex with the highest remaining degree.
        # Then we find the max clique in its neighborhood.
        
        # We need to ensure we don't pick isolated vertices or vertices with no edges.
        
        while True:
            if time.time() > deadline - 5:
                # Fallback: add remaining edges as singletons (cliques of size 2)
                # This is inefficient but ensures correctness if we run out of time
                # We need to reconstruct remaining edges from adj_mask
                for u in range(n):
                    m = adj_mask[u]
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            adj_mask[u] &= ~(1 << v)
                            adj_mask[v] &= ~(1 << u)
                        m ^= lsb
                break
            
            # Find vertex with max degree
            max_deg = -1
            start_v = -1
            for u in range(n):
                deg = bin(adj_mask[u]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    start_v = u
            
            if max_deg == 0:
                break # No edges left
            
            # Find max clique starting from start_v
            # The clique must be a subset of {start_v} U N(start_v)
            # And all vertices in clique must be mutually connected.
            
            clique = get_max_clique(start_v, adj_mask)
            
            if not clique:
                # Should not happen if max_deg > 0, but safety
                break
            
            cliques.append(clique)
            
            # Remove all edges within this clique from adj_mask
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u = clique[i]
                    v = clique[j]
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
            
        return cliques

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
