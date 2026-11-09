import random
import time
import sys

def construct():
    n = 20
    deadline = time.time() + 110  # 10s margin
    
    def is_valid_add(u, v, neighbors):
        # Check if adding edge (u, v) creates a K3,3
        # Only need to check triples involving u or v
        # A triple (a, b, c) has common neighbors N(a) & N(b) & N(c)
        # Adding (u, v) means u is in N(v) and v is in N(u)
        # This increases common neighbor count for triples containing u or v
        
        # Pre-compute neighbors sets as bitmasks for speed? 
        # Or just use sets. n=20 is small.
        
        # Let's use bitmasks for the check.
        # neighbors is a list of int bitmasks.
        
        # Update neighbors temporarily? No, just check.
        # If we add (u, v):
        # new_N[u] = neighbors[u] | (1 << v)
        # new_N[v] = neighbors[v] | (1 << u)
        # Others unchanged.
        
        new_N_u = neighbors[u] | (1 << v)
        new_N_v = neighbors[v] | (1 << u)
        
        # Check all triples (a, b, c)
        # We only care if any triple has >= 3 common neighbors.
        # If a triple doesn't contain u or v, its common neighbors are unchanged.
        # Since the graph was valid before, only triples containing u or v can become invalid.
        
        # Triples containing u: (u, i, j) for i < j, i != u, j != u
        # Triples containing v: (v, i, j) for i < j, i != v, j != v
        
        # To avoid double checking triples containing both u and v, we can just check all triples containing u OR v.
        # It's simpler to just check all triples containing u, and all triples containing v.
        # Overlap is fine, just redundant.
        
        for i in range(n):
            if i == u: continue
            for j in range(i+1, n):
                if j == u: continue
                # Triple (u, i, j)
                common = new_N_u & neighbors[i] & neighbors[j]
                if common.bit_count() >= 3:
                    return False
        
        for i in range(n):
            if i == v: continue
            for j in range(i+1, n):
                if j == v: continue
                # Triple (v, i, j)
                common = new_N_v & neighbors[i] & neighbors[j]
                if common.bit_count() >= 3:
                    return False
        return True

    def greedy_construct(seed):
        random.seed(seed)
        neighbors = [0] * n
        edges = []
        
        # Generate all possible edges
        possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if is_valid_add(u, v, neighbors):
                # Add edge
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                edges.append([u, v])
        
        return edges

    best_edges = []
    best_count = 0
    seed = 0
    
    while time.time() < deadline:
        edges = greedy_construct(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        seed += 1
    
    return best_edges
