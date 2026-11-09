import time
import random
import numpy as np

N = 40
DEADLINE = 110  # Seconds

def _creates_c6(adj, u, v):
    """
    Checks if adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    """
    nu = adj[u]
    nv = adj[v]
    
    # Iterate a in N(u)
    a_mask = nu
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a
        
        if a == v: continue
        
        # Iterate d in N(v)
        d_mask = nv
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d
            
            if d == u or d == a: continue
            
            # We need b in N(a) and c in N(d) such that (b, c) is an edge.
            # b, c must be distinct from u, v, a, d.
            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            
            # Candidates for b: neighbors of a, excluding forbidden
            b_mask = adj[a] & ~forbidden
            if not b_mask: continue
            
            # Candidates for c: neighbors of d, excluding forbidden
            c_mask = adj[d] & ~forbidden
            if not c_mask: continue
            
            # Check if any b in b_mask is connected to any c in c_mask
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                if (adj[b] & c_mask):
                    return True
                        
    return False

def construct():
    start_time = time.time()
    best_edges = []
    best_count = 0
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Run random restarts until deadline
    while time.time() - start_time < DEADLINE:
        random.shuffle(all_edges)
        adj = [0] * N
        count = 0
        
        for u, v in all_edges:
            if not _creates_c6(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        if count > best_count:
            best_count = count
            # Reconstruct edge list
            best_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if (adj[i] >> j) & 1:
                        best_edges.append([i, j])
    
    return best_edges
