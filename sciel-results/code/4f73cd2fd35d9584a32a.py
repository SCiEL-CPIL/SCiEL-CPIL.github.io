import time
import random
import sys

# Edge generation and conflict graph precomputation
# Edges are represented by (u, v) where u < v.
# Index: 0..447

def get_edges():
    edges = []
    for u in range(128):
        for i in range(7):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
    return edges

# Precompute conflicts
# Two edges (u,v) and (w,z) conflict if they form a square.
# This happens if they share a vertex? No.
# If they share a vertex, they are part of a square only if their directions are different.
# Let e1 = (u, u^2^i), e2 = (v, v^2^j).
# They form a square if {u, u^2^i, v, v^2^j} is a set of 4 vertices forming a 4-cycle.
# This requires u and v to be connected by a path of length 2 in the hypercube using directions i and j?
# Actually, the condition is:
# Let d1 = u ^ v.
# If d1 is a single bit (say k), then u and v are adjacent.
# If i != j and i != k and j != k? No.
# Let's use the property:
# e1 and e2 conflict iff there exists a square containing both.
# A square is defined by two distinct bit positions i, j and a base vertex x.
# Vertices: x, x^2^i, x^2^j, x^2^i^2^j.
# Edges: (x, x^2^i), (x, x^2^j), (x^2^i, x^2^i^2^j), (x^2^j, x^2^i^2^j).
# So e1 and e2 conflict if they are both in the edge set of some square.

# Efficient check:
# For each edge e1 = (u, v) with dir i.
# It is in 6 squares (one for each j != i).
# For a fixed j, the square is determined by u (or v).
# The other edges of direction j in this square are:
# (u, u^2^j) and (v, v^2^j).
# So e1 conflicts with (u, u^2^j) and (v, v^2^j) for all j != i.

def build_conflicts(edges):
    # Map edge tuple to index
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    n = len(edges)
    conflicts = [[] for _ in range(n)]
    
    for i, e in enumerate(edges):
        u, v = e
        # Determine direction
        diff = u ^ v
        # diff is a power of 2
        i_dir = diff.bit_length() - 1
        
        for j in range(7):
            if j == i_dir:
                continue
            # The two edges of direction j in the square
            # Square vertices: u, v, u^2^j, v^2^j
            # Edges of dir j: (u, u^2^j) and (v, v^2^j)
            e1 = (u, u ^ (1 << j))
            e2 = (v, v ^ (1 << j))
            
            # Normalize e1, e2
            if e1[0] > e1[1]: e1 = (e1[1], e1[0])
            if e2[0] > e2[1]: e2 = (e2[1], e2[0])
            
            idx1 = edge_to_idx[e1]
            idx2 = edge_to_idx[e2]
            
            conflicts[i].append(idx1)
            conflicts[i].append(idx2)
            
    # Remove duplicates just in case (though logic seems sound)
    for i in range(n):
        conflicts[i] = list(set(conflicts[i]))
        
    return conflicts

EDGES = get_edges()
N_EDGES = len(EDGES)
CONFLICTS = build_conflicts(EDGES)

def run_greedy_mis(seed):
    random.seed(seed)
    active = set(range(N_EDGES))
    selected = []
    
    while active:
        # Find edge with min active neighbors
        min_count = float('inf')
        best_edges = []
        
        for e in active:
            # Count active neighbors
            count = 0
            for nb in CONFLICTS[e]:
                if nb in active:
                    count += 1
            if count < min_count:
                min_count = count
                best_edges = [e]
            elif count == min_count:
                best_edges.append(e)
                
        # Randomly pick one of the best
        e = random.choice(best_edges)
        selected.append(e)
        
        # Remove e and its neighbors from active
        active.remove(e)
        for nb in CONFLICTS[e]:
            if nb in active:
                active.remove(nb)
                
    return selected

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    best_set = None
    best_size = -1
    
    # Run multiple iterations
    while time.time() - start_time < timeout:
        seed = random.randint(0, 2**32 - 1)
        current_set = run_greedy_mis(seed)
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            
        # Optional: If we hit the upper bound, break
        if best_size >= 308:
            break
            
    # Convert indices to edge tuples
    result = [EDGES[i] for i in best_set]
    return result
