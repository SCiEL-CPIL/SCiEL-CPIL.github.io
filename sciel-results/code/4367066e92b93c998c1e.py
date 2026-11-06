import time
import random
from collections import defaultdict

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448

    # Build edges: for each vertex u and each bit i, edge (u, u^(1<<i)) if u < u^(1<<i)
    # But we need a consistent indexing. Let's index edges by (u, i) where u has bit i = 0.
    # Actually, let's just enumerate all 448 edges.
    
    # Edge representation: (u, v) with u < v, or better, index by (min(u,v), max(u,v))
    # Let's use a different indexing: edge i corresponds to (vertex, bit) where vertex has that bit = 0.
    # So edge (u, i) connects u and u^(1<<i), where u has bit i = 0.
    # This gives exactly 448 edges, each uniquely identified.
    
    # Let's precompute:
    # edge_idx[u][i] = global index of edge from u in direction i (only when u has bit i = 0)
    # Or simpler: list of (u, v) for all 448 edges.
    
    edges = []
    edge_index = {}  # (u, v) with u<v -> idx
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx
    
    # Now, for each edge, find the 4 C4s it belongs to (as one of 4 edges in a face)
    # A C4 in Q7 is a 2D face: fix 5 bits, vary 2 bits.
    # For each pair of bits (i, j), i < j, and for each assignment of the other 5 bits,
    # we get a C4 with 4 edges.
    
    # For each edge, we need to know which C4s contain it.
    # A C4 containing edge e = (u, v) where u^v = 1<<i:
    # The C4 is determined by the other bit j that varies.
    # The 4 vertices are: u, v, u^(1<<j), v^(1<<j) for some j != i.
    # But u^(1<<j) and v^(1<<j) must be valid.
    # Actually, for a fixed edge (u, v) with u^v = 1<<i, and for each j != i,
    # we can form a C4 if we consider the face where bits i and j vary.
    # The 4 vertices are: u, u^(1<<i)=v, u^(1<<j), v^(1<<j)=u^(1<<i)^(1<<j)
    # This is always a valid C4 in Q7.
    
    # So each edge belongs to (d-1) = 6 C4s.
    
    # Let's build: for each edge idx, list of C4s (as tuples of 4 edge indices) that contain it.
    # And for each C4, the list of its 4 edge indices.
    
    # First, enumerate all C4s.
    # A C4 is determined by (i, j, base) where i<j are the varying bits,
    # and base is the value of the other 5 bits.
    # The 4 vertices: base, base^(1<<i), base^(1<<j), base^(1<<i)^(1<<j)
    # where base has bits i and j = 0.
    
    c4s = []  # list of tuples of 4 edge indices
    edge_to_c4s = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i+1, d):
            # base has bits i and j = 0, other bits can be 0 or 1
            mask = (1 << i) | (1 << j)
            other_mask = ((1 << d) - 1) ^ mask
            for other in range(1 << (d - 2)):
                # Construct base with bits i,j = 0 and other bits from 'other'
                # Need to map 'other' (d-2 bits) to the actual bit positions
                base = 0
                bit_pos = 0
                for b in range(d):
                    if b == i or b == j:
                        continue
                    if (other >> bit_pos) & 1:
                        base |= (1 << b)
                    bit_pos += 1
                
                # 4 vertices
                v0 = base
                v1 = base ^ (1 << i)
                v2 = base ^ (1 << j)
                v3 = base ^ (1 << i) ^ (1 << j)
                
                # 4 edges
                e01 = edge_index.get((min(v0,v1), max(v0,v1)))
                e02 = edge_index.get((min(v0,v2), max(v0,v2)))
                e13 = edge_index.get((min(v1,v3), max(v1,v3)))
                e23 = edge_index.get((min(v2,v3), max(v2,v3)))
                
                if e01 is not None and e02 is not None and e13 is not None and e23 is not None:
                    c4_idx = len(c4s)
                    c4_edges = (e01, e02, e13, e23)
                    c4s.append(c4_edges)
                    for e in c4_edges:
                        edge_to_c4s[e].append(c4_idx)
    
    # Now we have:
    # c4s: list of 4-tuples of edge indices
    # edge_to_c4s: for each edge, list of C4 indices
    
    # Strategy: Simulated Annealing or local search.
    # State: set of selected edges.
    # Objective: maximize |S| subject to no C4 has all 4 edges in S.
    
    # We'll use a greedy + local search approach.
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    best_edges = set()
    best_size = 0
    
    def is_valid_add(selected, e_idx):
        """Check if adding e_idx to selected creates a C4 with all 4 edges."""
        for c4_idx in edge_to_c4s[e_idx]:
            c4 = c4s[c4_idx]
            count = sum(1 for e in c4 if e in selected or e == e_idx)
            if count == 4:
                return False
        return True
    
    def greedy_random():
        """Randomized greedy construction."""
        selected = set()
        candidates = list(range(n_edges))
        random.shuffle(candidates)
        for e in candidates:
            if is_valid_add(selected, e):
                selected.add(e)
        return selected
    
    def local_search(selected, time_limit):
        """1-swap local search: try to remove one edge and add one or more."""
        selected = set(selected)
        size = len(selected)
        end_time = time.time() + time_limit
        improved = True
        
        while improved and time.time() < end_time:
            improved = False
            # Try all removals
            for e_remove in list(selected):
                if time.time() > end_time:
                    break
                new_set = selected - {e_remove}
                # Try to add edges
                added = 0
                # Try edges that were blocked by e_remove
                # Or just try all non-selected edges
                for e_add in range(n_edges):
                    if e_add in new_set:
                        continue
                    if is_valid_add(new_set, e_add):
                        new_set.add(e_add)
                        added += 1
                if len(new_set) > size:
                    selected = new_set
                    size = len(new_set)
                    improved = True
                    break
        
        return selected, len(selected)
    
    # Run multiple rounds
    while time.time() < deadline:
        # Greedy
        S = greedy_random()
        if len(S) > best_size:
            best_edges = S.copy()
            best_size = len(S)
        
        # Local search with remaining time
        remaining = deadline - time.time()
        if remaining > 2:
            S_ls, size_ls = local_search(S, remaining - 2)
            if size_ls > best_size:
                best_edges = S_ls
                best_size = size_ls
    
    # Convert to result
    result = []
    for e_idx in best_edges:
        u, v = edges[e_idx]
        result.append([u, v])
    
    return result
