import time
import random
import sys

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1)) # 448
    
    # Edge mapping
    # Edge (u, v) where v = u ^ (1 << i) and u < v
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v, i)) # store dim too
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx

    # C4 mapping
    # A C4 is defined by two dimensions i, j and a "base" vertex x 
    # such that x has 0 in bits i and j? 
    # Actually, a face is defined by fixing d-2 bits.
    # Let's enumerate all C4s.
    c4s = []
    c4_index = {}
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of the other d-2 bits
            # There are 2^(d-2) such faces for each pair (i,j)
            # The face vertices are x, x^i, x^j, x^i^j
            # where x has 0 in bits i and j? No, x can be any vertex, 
            # but to avoid duplicates, we can fix bits i and j to 0 in the "base" representation?
            # Actually, just iterate u from 0 to 127.
            # A face is uniquely identified by the set of 4 vertices.
            pass
            
    # Better enumeration:
    # For each pair of dims (i, j):
    #   For each vertex u:
    #     If u has bit i = 0 and bit j = 0, then u, u^i, u^j, u^i^j form a unique face.
    #     This generates each face exactly once.
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n_vertices):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_index[(v1, v2)]
                    e2 = edge_index[(v2, v4)]
                    e3 = edge_index[(v4, v3)]
                    e4 = edge_index[(v3, v1)]
                    
                    c4_edges = [e1, e2, e3, e4]
                    c4_idx = len(c4s)
                    c4s.append(c4_edges)

    n_c4s = len(c4s) # Should be 672

    # Incidence: for each edge, list of c4 indices
    edge_to_c4s = [[] for _ in range(n_edges)]
    for c_idx, c4_edges in enumerate(c4s):
        for e_idx in c4_edges:
            edge_to_c4s[e_idx].append(c_idx)

    # State
    # R: set of removed edges (indices)
    # hit_count: array of length n_c4s, count of removed edges in that c4
    
    # Initialization: Greedy
    R = set()
    hit_count = [0] * n_c4s
    
    # We need to hit all C4s.
    # Greedy: pick random C4, pick random edge from it, add to R.
    unhit_c4s = set(range(n_c4s))
    while unhit_c4s:
        c_idx = random.choice(list(unhit_c4s))
        # Pick a random edge from this C4
        e_idx = random.choice(c4s[c_idx])
        
        # Add to R
        R.add(e_idx)
        
        # Update hit counts for all C4s containing e_idx
        for c in edge_to_c4s[e_idx]:
            hit_count[c] += 1
            if hit_count[c] > 0:
                unhit_c4s.discard(c)
                
    # Local Search
    best_removed = len(R)
    current_removed = len(R)
    
    start_time = time.time()
    deadline = start_time + 110 # 110s limit to be safe
    
    # Precompute list of edges in R for faster iteration?
    # R is a set, iteration is fine.
    
    while time.time() < deadline:
        # Try to remove an edge
        if not R:
            break
            
        # Pick random edge from R
        e_idx = random.choice(list(R))
        
        # Check if we can remove it
        # Find all C4s that would become unhit
        newly_unhit = []
        for c_idx in edge_to_c4s[e_idx]:
            if hit_count[c_idx] == 1:
                newly_unhit.append(c_idx)
                
        if not newly_unhit:
            # Safe to remove
            R.remove(e_idx)
            for c_idx in edge_to_c4s[e_idx]:
                hit_count[c_idx] -= 1
            current_removed -= 1
            if current_removed < best_removed:
                best_removed = current_removed
        else:
            # Try to swap
            # Find an edge f not in R that is in all newly_unhit C4s
            # Take the first C4 in newly_unhit
            c0 = newly_unhit[0]
            candidates = [e for e in c4s[c0] if e != e_idx]
            
            found_swap = False
            for f_idx in candidates:
                if f_idx in R:
                    continue
                # Check if f_idx is in all other newly_unhit C4s
                valid = True
                for c_idx in newly_unhit[1:]:
                    if f_idx not in c4s[c_idx]:
                        valid = False
                        break
                if valid:
                    # Perform swap
                    R.remove(e_idx)
                    R.add(f_idx)
                    
                    # Update counts for e_idx (decrement)
                    for c_idx in edge_to_c4s[e_idx]:
                        hit_count[c_idx] -= 1
                    # Update counts for f_idx (increment)
                    for c_idx in edge_to_c4s[f_idx]:
                        hit_count[c_idx] += 1
                        
                    found_swap = True
                    break
            
            if not found_swap:
                # Kick: add a random edge to R to change landscape
                # Pick a random edge not in R
                # This increases |R|, but might allow future reductions
                if current_removed < n_edges - 1:
                    # Pick random edge not in R
                    # Simple way: pick random edge, if in R, try again (rare)
                    while True:
                        kick_e = random.randint(0, n_edges - 1)
                        if kick_e not in R:
                            break
                    R.add(kick_e)
                    for c_idx in edge_to_c4s[kick_e]:
                        hit_count[c_idx] += 1
                    current_removed += 1

    # Construct output
    # S = E \ R
    out = []
    for i in range(n_edges):
        if i not in R:
            u, v, _ = edges[i]
            out.append([u, v])
            
    return out
