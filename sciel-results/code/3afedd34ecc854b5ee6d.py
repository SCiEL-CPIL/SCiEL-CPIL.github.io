"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_triangle_free_circulant(n, S):
    """
    Check if circulant graph C_n(S) is triangle-free.
    S is a set of offsets.
    Triangle exists if there are d1, d2, d3 in S such that d1+d2+d3 == 0 mod n.
    Note: The edges are undirected. The offsets in S are typically 1..n//2.
    The actual differences between adjacent vertices in a cycle u-v-w are d1, d2, d3.
    Since the graph is undirected, if d is in S, n-d is effectively the same edge.
    However, for the sum condition, we must consider the signed offsets.
    If u~v (diff d1), v~w (diff d2), w~u (diff d3).
    Then d1 + d2 + d3 = 0 mod n.
    The values d1, d2, d3 must be in the "difference set" D.
    If S = {s1, ..., sk}, then D = {s1, ..., sk, n-s1, ..., n-sk}.
    We need to check if there exist x, y, z in D such that x+y+z = 0 mod n and {x,y,z} corresponds to distinct vertices.
    Actually, simpler:
    Iterate all pairs (d1, d2) in S x S.
    Let d3 = (-d1 - d2) % n.
    If d3 is in S (or n-d3 is in S, i.e., d3 is a valid offset), then we have a triangle.
    We must ensure u, v, w are distinct.
    If d1=0 or d2=0 or d3=0, it's not a simple triangle (self-loop or multi-edge, but S doesn't have 0).
    If d1=d2=d3=0, impossible.
    If d1=d2, then d3 = -2d1. If d3 in S, and d1 != d3, we have vertices u, u+d1, u+2d1.
    If d1=d3, then 2d1+d2=0.
    If d2=d3, then d1+2d2=0.
    Basically, if (-d1-d2) % n is in S or (n - ((-d1-d2) % n)) % n is in S?
    No, S contains positive offsets. The difference w-u is (-d1-d2) % n.
    This difference corresponds to an edge if it is in S or its complement is in S.
    Since S is a set of offsets defining edges (u, u+s), the edge exists if diff is in S.
    So we check if d3 = (-d1 - d2) % n is in S.
    """
    S_set = set(S)
    for d1 in S:
        for d2 in S:
            d3 = (-d1 - d2) % n
            if d3 in S_set:
                # Check if vertices are distinct
                # d1, d2, d3 are the steps.
                # If d1=0, invalid (0 not in S).
                # If d1+d2=0 mod n, then d3=0. 0 not in S.
                # So if d3 in S, then d3 != 0.
                # Thus d1+d2 != 0.
                # Are vertices distinct?
                # v = u + d1
                # w = v + d2 = u + d1 + d2
                # u = w + d3 = u + d1 + d2 + d3 = u + 0.
                # Vertices are u, u+d1, u+d1+d2.
                # They are distinct if d1 != 0 and d2 != 0 and d1+d2 != 0.
                # Since d1, d2 in S (1..n/2), d1!=0, d2!=0.
                # d1+d2=0 mod n implies d1+d2=n.
                # If d1+d2=n, then d3=0, which is not in S.
                # So if d3 in S, vertices are distinct.
                return False
    return True

def _max_independent_set_size_circulant(n, S):
    """
    Exact branch and bound for maximum independent set size in C_n(S).
    """
    S_set = set(S)
    # Precompute adjacency mask for each vertex
    # adj[i] is a bitmask of neighbors
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for s in S:
            v1 = (i + s) % n
            v2 = (i - s) % n
            mask |= (1 << v1)
            mask |= (1 << v2)
        adj_mask[i] = mask
    
    best = 0
    
    def branch(candidates_mask, size):
        nonlocal best
        if size + bin(candidates_mask).count('1') <= best:
            return
        if not candidates_mask:
            if size > best:
                best = size
            return
        
        # Pick a vertex
        # Heuristic: vertex with max degree in candidates
        max_deg = -1
        chosen = -1
        cm = candidates_mask
        v_idx = 0
        while cm:
            if cm & 1:
                deg = bin(adj_mask[v_idx] & candidates_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    chosen = v_idx
            cm >>= 1
            v_idx += 1
            
        if chosen == -1:
            return

        # Branch 1: Include chosen
        # New candidates: candidates without chosen and its neighbors
        new_cand = candidates_mask & ~(adj_mask[chosen] | (1 << chosen))
        branch(new_cand, size + 1)
        
        # Branch 2: Exclude chosen
        new_cand = candidates_mask & ~(1 << chosen)
        branch(new_cand, size)

    branch((1 << n) - 1, 0)
    return best

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Fallback: Andrásfai graph And(9) on 26 vertices
    n_seed = 26
    best_n = n_seed
    best_edges = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    
    # Helper to convert circulant offsets to edge list
    def circulant_to_edges(n, S):
        edges = []
        for u in range(n):
            for s in S:
                v = (u + s) % n
                if u < v:
                    edges.append([u, v])
        return edges

    # Try n from 40 down to 27
    # We focus on circulant graphs.
    # For n=40, offsets 1..19.
    # For n=39, offsets 1..19.
    
    # To make the search efficient, we can use a random subset search first, 
    # or iterate through subsets of a certain size.
    # The degree should be high enough to keep alpha < 10.
    # For n=40, if alpha < 10, then alpha/n < 0.25.
    # Turan bound for independent set: alpha <= n / (d+1) is not tight.
    # Ramsey bound: R(3,10) <= 41.
    # Known graphs for R(3,10) are often circulant.
    
    for target_n in [40, 39, 38, 37]:
        if time.time() > deadline - 10:
            break
            
        max_offset = target_n // 2
        found = False
        
        # Strategy: Random sampling of subsets of offsets
        # We need the graph to be dense enough.
        # Let's try sizes k from max_offset down to 3.
        # Or just random subsets of size ~ n/4 to n/3.
        
        attempts = 0
        max_attempts = 20000 # Limit to avoid timeout, but bitset check is fast
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline - 5:
                break
                
            # Choose a random size k
            # For n=40, max k=19.
            # If k is too small, alpha is large.
            # If k is too large, triangles likely.
            # Let's sample k uniformly from [max_offset//2, max_offset]
            min_k = max(3, max_offset // 2)
            max_k = max_offset
            k = random.randint(min_k, max_k)
            
            # Sample k offsets from 1..max_offset
            S = random.sample(range(1, max_offset + 1), k)
            
            # Check triangle-free
            if not _is_triangle_free_circulant(target_n, S):
                continue
                
            # Check independent set
            max_ind = _max_independent_set_size_circulant(target_n, S)
            
            if max_ind < 10:
                # Found valid graph
                edges = circulant_to_edges(target_n, S)
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                found = True
                break
        
        if found and best_n >= target_n:
            # If we found a graph for this n, and n is the highest we are trying, we can stop?
            # If we found 40, we stop.
            # If we found 39, we stop? Yes, because we iterate 40, 39...
            # If we failed 40, we try 39. If we find 39, it's the best so far.
            # We should break the outer loop if we found the current target_n?
            # Yes, because we are going downwards.
            break

    # If we didn't find anything better, return the seed
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
