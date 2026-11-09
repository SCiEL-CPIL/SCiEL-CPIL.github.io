import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Precompute all 8-vertex subsets and their Q3-checking data
    # Q3 structure: vertices labeled 0..7 (binary 000..111)
    # Edges between vertices differing in exactly one bit
    # Bipartition: even parity {0,3,5,6} vs odd parity {1,2,4,7}
    
    # For each 8-vertex subset, we need to check if it contains Q3 as a subgraph
    # This means: is there a bijection from the 8 vertices to {0..7} such that
    # all 12 cube edges are present in the graph?
    
    # Key insight: Q3 is determined by its bipartite structure.
    # For a given 8-vertex set, try all ways to partition into two 4-sets (A, B),
    # then check if the bipartite graph G[A,B] contains Q3 as a bipartite subgraph.
    
    # Q3 bipartite adjacency (A=even, B=odd):
    # A[0]=000 connects to B: 001,010,100 -> indices 0,1,3 in B={001,010,111,100}
    # A[1]=011 connects to B: 001,010,111 -> indices 0,1,2
    # A[2]=101 connects to B: 001,111,100 -> indices 0,2,3
    # A[3]=110 connects to B: 010,111,100 -> indices 1,2,3
    
    # Precompute the 12 required edge patterns for each of the 24 permutations of B
    # and for each of the 24 permutations of A (when swapping roles)
    
    perms24 = list(itertools.permutations(range(4)))
    idx4_of_8 = list(itertools.combinations(range(8), 4))
    
    # For a fixed partition (A_indices, B_indices) of {0..7}, and a permutation p of B,
    # the required edges (in terms of positions in A and B) are:
    # A[0] -> B[p[0]], B[p[1]], B[p[3]]
    # A[1] -> B[p[0]], B[p[1]], B[p[2]]
    # A[2] -> B[p[0]], B[p[2]], B[p[3]]
    # A[3] -> B[p[1]], B[p[2]], B[p[3]]
    
    # Precompute required neighbor masks for each permutation
    # req_A[p] = list of 4 tuples, each tuple has the 3 B-indices that A[i] must connect to
    req_A = []
    for p in perms24:
        req = [
            (p[0], p[1], p[3]),
            (p[0], p[1], p[2]),
            (p[0], p[2], p[3]),
            (p[1], p[2], p[3])
        ]
        req_A.append(req)
    
    # Similarly for swapped roles: B[0] -> A[p[0]], A[p[1]], A[p[3]], etc.
    # This is the same pattern but transposed
    req_B = req_A  # Same structure, just applied to B vertices checking A neighbors
    
    # Precompute all 8-vertex subsets
    all_8sets = list(itertools.combinations(range(16), 8))
    
    # For each 8-set, precompute all (partition, permutation) checks
    # We'll store: for each 8-set, a list of (A_tuple, B_tuple, req_pattern) 
    # where A_tuple and B_tuple are 4-tuples of vertex indices
    # and req_pattern is which permutation to use
    
    # Actually, let's just precompute for each 8-set the list of all (A, B, p) triples
    # that we need to check. There are C(8,4)*24*2 = 70*24*2 = 3360 checks per 8-set.
    # With C(16,8) = 12870 8-sets, that's ~42 million checks total if we do them all.
    # Too many to precompute. Instead, we'll compute on the fly but efficiently.
    
    # Better approach: for the greedy search, when we add edge (u,v), we only need
    # to check 8-sets containing both u and v. There are C(14,6) = 3003 such sets.
    # For each such set, we check if it contains Q3.
    
    # Let's precompute for each pair (u,v) the list of 6-tuples of other vertices
    # that would form an 8-set with u and v
    pair_to_6sets = {}
    for u in range(16):
        for v in range(u+1, 16):
            others = [x for x in range(16) if x != u and x != v]
            pair_to_6sets[(u,v)] = list(itertools.combinations(others, 6))
    
    # Helper function to check if a specific 8-vertex set contains Q3
    def contains_q3_in_set(v8, adj_mask):
        """Check if the induced subgraph on v8 contains Q3 as a subgraph."""
        v_list = list(v8)
        
        for (i, j, k, l) in idx4_of_8:
            A = (v_list[i], v_list[j], v_list[k], v_list[l])
            B = tuple(v_list[idx] for idx in range(8) if idx not in (i, j, k, l))
            
            # Check orientation: A=even, B=odd
            for pi, req in enumerate(req_A):
                # Check A[0]
                m = adj_mask[A[0]]
                if not (m & (1 << B[req[0][0]])): continue
                if not (m & (1 << B[req[0][1]])): continue
                if not (m & (1 << B[req[0][2]])): continue
                # Check A[1]
                m = adj_mask[A[1]]
                if not (m & (1 << B[req[1][0]])): continue
                if not (m & (1 << B[req[1][1]])): continue
                if not (m & (1 << B[req[1][2]])): continue
                # Check A[2]
                m = adj_mask[A[2]]
                if not (m & (1 << B[req[2][0]])): continue
                if not (m & (1 << B[req[2][1]])): continue
                if not (m & (1 << B[req[2][2]])): continue
                # Check A[3]
                m = adj_mask[A[3]]
                if not (m & (1 << B[req[3][0]])): continue
                if not (m & (1 << B[req[3][1]])): continue
                if not (m & (1 << B[req[3][2]])): continue
                return True
            
            # Check orientation: A=odd, B=even (swap roles)
            for pi, req in enumerate(req_A):
                # Check B[0]
                m = adj_mask[B[0]]
                if not (m & (1 << A[req[0][0]])): continue
                if not (m & (1 << A[req[0][1]])): continue
                if not (m & (1 << A[req[0][2]])): continue
                # Check B[1]
                m = adj_mask[B[1]]
                if not (m & (1 << A[req[1][0]])): continue
                if not (m & (1 << A[req[1][1]])): continue
                if not (m & (1 << A[req[1][2]])): continue
                # Check B[2]
                m = adj_mask[B[2]]
                if not (m & (1 << A[req[2][0]])): continue
                if not (m & (1 << A[req[2][1]])): continue
                if not (m & (1 << A[req[2][2]])): continue
                # Check B[3]
                m = adj_mask[B[3]]
                if not (m & (1 << A[req[3][0]])): continue
                if not (m & (1 << A[req[3][1]])): continue
                if not (m & (1 << A[req[3][2]])): continue
                return True
        
        return False
    
    # Initial graph: K_{3,13}
    # U = {0, 1, 2}, V = {3, ..., 15}
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))
    
    # Build adjacency masks
    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    all_pairs = set(itertools.combinations(range(16), 2))
    
    def degree(v):
        return bin(adj_mask[v]).count('1')
    
    def try_add_edge(u, v):
        """Try to add edge (u,v). Returns True if safe."""
        if (u, v) in edges:
            return True
        if (v, u) in edges:
            return True
        
        # Temporarily add
        edges.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        # Check all 8-sets containing u and v
        for combo6 in pair_to_6sets[(u, v)]:
            v8 = (u, v) + combo6
            if contains_q3_in_set(v8, adj_mask):
                # Remove and return False
                edges.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        
        return True
    
    # Greedy addition with multiple passes
    # Use randomized restarts to escape local optima
    best_edges = set(edges)
    best_count = len(edges)
    
    # Multiple passes
    for pass_num in range(20):
        if time.time() > deadline - 5:
            break
        
        # Get current non-edges
        non_edges = [e for e in all_pairs if e not in edges]
        
        # Sort by combined degree (higher first) for deterministic pass
        if pass_num % 2 == 0:
            non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
        else:
            # Random shuffle for diversity
            random.seed(pass_num * 12345)
            random.shuffle(non_edges)
        
        changed = False
        for e in non_edges:
            if time.time() > deadline - 3:
                break
            u, v = e
            if try_add_edge(u, v):
                changed = True
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = set(edges)
        
        if not changed:
            break
    
    # Restore best if current is worse (shouldn't happen since we only add edges)
    if len(best_edges) > len(edges):
        edges = best_edges
        # Rebuild adj_mask
        adj_mask = [0] * 16
        for (u, v) in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
    
    return [list(e) for e in edges]
