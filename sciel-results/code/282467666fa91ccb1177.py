import random
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # Fallback: 8 disjoint K4s -> n=32, alpha=8, no C5
    def k4_blocks(nblocks):
        n = 4 * nblocks
        edges = []
        for b in range(nblocks):
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([4 * b + i, 4 * b + j])
        return {'n': n, 'edges': edges}

    fallback = k4_blocks(8)  # n = 32
    best = fallback

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5(adj, n):
        """
        Check if graph has a C5.
        A C5 exists if there is a sequence v0-v1-v2-v3-v4-v0 with distinct vertices.
        We can iterate over v0 and check if N(v0) contains an induced P3 (u-w-x) 
        such that u and x have a common neighbor outside {v0, u, w, x}? 
        Actually, simpler:
        For each v0:
          Let N0 = neighbors of v0.
          For each pair (a, b) in N0:
            If a ~ b:
              # Triangle v0-a-b. 
              # To form C5, we need b-c-d-v0. 
              # c in N(b) \ {v0, a}, d in N(c) \ {b}, d in N(v0) \ {a, c}
              # This is getting complex.
              
              # Alternative: Just enumerate all 5-cycles? n=35, C(35,5) is ~324k. 
              # For each set of 5 vertices, check if it contains a C5.
              # That's 324k * 5! checks? No, just check if the induced subgraph is C5 or supergraph.
              # Actually, "contains no 5-cycle as a subgraph" means no subset of 5 vertices 
              # induces a graph that has a C5 as a subgraph.
              # A graph on 5 vertices has a C5 subgraph iff it is isomorphic to C5, 
              # or C5+edge, or K5, etc.
              # Easiest: For each v0, check if there is a path of length 4 starting and ending at v0.
              # v0 - a - b - c - d - v0.
              # a in N(v0), b in N(a)\{v0}, c in N(b)\{v0,a}, d in N(c)\{v0,a,b}, d in N(v0)\{a,b,c}
              # and d != a, d != b.
              
              pass
        """
        # Fast check: 
        # A C5 is a cycle of length 5.
        # Iterate v0.
        for v0 in range(n):
            nbs0 = adj[v0]
            # List of neighbors
            nb_list = []
            tmp = nbs0
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                nb_list.append(v)
                tmp ^= lb
            
            # For each a in nb_list
            for a in nb_list:
                # b in N(a) \ {v0}
                nbs_a = adj[a] & ~(1 << v0)
                tmp_a = nbs_a
                while tmp_a:
                    lb_a = tmp_a & (-tmp_a)
                    b = lb_a.bit_length() - 1
                    tmp_a ^= lb_a
                    
                    # c in N(b) \ {v0, a}
                    nbs_b = adj[b] & ~(1 << v0) & ~(1 << a)
                    tmp_b = nbs_b
                    while tmp_b:
                        lb_b = tmp_b & (-tmp_b)
                        c = lb_b.bit_length() - 1
                        tmp_b ^= lb_b
                        
                        # d in N(c) \ {v0, a, b}
                        nbs_c = adj[c] & ~(1 << v0) & ~(1 << a) & ~(1 << b)
                        tmp_c = nbs_c
                        while tmp_c:
                            lb_c = tmp_c & (-tmp_c)
                            d = lb_c.bit_length() - 1
                            tmp_c ^= lb_c
                            
                            # Check if d is in N(v0) and d != a, d != b
                            # We already ensured d != a, d != b by mask? 
                            # nbs_c excluded a, b.
                            # Need d in N(v0).
                            if adj[v0] & (1 << d):
                                # Also need d != a, d != b (already done)
                                # And d != c (no self loop)
                                # And v0, a, b, c, d distinct.
                                # v0 distinct from others by construction.
                                # a, b, c, d distinct?
                                # b != a (mask), c != a, b (mask), d != a, b (mask).
                                # c != d? No self loop.
                                # So if adj[v0] & (1 << d) is true, we found a C5.
                                return True
        return False

    def alpha_at_least_9(adj, n, time_limit=2.0):
        """
        Check if alpha(G) >= 9.
        Equivalent to omega(complement) >= 9.
        Branch and bound on complement.
        """
        full = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = full ^ (adj[i] | (1 << i))
        
        # Order vertices by degree in complement (descending)
        order = sorted(range(n), key=lambda i: -bin(comp[i]).count('1'))
        pos = {v: i for i, v in enumerate(order)}
        comp2 = [0] * n
        for i in range(n):
            orig = order[i]
            mask = 0
            cm = comp[orig]
            tmp = cm
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                if v != orig:
                    mask |= (1 << pos[v])
                tmp ^= lb
            comp2[i] = mask
            
        def popcount(x):
            return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
        
        t0 = time.time()
        deadline_local = t0 + time_limit
        
        def rec(cand, k):
            if k >= 9:
                return True
            if time.time() > deadline_local:
                return None
            pc = popcount(cand)
            if pc + k < 9:
                return False
            
            # Greedy upper bound
            cb = cand
            gc = 0
            while cb:
                best_v = -1
                best_deg = -1
                cb2 = cb
                while cb2:
                    lb = cb2 & (-cb2)
                    v = lb.bit_length() - 1
                    deg = popcount(cb & comp2[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    cb2 ^= lb
                if best_v < 0:
                    break
                gc += 1
                cb &= comp2[best_v]
            
            if gc + k < 9:
                return False
            
            # Branch
            best_v = -1
            best_deg = -1
            cb2 = cand
            while cb2:
                lb = cb2 & (-cb2)
                v = lb.bit_length() - 1
                deg = popcount(cand & comp2[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                cb2 ^= lb
            
            v = best_v
            r = rec(cand & comp2[v], k + 1)
            if r is True:
                return True
            if r is None:
                return None
            return rec(cand & ~((1 << v)), k)

        return rec(full, 0)

    def is_valid(n, edges, time_limit=2.0):
        adj = build_adj(n, edges)
        if has_c5(adj, n):
            return False
        a9 = alpha_at_least_9(adj, n, time_limit)
        if a9 is True:
            return False
        if a9 is None:
            return False
        return True

    def find_c5_edge(adj, n):
        """Find an edge that is part of a C5, to remove it."""
        # Re-use logic from has_c5 but return the edge
        for v0 in range(n):
            nbs0 = adj[v0]
            nb_list = []
            tmp = nbs0
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                nb_list.append(v)
                tmp ^= lb
            
            for a in nb_list:
                nbs_a = adj[a] & ~(1 << v0)
                tmp_a = nbs_a
                while tmp_a:
                    lb_a = tmp_a & (-tmp_a)
                    b = lb_a.bit_length() - 1
                    tmp_a ^= lb_a
                    
                    nbs_b = adj[b] & ~(1 << v0) & ~(1 << a)
                    tmp_b = nbs_b
                    while tmp_b:
                        lb_b = tmp_b & (-tmp_b)
                        c = lb_b.bit_length() - 1
                        tmp_b ^= lb_b
                        
                        nbs_c = adj[c] & ~(1 << v0) & ~(1 << a) & ~(1 << b)
                        tmp_c = nbs_c
                        while tmp_c:
                            lb_c = tmp_c & (-tmp_c)
                            d = lb_c.bit_length() - 1
                            tmp_c ^= lb_c
                            
                            if adj[v0] & (1 << d):
                                # C5: v0-a-b-c-d-v0
                                # Edges: (v0,a), (a,b), (b,c), (c,d), (d,v0)
                                # Return one of them, e.g., (a,b)
                                return (min(a,b), max(a,b))
        return None

    def find_independent_set_9(adj, n, time_limit=2.0):
        """Find an independent set of size 9 in G, which is a clique of size 9 in complement."""
        full = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = full ^ (adj[i] | (1 << i))
        
        order = sorted(range(n), key=lambda i: -bin(comp[i]).count('1'))
        pos = {v: i for i, v in enumerate(order)}
        comp2 = [0] * n
        for i in range(n):
            orig = order[i]
            mask = 0
            cm = comp[orig]
            tmp = cm
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                if v != orig:
                    mask |= (1 << pos[v])
                tmp ^= lb
            comp2[i] = mask
            
        def popcount(x):
            return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
        
        t0 = time.time()
        deadline_local = t0 + time_limit
        found_set = []
        
        def rec(cand, k, path):
            if k >= 9:
                found_set.extend(path)
                return True
            if time.time() > deadline_local:
                return False
            pc = popcount(cand)
            if pc + k < 9:
                return False
            
            # Greedy bound
            cb = cand
            gc = 0
            while cb:
                best_v = -1
                best_deg = -1
                cb2 = cb
                while cb2:
                    lb = cb2 & (-cb2)
                    v = lb.bit_length() - 1
                    deg = popcount(cb & comp2[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    cb2 ^= lb
                if best_v < 0:
                    break
                gc += 1
                cb &= comp2[best_v]
            
            if gc + k < 9:
                return False
            
            best_v = -1
            best_deg = -1
            cb2 = cand
            while cb2:
                lb = cb2 & (-cb2)
                v = lb.bit_length() - 1
                deg = popcount(cand & comp2[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                cb2 ^= lb
            
            v = best_v
            if rec(cand & comp2[v], k + 1, path + [order[v]]):
                return True
            return rec(cand & ~((1 << v)), k, path)

        if rec(full, 0, []):
            # Map back to original indices
            # path contains indices in 'order' permutation
            # Wait, 'order' maps index in sorted list to original vertex.
            # In rec, v is index in comp2 (which is sorted index).
            # So path contains sorted indices.
            # We need original vertices.
            orig_set = [order[v] for v in found_set]
            return orig_set
        return None

    def repair(n, edges, max_iters=100, time_limit=10.0):
        t0 = time.time()
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u,v), max(u,v)))
        
        for _ in range(max_iters):
            if time.time() - t0 > time_limit:
                break
            
            adj = build_adj(n, list(edge_set))
            
            # Check C5
            c5_edge = find_c5_edge(adj, n)
            if c5_edge:
                edge_set.discard(c5_edge)
                continue
            
            # Check Alpha
            indep_set = find_independent_set_9(adj, n, 1.0)
            if indep_set:
                # Add an edge between two vertices in the independent set
                # Choose a random pair to add
                u, v = random.sample(indep_set, 2)
                edge_set.add((min(u,v), max(u,v)))
                continue
            
            # Valid
            return sorted(edge_set)
        
        return None

    # Try to find valid graphs for n=35, 34, 33
    for target in (35, 34, 33):
        if remaining() < 15.0:
            break
        
        t0 = time.time()
        limit = min(remaining() - 10.0, 20.0)
        found = None
        
        for attempt in range(1000):
            if time.time() - t0 > limit:
                break
            
            n = target
            # Start with 8 K4s and add extra vertices
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            
            # Add extra vertices 32..n-1
            # Connect them randomly to some vertices
            for v in range(32, n):
                # Connect to 3-5 random vertices
                num_conn = random.randint(3, 5)
                targets = random.sample(range(32), num_conn)
                for t in targets:
                    edges.append([v, t])
            
            # Randomly add some edges between blocks?
            # Let's just repair this initial structure
            repaired = repair(n, edges, max_iters=50, time_limit=5.0)
            if repaired:
                if is_valid(n, repaired, 2.0):
                    found = {'n': n, 'edges': repaired}
                    break
        
        if found:
            best = found
            break

    return best
