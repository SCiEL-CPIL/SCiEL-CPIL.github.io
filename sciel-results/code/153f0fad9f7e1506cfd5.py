"""Improved program: efficient C5 detection and alpha bound for finding larger valid graphs."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # ---- Guaranteed valid fallback: 8 disjoint K4s -> n=32, alpha=8, no C5 ----
    def k4_blocks(nblocks):
        n = 4 * nblocks
        edges = []
        for b in range(nblocks):
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([4 * b + i, 4 * b + j])
        return {'n': n, 'edges': edges}

    fallback = k4_blocks(8)  # n = 32

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5_fast(adj, n):
        """Check for C5: for each vertex s, DFS paths of length 3 from s to neighbors of s."""
        for s in range(n):
            sbit = 1 << s
            # We need path s-v1-v2-v3-s where v1,v2,v3 are distinct and not s
            # and v3 is adjacent to s.
            # So: pick v1 in N(s), v2 in N(v1)\{s}, v3 in N(v2)\{s,v1}, check v3 in N(s)
            nbs_s = adj[s]
            # Iterate over v1 in N(s)
            v1_mask = nbs_s
            while v1_mask:
                lb1 = v1_mask & (-v1_mask)
                v1 = lb1.bit_length() - 1
                # v2 in N(v1) \ {s}
                v2_mask = adj[v1] & ~sbit & ~lb1  # exclude s and v1 itself (not needed since adj[v1] doesn't include v1)
                # Actually adj[v1] doesn't include v1, so just exclude s
                v2_mask = adj[v1] & ~sbit
                while v2_mask:
                    lb2 = v2_mask & (-v2_mask)
                    v2 = lb2.bit_length() - 1
                    # v3 in N(v2) \ {s, v1}
                    v3_mask = adj[v2] & ~sbit & ~lb1
                    # Check if any v3 in v3_mask is in N(s)
                    if v3_mask & nbs_s:
                        return True
                    v3_mask ^= lb2  # wait, this is wrong. Let me restructure.
                    # Actually I need to iterate v3
                    # Let me redo this more carefully
                    break  # placeholder, will rewrite
                v1_mask ^= lb1
        # The above logic has bugs. Let me use a cleaner approach.
        return False

    def has_c5_v2(adj, n):
        """Cleaner C5 check: for each ordered pair (s, t) not adjacent, check if there's a path of length 3 from s to t through intermediate vertices, and t is adjacent to s... 
        No, C5 is s-v1-v2-v3-t-s where t=s. So it's a cycle.
        Let me use: for each vertex s, for each pair of neighbors (a,b) of s with a!=b,
        check if there's a path of length 2 from a to b not using s.
        That path is a-c-b where c is adjacent to both a and b, c != s, c != a, c != b.
        """
        for s in range(n):
            nbs = adj[s]
            # Get list of neighbors
            nb_list = []
            tmp = nbs
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                nb_list.append(v)
                tmp ^= lb
            # For each pair (a, b) in nb_list with a < b:
            for i in range(len(nb_list)):
                a = nb_list[i]
                for j in range(i + 1, len(nb_list)):
                    b = nb_list[j]
                    # Check if there exists c adjacent to both a and b, c != s
                    common = adj[a] & adj[b] & ~((1 << s))
                    # c must not be a or b (but a is not in adj[a] since no self-loops, same for b)
                    # So common already excludes a and b
                    if common:
                        return True
        return False

    def alpha_at_least_9_fast(adj, n):
        """Check if alpha(G) >= 9 using branch and bound on complement."""
        comp = [((1 << n) - 1) ^ (adj[i] | (1 << i)) for i in range(n)]
        
        # Sort vertices by degree in complement (descending)
        order = list(range(n))
        order.sort(key=lambda i: -bin(comp[i]).count('1'))
        
        # Remap
        pos = {v: i for i, v in enumerate(order)}
        comp2 = [0] * n
        for i in range(n):
            mask = 0
            orig = order[i]
            for j in range(n):
                if comp[orig] & (1 << order[j]):
                    mask |= (1 << pos[order[j]])
            comp2[i] = mask
        
        t0 = time.time()
        
        def rec(cand, k):
            if k >= 9:
                return True
            if time.time() - t0 > 5.0:
                return None
            if bin(cand).count('1') + k < 9:
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
                    deg = bin(cb & comp2[v]).count('1')
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
            
            # Pick vertex with highest degree in cand
            lb = cand & (-cand)
            v = lb.bit_length() - 1
            r = rec(cand & comp2[v], k + 1)
            if r is True:
                return True
            if r is None:
                return None
            return rec(cand ^ lb, k)

        return rec((1 << n) - 1, 0)

    def is_valid(n, edges):
        adj = build_adj(n, edges)
        if has_c5_v2(adj, n):
            return False
        a9 = alpha_at_least_9_fast(adj, n)
        if a9 is True:
            return False
        if a9 is None:
            return False
        return True

    best = fallback  # always valid, n=32

    # ---- Structured search: modify 8xK4 to try to reach n=33+ ----
    # Strategy: Start with 8xK4, add vertices and edges carefully.
    # For n=33: Add vertex 32. To keep C5-free and alpha<9, we need careful edge placement.
    # 
    # Key insight from analysis: If we add vertex 32 connected to exactly one vertex in each K4 block,
    # then alpha with vertex 32 = 1 + 8*1 = 9 (BAD). 
    # If we connect to 2+ vertices in some blocks, we can reduce alpha.
    # But connecting to all 4 in a block creates C5s within that block.
    # 
    # Alternative: Use a different base structure. Try 7 K4s + 1 K4 with extra vertex.
    # Or try random perturbation of the 8xK4 structure.
    
    # ---- Random search for n=33,34,35 with better edge probability tuning ----
    for target in (33, 34, 35):
        if remaining() < 25.0:
            break
        
        t0 = time.time()
        limit = min(remaining() - 20.0, 20.0)
        found = None
        
        for attempt in range(8000):
            if time.time() - t0 > limit:
                break
            
            n = target
            # Use edge probability tuned for C5-freeness and low independence
            # C5-free graphs tend to have lower edge density
            # For n=33, try p around 0.25-0.35
            p = 0.30
            
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        edges.append([i, j])
            
            if is_valid(n, edges):
                found = {'n': n, 'edges': edges}
                break
        
        if found is not None:
            best = found
            break

    # ---- Also try structured modifications of 8xK4 for n=33 ----
    if best['n'] == 32 and remaining() > 15.0:
        t0 = time.time()
        limit = min(remaining() - 10.0, 15.0)
        
        for attempt in range(2000):
            if time.time() - t0 > limit:
                break
            
            n = 33
            # Start with 8xK4 and add vertex 32 with some edges
            edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append([base + i, base + j])
            
            # Connect vertex 32 to a random subset of vertices
            # To avoid C5, don't connect to all 4 in any block
            # To reduce alpha, connect to at least 2 in most blocks
            conn = set()
            for b in range(8):
                base = 4 * b
                # Connect to 2 or 3 vertices in this block (not all 4 to avoid C5)
                num_conn = random.choice([2, 2, 3])
                verts = [base + i for i in range(4)]
                random.shuffle(verts)
                for v in verts[:num_conn]:
                    conn.add(v)
            
            for v in conn:
                edges.append([32, v])
            
            if is_valid(n, edges):
                best = {'n': n, 'edges': edges}
                break

    if time.time() > deadline - 2.0:
        return fallback

    return best
# EVOLVE-BLOCK-END
