import time
import random
import sys

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Attempts to find a valid graph with n=33 using fast local search.
    Falls back to the known valid n=32 construction (8 disjoint K4s).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s buffer for safety/overhead
    
    # Fallback: 8 disjoint K4s, n=32
    n_fallback = 32
    edges_fallback = []
    for block_idx in range(8):
        start = block_idx * 4
        verts = [start, start + 1, start + 2, start + 3]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                edges_fallback.append([verts[i], verts[j]])
    
    # Try to find n=33
    # n=33, vertices 0..32
    # Base: 8 K4s (0..31)
    # Vertex 32 is the new one.
    
    n = 33
    # Bitset representation for adjacency
    # adj[i] is an integer where bit j is 1 if i~j
    
    def check_c5_and_alpha(adj, n):
        """
        Returns (is_c5_free, alpha)
        Uses bitsets for speed.
        """
        # 1. Check C5
        # A C5 exists if there is a path of length 4 between u and v 
        # such that u~v, and the internal vertices are distinct and not u,v.
        # More efficiently: for each edge (u,v), check if there is a path of length 3 from v to u
        # avoiding u and v.
        
        # Precompute neighbors as bitsets
        # adj is already bitsets
        
        # To check if adding edge (u,v) creates C5, we check if there is a path u-a-b-c-v
        # where a,b,c distinct, not u,v.
        # This is equivalent to: (N(u) & N(v)) is not enough.
        # We need N(u) \ {v} and N(v) \ {u}.
        # Path u-a-b-c-v.
        # a in N(u), c in N(v).
        # b in N(a) & N(c).
        
        # Faster C5 check:
        # For each pair (u,v) that are NOT adjacent? No, C5 uses existing edges.
        # We are building the graph. We check the final graph.
        
        # Efficient C5 check:
        # For each vertex u, consider pairs of neighbors (a, c) of u.
        # If there is a path a-b-c of length 2 (i.e., common neighbor b)
        # such that b is not u, and a != c, and b != a, b != c.
        # Then u-a-b-c-u is a C5? No, that's a C4 if a~c.
        # If a and c are not adjacent, and b is a common neighbor,
        # then u-a-b-c-u is a C5 if u is not adjacent to a,b,c (except a,c) and a,b,c distinct.
        # Wait, u is adjacent to a and c.
        # Cycle: u-a-b-c-u.
        # Edges: (u,a), (a,b), (b,c), (c,u).
        # Vertices: u,a,b,c. This is a C4.
        
        # C5: u-v-a-b-c-u.
        # Edges: (u,v), (v,a), (a,b), (b,c), (c,u).
        # Check: For each edge (u,v), is there a path of length 3 from v to u
        # with internal vertices not u or v?
        
        # Let's iterate over all edges (u,v) in the graph.
        # For each edge, check if there exists a path v-a-b-c-u.
        # a in N(v)\{u}, c in N(u)\{v}, b in N(a)\{v,c} & N(c)\{u,a}.
        
        # This is O(E * N^3) in worst case, but with bitsets it's fast.
        
        for u in range(n):
            # Get neighbors of u
            nu = adj[u]
            # Iterate over neighbors v of u
            nv_bits = nu
            while nv_bits:
                lsb = nv_bits & -nv_bits
                v = lsb.bit_length() - 1
                nv_bits ^= lsb
                
                # We have edge (u,v). Check for path v-a-b-c-u.
                # a must be in N(v) \ {u}
                # c must be in N(u) \ {v}
                # b must be in N(a) \ {v,c} & N(c) \ {u,a}
                
                # Optimization:
                # Let S_a = N(v) & ~ (1<<u)
                # Let S_c = N(u) & ~ (1<<v)
                # For each a in S_a, for each c in S_c:
                #   if a == c: continue
                #   if N(a) & N(c) & ~(1<<u) & ~(1<<v) & ~(1<<a) & ~(1<<c) != 0:
                #       return False (C5 found)
                
                s_a = adj[v] & ~(1 << u)
                s_c = adj[u] & ~(1<<v)
                
                # Iterate a in s_a
                a_bits = s_a
                while a_bits:
                    lsb_a = a_bits & -a_bits
                    a = lsb_a.bit_length() - 1
                    a_bits ^= lsb_a
                    
                    # Iterate c in s_c
                    c_bits = s_c
                    while c_bits:
                        lsb_c = c_bits & -c_bits
                        c = lsb_c.bit_length() - 1
                        c_bits ^= lsb_c
                        
                        if a == c:
                            continue
                        
                        # Check for b
                        # b in N(a) & N(c)
                        # b not in {u, v, a, c}
                        common = adj[a] & adj[c]
                        # Mask out u, v, a, c
                        mask = (1 << u) | (1 << v) | (1 << a) | (1 << c)
                        if common & ~mask:
                            return False, 0 # C5 found
        
        # 2. Check Alpha
        # Max Independent Set
        # Branch and bound with bitsets
        best_alpha = [0]
        
        def dfs(candidates, size):
            if size + candidates.bit_count() <= best_alpha[0]:
                return
            if candidates == 0:
                if size > best_alpha[0]:
                    best_alpha[0] = size
                return
            
            # Pick a vertex v from candidates
            # Heuristic: pick vertex with highest degree in the induced subgraph?
            # Or just pick the first one.
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            # Branch 1: Include v
            # Remove v and its neighbors from candidates
            new_cands = candidates & ~adj[v]
            dfs(new_cands, size + 1)
            
            # Branch 2: Exclude v
            dfs(candidates, size)
        
        dfs((1 << n) - 1, 0)
        
        return True, best_alpha[0]

    def try_find_n33():
        rng = random.Random(42)
        
        # Initial setup: 8 K4s
        adj = [0] * n
        for block_idx in range(8):
            start = block_idx * 4
            for i in range(4):
                for j in range(i+1, 4):
                    u = start + i
                    v = start + j
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Vertex 32 is initially isolated.
        
        # We will try to add edges from 32 to the base, and between base blocks.
        # Strategy: Random restarts.
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Reset adj to base
            adj = [0] * n
            for block_idx in range(8):
                start = block_idx * 4
                for i in range(4):
                    for j in range(i+1, 4):
                        u = start + i
                        v = start + j
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Randomly add edges from 32 to base
            # We need to reduce alpha.
            # Let's try to connect 32 to many vertices.
            # But avoid C5.
            
            # Heuristic: Connect 32 to a random subset of each K4.
            for block_idx in range(8):
                start = block_idx * 4
                # Pick 2 or 3 random vertices from this block to connect to 32
                k = rng.choice([2, 3, 3, 4]) # Bias towards 3 or 4
                # If k=4, C5 risk inside block.
                # If k=3, Alpha risk.
                
                # Let's just pick a random subset
                verts = [start, start+1, start+2, start+3]
                rng.shuffle(verts)
                for v in verts[:k]:
                    adj[32] |= (1 << v)
                    adj[v] |= (1 << 32)
            
            # Check validity
            is_c5_free, alpha = check_c5_and_alpha(adj, n)
            if is_c5_free and alpha < 9:
                # Extract edges
                edges = []
                for u in range(n):
                    v_bits = adj[u]
                    while v_bits:
                        lsb = v_bits & -v_bits
                        v = lsb.bit_length() - 1
                        v_bits ^= lsb
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            # If not valid, try to fix by adding more edges?
            # Adding edges reduces alpha but increases C5 risk.
            # Let's try a local search: randomly add edges, check, revert if C5.
            
            # Actually, the above random initialization is too sparse for alpha < 9
            # if k is small.
            # If k=3 for all, alpha is likely 9.
            # We need to add edges between blocks to break the independent sets.
            
            # Let's try adding random edges between blocks.
            for step in range(50):
                if time.time() > deadline:
                    break
                
                # Pick two random vertices from different blocks
                b1 = rng.randint(0, 7)
                b2 = rng.randint(0, 7)
                if b1 == b2:
                    continue
                u = b1 * 4 + rng.randint(0, 3)
                v = b2 * 4 + rng.randint(0, 3)
                
                if adj[u] & (1 << v):
                    continue
                
                # Tentatively add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                is_c5_free, alpha = check_c5_and_alpha(adj, n)
                
                if not is_c5_free:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                elif alpha < 9:
                    # Success
                    edges = []
                    for u_idx in range(n):
                        v_bits = adj[u_idx]
                        while v_bits:
                            lsb = v_bits & -v_bits
                            v_idx = lsb.bit_length() - 1
                            v_bits ^= lsb
                            if u_idx < v_idx:
                                edges.append([u_idx, v_idx])
                    return {'n': n, 'edges': edges}
                else:
                    # Alpha still >= 9, keep edge? 
                    # Adding edges never increases alpha.
                    # So keeping it is safe for alpha, but might create C5 later.
                    # Since we checked C5 and it's free, we can keep it.
                    pass
                    
    try:
        result = try_find_n33()
        if result:
            return result
    except Exception:
        pass
        
    return {'n': n_fallback, 'edges': edges_fallback}

if __name__ == "__main__":
    r = construct()
    print(r['n'])
