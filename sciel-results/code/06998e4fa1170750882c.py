import random
import sys

def construct():
    n = 40
    best_edges = 0
    best_adj_masks = [0] * n
    
    # Helper to check if adding edge (u, v) creates a C6
    # Uses bitmasks for speed
    def can_add(u, v, masks):
        # Path u-a-b-c-d-v
        # a in N(u), b in N(a), c in N(b), d in N(c) AND d in N(v)
        # a, b, c, d distinct and not u, v
        
        # Iterate a in N(u)
        mask_a = masks[u]
        # Remove v from potential a's? No, if a==v, path is u-v... which is the edge itself. 
        # But we are checking if there is a path of length 4. 
        # If a==v, then we look for b in N(v). Path u-v-b-c-d-v? No, d must connect to v.
        # The path is u-a-b-c-d-v.
        # If a==v, then u-v-b-c-d-v is a walk, but we need a simple cycle.
        # If a==v, the first step is the edge we are adding. The rest must be a path of length 3 from v to v? No.
        # The condition is: does a path of length 4 exist in G \ {(u,v)}?
        # So we check on the current masks which do NOT contain (u,v).
        
        # So a can be any neighbor of u.
        # b can be any neighbor of a (except u, to avoid backtracking immediately? 
        # Actually, simple path u-a-b... implies b != u. 
        # Also b != a.
        # c != a, c != b.
        # d != c, d != b, d != a, d != u, d != v.
        
        # Let's iterate through bits.
        # This might still be slow in Python if we do it naively.
        # Number of edges in C6-free graph is ~105. Density is low.
        # N(u) size is ~5-10.
        # Loop sizes: 10 * 10 * 10 = 1000 iterations per edge check.
        # 1000 * 1000 (candidates) * 10 (runs) = 10M ops. Should be fine.
        
        mu = masks[u]
        mv = masks[v]
        
        # Iterate a in mu
        a = mu
        while a:
            lb = a & -a
            a ^= lb
            ai = lb.bit_length() - 1
            if ai == v: continue # a cannot be v? 
            # If a == v, path is u-v-b-c-d-v. 
            # The segment v-b-c-d-v is a path of length 3 from v to v? 
            # No, it's a cycle. 
            # If a==v, we are looking for a path v-b-c-d-v of length 3 in the rest of the graph?
            # No, the path is u-a-b-c-d-v. If a=v, it's u-v-b-c-d-v.
            # This forms a cycle u-v-b-c-d-u? No, d connects to v.
            # So u-v-b-c-d-v is not a simple cycle unless u=d? No.
            # The cycle would be u-v-b-c-d-...-u.
            # If a=v, the path starts u-v. Then we need v-b-c-d-v? That's a closed walk.
            # We need a simple path from u to v of length 4.
            # If a=v, the path is u-v-b-c-d-v. The vertices are u,v,b,c,d,v. 
            # v is repeated. Not a simple path.
            # So a != v.
            
            ma = masks[ai]
            # b in ma, b != u, b != a
            # We can just iterate b in ma
            mb = ma
            # Exclude u and a
            mb &= ~(1 << u)
            mb &= ~(1 << ai)
            
            while mb:
                lb2 = mb & -mb
                mb ^= lb2
                bi = lb2.bit_length() - 1
                
                mb2 = masks[bi]
                # c in mb2, c != b, c != a
                mc = mb2
                mc &= ~(1 << bi)
                mc &= ~(1 << ai)
                
                while mc:
                    lb3 = mc & -mc
                    mc ^= lb3
                    ci = lb3.bit_length() - 1
                    
                    mc2 = masks[ci]
                    # d in mc2 AND d in mv
                    # d != c, d != b, d != a, d != u, d != v
                    md = mc2 & mv
                    # Exclude c, b, a, u, v
                    md &= ~(1 << ci)
                    md &= ~(1 << bi)
                    md &= ~(1 << ai)
                    md &= ~(1 << u)
                    md &= ~(1 << v)
                    
                    if md:
                        return False # C6 found, cannot add
                    
        return True

    def run_greedy(seed):
        random.seed(seed)
        masks = [0] * n
        edge_count = 0
        
        # Start with empty graph or random sparse?
        # Empty is safer for correctness, but slower to reach high density.
        # Let's start with a random set of ~50 edges that are C6-free?
        # Or just start empty and add.
        
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if masks[u] & (1 << v):
                continue
            if can_add(u, v, masks):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                edge_count += 1
                
        return edge_count, masks

    # Try multiple seeds
    best_count = 0
    best_final_masks = [0] * n
    
    for i in range(50): # 50 runs
        count, masks = run_greedy(i)
        if count > best_count:
            best_count = count
            best_final_masks = masks
            # Early stop if we hit the theoretical max or known best?
            # Known best is 105. If we hit 105, we can try to improve or stop.
            # But let's just keep going to see if we can beat 105.
            # 50 runs * (2000 edges * check time)
            # Check time ~ 1000 ops. 2M ops per run. 100M ops total.
            # Python might be too slow for 100M ops in 120s?
            # 100M simple ops in Python is ~10-20s? Maybe.
            # Let's reduce runs to 20 or optimize.
            
    # Convert best_final_masks to edge list
    edges = []
    for u in range(n):
        m = best_final_masks[u]
        while m:
            lb = m & -m
            m ^= lb
            v = lb.bit_length() - 1
            if u < v:
                edges.append([u, v])
                
    return edges
