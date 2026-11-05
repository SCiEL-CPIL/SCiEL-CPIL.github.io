import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Attempts to find a valid graph with n=33, 34, or 35 using randomized maximal C5-free construction.
    Falls back to the known valid n=32 construction (8 disjoint K4s).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for safety/overhead

    # Fallback: 8 disjoint K4s, n=32
    n_fallback = 32
    edges_fallback = []
    for block_idx in range(8):
        start = block_idx * 4
        verts = [start, start + 1, start + 2, start + 3]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                edges_fallback.append([verts[i], verts[j]])

    def check_alpha_less_than_9(adj, n):
        """
        Returns True if the independence number is less than 9.
        Uses branch and bound with bitsets.
        """
        # Precompute neighbor bitsets
        # adj is a list of integers, bit j set if i~j
        
        def find_iset(candidates, k):
            # Check if there is an independent set of size k in candidates
            if k == 0:
                return True
            if candidates == 0:
                return False
            if candidates.bit_count() < k:
                return False
            
            # Heuristic: pick vertex with minimum degree in the induced subgraph?
            # Or just the first one. For speed, first one is fine.
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            # Remove v and its neighbors
            new_cands = candidates & ~adj[v] & ~(1 << v)
            if find_iset(new_cands, k - 1):
                return True
            
            # Branch 2: Exclude v
            if find_iset(candidates & ~(1 << v), k):
                return True
            
            return False

        # Try to find an independent set of size 9
        # If found, alpha >= 9, so return False
        # If not found, alpha < 9, so return True
        if find_iset((1 << n) - 1, 9):
            return False
        return True

    def has_c5_after_adding_edge(adj, u, v, n):
        """
        Checks if adding edge (u, v) creates a C5.
        Assumes (u, v) is not currently an edge.
        """
        nu = adj[u]
        nv = adj[v]
        
        # We need a path u-a-b-c-v
        # a in N(u), c in N(v)
        # b in N(a) and b in N(c)
        # a, b, c distinct and not u, v
        
        # Iterate over a in N(u)
        a_bits = nu
        while a_bits:
            lsb_a = a_bits & -a_bits
            a = lsb_a.bit_length() - 1
            a_bits ^= lsb_a
            
            if a == v:
                continue
                
            # Iterate over c in N(v)
            c_bits = nv
            while c_bits:
                lsb_c = c_bits & -c_bits
                c = lsb_c.bit_length() - 1
                c_bits ^= lsb_c
                
                if c == u or c == a:
                    continue
                    
                # Find b in N(a) intersect N(c)
                # b must not be u, v, a, c
                common = adj[a] & adj[c]
                mask = (1 << u) | (1 << v) | (1 << a) | (1 << c)
                
                if common & ~mask:
                    return True
                    
        return False

    def try_find_n(n_target, max_restarts):
        """
        Tries to find a valid graph with n_target vertices.
        """
        rng = random.Random(42 + n_target)
        
        # Pre-generate all possible edges
        all_edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Shuffle edges
            edges_order = all_edges[:]
            rng.shuffle(edges_order)
            
            # Initialize empty graph
            adj = [0] * n_target
            
            for u, v in edges_order:
                if time.time() > deadline:
                    return None
                
                # Check if adding (u, v) creates C5
                if has_c5_after_adding_edge(adj, u, v, n_target):
                    continue
                
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Check alpha
            if check_alpha_less_than_9(adj, n_target):
                # Extract edges
                edges = []
                for u in range(n_target):
                    v_bits = adj[u]
                    while v_bits:
                        lsb = v_bits & -v_bits
                        v = lsb.bit_length() - 1
                        v_bits ^= lsb
                        if u < v:
                            edges.append([u, v])
                return {'n': n_target, 'edges': edges}
                
        return None

    # Try n=33 first, then 34, 35
    # We allocate time for each
    # If n=33 succeeds, we return it.
    
    # Let's try a few restarts for each n
    # Since n=33 is the most likely to succeed and improve over 32, we give it more chance.
    
    # Strategy: 
    # 1. Try n=33 with, say, 20 restarts.
    # 2. If fails, try n=34 with 10 restarts.
    # 3. If fails, try n=35 with 5 restarts.
    # 4. Fallback to 32.
    
    # Actually, if n=33 fails, n=34 is less likely to succeed with the same method?
    # Larger n means more edges, harder to keep C5-free and alpha small.
    # So we should focus on n=33.
    
    result = try_find_n(33, 50)
    if result:
        return result
        
    # If n=33 fails, maybe try n=34?
    # But time is limited.
    # Let's check time.
    if time.time() < deadline - 10:
        result = try_find_n(34, 20)
        if result:
            return result
            
    if time.time() < deadline - 5:
        result = try_find_n(35, 10)
        if result:
            return result

    return {'n': n_fallback, 'edges': edges_fallback}

if __name__ == "__main__":
    r = construct()
    print(r['n'])
