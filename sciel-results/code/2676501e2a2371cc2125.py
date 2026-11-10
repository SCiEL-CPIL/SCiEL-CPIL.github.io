import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def is_regular_subset(mask, adj, n):
        """
        Check if the induced subgraph on vertices in 'mask' is regular.
        adj is a list of bitmasks.
        """
        # Get vertices in the subset
        verts = []
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        
        if len(verts) < 6:
            return False
            
        # Calculate degrees within the subset
        degs = []
        for v in verts:
            # adj[v] & mask gives neighbors of v within the subset
            degs.append((adj[v] & mask).bit_count())
        
        # Check if all degrees are equal
        d0 = degs[0]
        for d in degs[1:]:
            if d != d0:
                return False
        return True

    def has_violation(n, adj):
        """
        Check if there is any subset of size >= 6 that is regular.
        """
        # We check subsets of size k from 6 to n
        # Using combinations of indices is slower than bitmask iteration for large n,
        # but for n=21, itertools.combinations is fine and easier to read.
        # However, generating all combinations of size 6..21 is expensive.
        # Let's iterate over all masks from 1 to 2**n - 1? No, too many.
        # Let's use itertools.combinations for each k.
        
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if is_regular_subset(mask, adj, n):
                    return True
        return False

    def try_find_graph(n, time_limit):
        """
        Randomly search for a valid graph on n vertices.
        """
        # We can bias p. p=0.5 is standard.
        for _ in range(1000):
            if time.time() > time_limit:
                return None
            
            adj = [0] * n
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
            
            if not has_violation(n, adj):
                return {'n': n, 'edges': edges}
        
        return None

    # Try n=21 first
    # If we find it, it's a huge win.
    # If not, fall back to 20.
    
    # To maximize chance, we can run searches for 21 and 20 in parallel? 
    # No, sequential is easier. 
    # But 20 is "known", so it should be found instantly. 
    # 21 might be harder or impossible.
    
    # Let's allocate time.
    # Try 21 for 70 seconds.
    # Try 20 for 70 seconds.
    
    best_graph = None
    
    # Attempt n=21
    t_limit_21 = start_time + 70
    g21 = try_find_graph(21, t_limit_21)
    if g21:
        return g21
        
    # Attempt n=20
    t_limit_20 = start_time + 145
    g20 = try_find_graph(20, t_limit_20)
    if g20:
        return g20
        
    # If even 20 fails (unlikely), return a known small valid graph or best effort
    # A graph with n=5 is always valid.
    return {'n': 5, 'edges': []}
