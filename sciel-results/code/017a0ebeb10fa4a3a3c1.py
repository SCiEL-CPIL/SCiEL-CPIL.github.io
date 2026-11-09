import time
import random

def construct():
    n = 18
    start = time.time()
    deadline = start + 145  # 5s safety margin

    def is_k4_free(adj):
        """Check if graph is K4-free using bitmasks."""
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] & (1 << b)): continue
                for c in range(b+1, n):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                    common = adj[a] & adj[b] & adj[c]
                    if common:
                        return False
        return True

    def compute_L(adj):
        """Compute largest triangle-free induced subgraph size using DP over masks."""
        size = 1 << n
        is_tf = [False] * size
        is_tf[0] = True
        max_size = 0
        
        # Precompute bit_count for masks? No, just use bit_count()
        for m in range(1, size):
            # Get lowest set bit
            lowbit = m & -m
            v = lowbit.bit_length() - 1
            m_rest = m ^ lowbit
            
            if not is_tf[m_rest]:
                continue
                
            # Check if adding v creates a triangle
            # Triangle must involve v and two neighbors u, w in m_rest
            cand = adj[v] & m_rest
            if cand == 0:
                is_tf[m] = True
                sz = m.bit_count()
                if sz > max_size:
                    max_size = sz
            else:
                # Check if G[cand] has any edge
                has_edge = False
                c = cand
                while c:
                    u_bit = c & -c
                    u = u_bit.bit_length() - 1
                    c ^= u_bit
                    # If u has any neighbor in cand, there is an edge in G[cand]
                    # adj[u] contains neighbors of u.
                    # We need to check if (adj[u] & cand) is non-zero.
                    # Note: u is in cand, but u is not in adj[u] (simple graph).
                    # So if adj[u] & cand != 0, there is a neighbor w in cand, w != u.
                    # Thus edge (u,w) exists in G[cand].
                    if adj[u] & cand:
                        has_edge = True
                        break
                
                if not has_edge:
                    is_tf[m] = True
                    sz = m.bit_count()
                    if sz > max_size:
                        max_size = sz
                    
        return max_size

    def make_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def random_k4_free():
        """Generate a random K4-free graph."""
        adj = [0] * n
        # Random edges
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair to K4-free
        # Simple repair: if K4 found, remove a random edge from it
        changed = True
        iters = 0
        while changed and iters < 1000:
            changed = False
            iters += 1
            for a in range(n):
                if changed: break
                for b in range(a+1, n):
                    if changed: break
                    if not (adj[a] & (1 << b)): continue
                    for c in range(b+1, n):
                        if changed: break
                        if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                        common = adj[a] & adj[b] & adj[c]
                        if common:
                            # Found K4 (a,b,c,d) where d in common
                            # Remove edge (a, d)
                            d = (common & -common).bit_length() - 1
                            adj[a] &= ~(1 << d)
                            adj[d] &= ~(1 << a)
                            changed = True
        return adj

    # Start with Turan(18,3)
    adj_turan = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if i % 3 != j % 3:
                adj_turan[i] |= (1 << j)
                adj_turan[j] |= (1 << i)
    
    best_adj = adj_turan[:]
    best_L = compute_L(best_adj)
    
    # Local Search
    # We will perform random restarts and local search
    # Since compute_L is now faster, we can afford more steps
    
    # Try some random graphs
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        adj = random_k4_free()
        if not is_k4_free(adj):
            continue
        L = compute_L(adj)
        if L < best_L:
            best_L = L
            best_adj = adj[:]
            
    # Local search on best_adj
    current_adj = best_adj[:]
    current_L = best_L
    
    # Precompute list of edges for flipping
    # We iterate over all pairs (i,j)
    for step in range(1000):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Shuffle order of edges to avoid bias
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for i, j in pairs:
            if time.time() > deadline - 5:
                break
            
            # Try flipping edge (i,j)
            old_state = bool(current_adj[i] & (1 << j))
            
            # Apply flip
            if old_state:
                current_adj[i] &= ~(1 << j)
                current_adj[j] &= ~(1 << i)
            else:
                current_adj[i] |= (1 << j)
                current_adj[j] |= (1 << i)
            
            if is_k4_free(current_adj):
                L = compute_L(current_adj)
                if L < current_L:
                    current_L = L
                    best_L = L
                    best_adj = [a | 0 for a in current_adj]
                    improved = True
                else:
                    # Revert
                    if old_state:
                        current_adj[i] |= (1 << j)
                        current_adj[j] |= (1 << i)
                    else:
                        current_adj[i] &= ~(1 << j)
                        current_adj[j] &= ~(1 << i)
            else:
                # Revert
                if old_state:
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
                else:
                    current_adj[i] &= ~(1 << j)
                    current_adj[j] &= ~(1 << i)
            
            if improved:
                # Restart local search from new best
                current_adj = best_adj[:]
                current_L = best_L
                break
        
        if not improved:
            # Try a random restart from a new random graph
            adj = random_k4_free()
            if is_k4_free(adj):
                L = compute_L(adj)
                if L < best_L:
                    best_L = L
                    best_adj = adj[:]
                    current_adj = best_adj[:]
                    current_L = best_L
                else:
                    # Keep searching from current best or break if stuck?
                    # Let's try a few random perturbations
                    pass

    return make_edge_list(best_adj)
