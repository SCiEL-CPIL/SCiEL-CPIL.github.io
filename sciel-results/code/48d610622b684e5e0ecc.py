import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 20
    # Precompute all triples of vertices
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    def is_valid(adj):
        """
        adj: list of integers representing adjacency bitmasks.
        Returns True if graph is K3,3-free.
        Optimized: early termination, no dict overhead.
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def get_violating_triple(adj):
        """
        Returns a triple (i, j, k) such that common neighbors >= 3, or None.
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k, common)
        return None

    def repair(adj):
        """
        Removes edges until the graph is valid.
        Uses edge impact scoring: for each violating triple, count how many
        other violations each candidate edge participates in, remove the most impactful.
        """
        removed = 0
        while True:
            v = get_violating_triple(adj)
            if v is None:
                break
            i, j, k, common = v
            common_nodes = []
            for m in range(n):
                if common & (1 << m):
                    common_nodes.append(m)
            
            # Score each candidate edge (x, m) by how many violations it's in
            # For each (x, m), count triples (a,b,c) where x is in {a,b,c} and m is common neighbor
            best_edge = None
            best_score = -1
            
            for x in [i, j, k]:
                for m in common_nodes:
                    # Count how many other triples have x and m in a violating config
                    score = 0
                    for a, b, c in triples:
                        if x not in (a, b, c):
                            continue
                        # Check if m is common neighbor of a,b,c
                        common_abc = adj[a] & adj[b] & adj[c]
                        if common_abc & (1 << m):
                            if common_abc.bit_count() >= 3:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_edge = (x, m)
            
            if best_edge is None:
                # Fallback: remove highest degree edge
                x = max([i, j, k], key=lambda v: adj[v].bit_count())
                m = max(common_nodes, key=lambda v: adj[v].bit_count())
                best_edge = (x, m)
            
            x, m = best_edge
            adj[x] &= ~(1 << m)
            adj[m] &= ~(1 << x)
            removed += 1
            
        return removed

    def spectral_init(rng):
        """
        Create a dense graph using spectral method.
        """
        A = rng.random((n, n))
        A = (A + A.T) / 2
        
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
            idx = np.argmax(eigvals)
            v = eigvecs[:, idx]
            
            if v.sum() < 0:
                v = -v
            
            outer = np.outer(v, v)
            mean_val = np.mean(outer)
            
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if outer[i, j] > mean_val:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            return adj
        except:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.7:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            return adj

    def random_dense_init(rng, density=0.65):
        """
        Create a random dense graph.
        """
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(adj, current_edges, deadline, rng):
        """
        Local search with edge-add, edge-swap, and systematic polish.
        Returns (adj, current_edges).
        """
        no_improve_count = 0
        max_iterations = 20000
        iter_count = 0
        
        while iter_count < max_iterations:
            if time.time() > deadline - 2.0:
                break
            
            iter_count += 1
            
            # Try to add a random non-edge
            u = rng.integers(n)
            v = rng.integers(n)
            if u == v:
                continue
            
            if adj[u] & (1 << v):
                continue
            
            # Try adding it
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid(adj):
                current_edges += 1
                no_improve_count = 0
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                no_improve_count += 1
                
                # If stuck, try an edge swap: remove one edge, add another
                if no_improve_count > 30:
                    # Pick a random existing edge to remove
                    while True:
                        u_rm = rng.integers(n)
                        if adj[u_rm] == 0:
                            continue
                        neighbors = [w for w in range(n) if adj[u_rm] & (1 << w)]
                        if neighbors:
                            v_rm = neighbors[rng.integers(len(neighbors))]
                            adj[u_rm] &= ~(1 << v_rm)
                            adj[v_rm] &= ~(1 << u_rm)
                            current_edges -= 1
                            break
                    no_improve_count = 0
        
        # Systematic polish: try all non-edges
        if time.time() < deadline - 3.0:
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline - 1.0:
                        break
                    if adj[u] & (1 << v):
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if is_valid(adj):
                        current_edges += 1
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
        
        return adj, current_edges

    best_edges = 0
    best_adj = None
    
    # Try multiple random starts
    for seed in range(150):
        if time.time() > deadline - 5.0:
            break
            
        rng = np.random.default_rng(seed)
        
        # Mix of initialization strategies
        if seed % 3 == 0:
            adj = spectral_init(rng)
        else:
            adj = random_dense_init(rng, density=0.6 + 0.1 * (seed % 5))
        
        # Repair to valid state
        repair(adj)
        
        current_edges = count_edges(adj)
        
        # Local search
        adj, current_edges = local_search(adj, current_edges, deadline, rng)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [x for x in adj]

    # Final global polish on the best solution found
    if best_adj is not None and time.time() < deadline - 2.0:
        # Do multiple systematic passes
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline - 0.5:
                        break
                    if best_adj[u] & (1 << v):
                        continue
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    if is_valid(best_adj):
                        best_edges += 1
                    else:
                        best_adj[u] &= ~(1 << v)
                        best_adj[v] &= ~(1 << u)
    
    # Construct edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
