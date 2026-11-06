import time
import random
import numpy as np

def count_k33_fast(adj_bitmasks):
    """
    Count the number of K3,3 subgraphs.
    A K3,3 exists if there are 3 vertices {u, v, w} such that they have at least 3 common neighbors.
    Actually, the condition "no three vertices have three common neighbours" is equivalent to being K3,3-free?
    Let's verify: A K3,3 has bipartition {a,b,c} and {x,y,z}. 
    The neighbors of a,b,c include x,y,z. So |N(a) \cap N(b) \cap N(c)| >= 3.
    Conversely, if |N(a) \cap N(b) \cap N(c)| >= 3, let {x,y,z} be three such common neighbors.
    Then a,b,c are connected to x,y,z. This forms a K3,3 subgraph.
    So yes, checking triples for >= 3 common neighbors is sufficient and necessary.
    """
    n = len(adj_bitmasks)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            common_ij = adj_bitmasks[i] & adj_bitmasks[j]
            # Optimization: if common_ij has < 3 bits, skip k loop? No, we need to check intersections with k.
            # But if common_ij has < 3 bits, then common_ij & adj_bitmasks[k] will have < 3 bits.
            if bin(common_ij).count('1') < 3:
                continue
            for k in range(j + 1, n):
                common_ijk = common_ij & adj_bitmasks[k]
                if bin(common_ijk).count('1') >= 3:
                    count += 1
    return count

def has_k33_fast(adj_bitmasks):
    """
    Check if graph is K3,3-free.
    Returns True if it IS K3,3-free (valid), False otherwise.
    """
    n = len(adj_bitmasks)
    for i in range(n):
        for j in range(i + 1, n):
            common_ij = adj_bitmasks[i] & adj_bitmasks[j]
            if bin(common_ij).count('1') < 3:
                continue
            for k in range(j + 1, n):
                common_ijk = common_ij & adj_bitmasks[k]
                if bin(common_ijk).count('1') >= 3:
                    return False
    return True

def construct():
    n = 20
    deadline = time.time() + 110.0  # Safety margin
    
    best_edges = 0
    best_adj_bitmasks = None
    best_adj_matrix = None
    
    # Helper to convert bitmask list to edge list
    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            mask = bitmasks[i]
            # Iterate over set bits
            # Standard trick
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    # Helper to convert edge list to bitmasks
    def edges_to_bitmasks(edges):
        bitmasks = [0] * n
        for u, v in edges:
            bitmasks[u] |= (1 << v)
            bitmasks[v] |= (1 << u)
        return bitmasks

    # Local Search / Simulated Annealing
    def local_search(initial_bitmasks, temp):
        bitmasks = initial_bitmasks[:]
        current_edges = sum(bin(b).count('1') for b in bitmasks) // 2
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]
        
        # We want to maximize edges while staying K3,3-free.
        # Move types:
        # 1. Add a non-edge (if valid)
        # 2. Remove an edge (if it helps escape a trap? Usually we only add, but sometimes removing allows adding 2 more)
        # 3. Swap: Remove (u,v), Add (u,w), Add (v,x) ? Complex.
        
        # Let's stick to: 
        # - Try adding a random non-edge. If valid, accept.
        # - If stuck, try removing a random edge. Then try adding edges again.
        
        stuck_count = 0
        while time.time() < deadline:
            moved = False
            # Try to add an edge
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (bitmasks[i] & (1 << j)):
                        non_edges.append((i, j))
            
            if not non_edges:
                break # Complete graph (unlikely for K3,3-free)
            
            random.shuffle(non_edges)
            
            added = False
            for u, v in non_edges:
                # Try add
                new_bitmasks = bitmasks[:]
                new_bitmasks[u] |= (1 << v)
                new_bitmasks[v] |= (1 << u)
                
                if has_k33_fast(new_bitmasks):
                    bitmasks = new_bitmasks
                    current_edges += 1
                    moved = True
                    added = True
                    if current_edges > best_local_edges:
                        best_local_edges = current_edges
                        best_local_bitmasks = bitmasks[:]
                    break
            
            if not added:
                # Try to remove an edge to escape local optimum
                if current_edges == 0:
                    break
                all_edges = []
                for i in range(n):
                    mask = bitmasks[i]
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        if i < j:
                            all_edges.append((i, j))
                        m -= lsb
                
                if all_edges:
                    u, v = random.choice(all_edges)
                    # Remove
                    bitmasks[u] &= ~(1 << v)
                    bitmasks[v] &= ~(1 << u)
                    current_edges -= 1
                    moved = True
            
            if not moved:
                stuck_count += 1
                if stuck_count > 50:
                    break
            else:
                stuck_count = 0
                
        return best_local_bitmasks, best_local_edges

    # Run multiple restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with a random sparse graph
        # Start with a random graph with probability p
        # p should be low enough to likely be K3,3-free, or we check and repair
        # ex(20, K3,3) is roughly n^(3/2) ? No, K3,3 is bipartite.
        # General bound for K_{k,k} is O(n^{2-1/k}). For k=3, O(n^{5/3}).
        # 20^(5/3) ~ 85. So max edges might be around 50-80.
        # Random graph G(20, 0.3) has 60 edges. Likely contains K3,3.
        # Let's start with p=0.2 (40 edges) and repair? Or just start empty and add?
        
        # Strategy: Start with a random graph, if invalid, remove random edges until valid.
        # Then run local search.
        
        init_bitmasks = [0] * n
        p = 0.2 + 0.05 * r  # Slightly denser starts later?
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    init_bitmasks[i] |= (1 << j)
                    init_bitmasks[j] |= (1 << i)
        
        # Repair: Remove random edges until valid
        valid = has_k33_fast(init_bitmasks)
        attempts = 0
        while not valid and attempts < 1000:
            # Find a K3,3 and remove an edge from it
            # Simple repair: just remove a random edge
            all_edges = []
            for i in range(n):
                mask = init_bitmasks[i]
                m = mask
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if i < j:
                        all_edges.append((i, j))
                    m -= lsb
            
            if not all_edges:
                break
            
            u, v = random.choice(all_edges)
            init_bitmasks[u] &= ~(1 << v)
            init_bitmasks[v] &= ~(1 << u)
            valid = has_k33_fast(init_bitmasks)
            attempts += 1
        
        if not valid:
            continue
            
        # Run local search
        local_best_bitmasks, local_best_edges = local_search(init_bitmasks, 1.0)
        
        if local_best_edges > best_edges:
            best_edges = local_best_edges
            best_adj_bitmasks = local_best_bitmasks
            # Update global best matrix
            best_adj_matrix = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj_bitmasks[i] & (1 << j):
                        best_adj_matrix[i, j] = True
                        best_adj_matrix[j, i] = True

    # If no valid graph found (should not happen), return empty
    if best_adj_bitmasks is None:
        return []
        
    return bitmasks_to_edges(best_adj_bitmasks)

# Quick self-test
if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
    # Verify
    n = 20
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    
    # Verify K3,3 free
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                common = adj[i] & adj[j] & adj[k]
                if np.sum(common) >= 3:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid K3,3-free: {valid}")
