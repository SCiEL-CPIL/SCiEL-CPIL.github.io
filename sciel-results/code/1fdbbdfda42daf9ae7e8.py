import itertools
import time
import random
import sys

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = None
    best_count = -1
    
    # Precompute all triples for violation checking
    triples = list(itertools.combinations(range(n), 3))
    
    def is_valid_edge(adj_mask, u, v):
        """
        Check if adding edge (u, v) to graph with adjacency masks adj_mask
        creates a K3,3.
        """
        # We need to check if there exist b1, b2 in N(u) and a1, a2 in N(v)
        # such that a1, a2 are connected to b1, b2.
        # Note: u and v are not connected yet in this check context (we are considering adding them).
        # But adj_mask currently represents the graph WITHOUT (u,v).
        
        nu = adj_mask[u]
        nv = adj_mask[v]
        
        # Iterate over pairs of neighbors of u
        # Get list of neighbors of u
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        
        # Optimization: If degree is low, it's fast.
        # For each pair b1, b2 in nu_list:
        for i in range(len(nu_list)):
            b1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                b2 = nu_list[j]
                # Common neighbors of b1 and b2
                common = adj_mask[b1] & adj_mask[b2]
                # We need a1, a2 in N(v) AND in common
                # So we need at least 2 bits set in (nv & common)
                candidates = nv & common
                # Count bits
                # If count >= 2, invalid
                if candidates & (candidates - 1): # If at least 2 bits are set
                    return False
        return True

    def count_violations(adj_mask):
        """
        Count number of K3,3 subgraphs.
        """
        count = 0
        # For each triple B, find A_B
        for B in triples:
            b1, b2, b3 = B
            # A_B = {u | u connected to b1, b2, b3}
            # Using bitmasks
            # u is in A_B if (adj_mask[u] & (1<<b1) & (1<<b2) & (1<<b3)) != 0
            # Wait, this checks if u is connected to ALL of b1,b2,b3.
            # We need to find all such u.
            A_B_mask = 0
            for u in range(n):
                if (adj_mask[u] >> b1) & 1 and (adj_mask[u] >> b2) & 1 and (adj_mask[u] >> b3) & 1:
                    A_B_mask |= (1 << u)
            
            # k = number of bits in A_B_mask
            k = bin(A_B_mask).count('1')
            if k >= 3:
                # Number of K3,3s with this B is C(k, 3)
                count += k * (k-1) * (k-2) // 6
        return count

    # Main search loop
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Initialize complete graph
        adj_mask = [0]*n
        for u in range(n):
            mask = 0
            for v in range(n):
                if u != v:
                    mask |= (1 << v)
            adj_mask[u] = mask
            
        # Calculate initial edge counts for greedy removal
        # We need edge_cnt[e] = number of K3,3 containing e
        # This is expensive to compute from scratch every time.
        # Let's compute it once per attempt.
        
        def compute_edge_counts():
            edge_cnt = {}
            for u in range(n):
                for v in range(u+1, n):
                    edge_cnt[(u,v)] = 0
            
            # For each triple B, find A_B
            for B in triples:
                b1, b2, b3 = B
                A_B = []
                for u in range(n):
                    if (adj_mask[u] >> b1) & 1 and (adj_mask[u] >> b2) & 1 and (adj_mask[u] >> b3) & 1:
                        A_B.append(u)
                k = len(A_B)
                if k >= 3:
                    # Each pair of vertices in A_B forms a K3,3 with B?
                    # No. A K3,3 is defined by 3 vertices in A and 3 in B.
                    # If A_B has size k, the number of ways to choose 3 vertices from A_B is C(k,3).
                    # Each such choice forms a K3,3 with B.
                    # The edges involved are all (a, b) for a in A_subset, b in B.
                    # For a specific edge (u, b) where u in A_B and b in B:
                    # It is in a K3,3 if we choose 2 other vertices from A_B \ {u}.
                    # So count for (u,b) increases by C(k-1, 2).
                    add_val = (k-1)*(k-2)//2
                    for u in A_B:
                        for b in B:
                            e = (min(u,b), max(u,b))
                            edge_cnt[e] += add_val
            return edge_cnt

        edge_cnt = compute_edge_counts()
        
        # Greedy Removal
        while time.time() < deadline:
            if not edge_cnt:
                break
            # Find max count edge
            max_cnt = -1
            candidates = []
            for e, c in edge_cnt.items():
                if c > max_cnt:
                    max_cnt = c
                    candidates = [e]
                elif c == max_cnt:
                    candidates.append(e)
            
            if max_cnt <= 0:
                break
                
            # Random tie breaking
            e = random.choice(candidates)
            u, v = e
            # Remove edge
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            del edge_cnt[e]
            
            # Update counts
            # Instead of full recompute, we can update incrementally?
            # Full recompute is O(n^4) ~ 160k ops.
            # If we do 100 removals, 16M ops. Fast.
            edge_cnt = compute_edge_counts()
            
        # End of greedy removal. We have a valid graph.
        # Now try to add back edges (Repair Phase)
        # Get list of all possible edges not in adj_mask
        # Actually, we can just iterate all edges and try to add them if valid.
        
        # To make repair effective, we should try edges that were removed recently?
        # Or just random order.
        
        # Let's create a list of missing edges
        missing_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj_mask[u] >> v) & 1):
                    missing_edges.append((u,v))
                    
        random.shuffle(missing_edges)
        
        improved = True
        while improved:
            improved = False
            for e in missing_edges:
                u, v = e
                if (adj_mask[u] >> v) & 1:
                    continue # Already added
                if is_valid_edge(adj_mask, u, v):
                    # Add edge
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    missing_edges.remove(e)
                    improved = True
                    # Optional: Stop repair early if time is tight?
                    # No, repair is fast.
                    
        # Count edges
        edge_count = sum(bin(m).count('1') for m in adj_mask) // 2
        
        if edge_count > best_count:
            best_count = edge_count
            # Convert to list of edges
            best_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_mask[u] >> v) & 1:
                        best_edges.append([u, v])
                        
    if best_edges is None:
        return []
        
    return best_edges
