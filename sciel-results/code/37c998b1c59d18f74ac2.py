"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n = 40
    best_edges = []
    best_count = 0
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_c6_fast(adj_bits, u, v):
        """
        Check if adding edge (u,v) creates a C6 using bitset operations.
        A C6 exists if there's a path u-a-b-c-d-v of length 5.
        We check: exists a in N(u), b in N(a)\{u,v}, c in N(b)\{u,a,v}, d in N(c)∩N(v)\{u,a,b,c}
        """
        # For each a in N(u):
        a_mask = adj_bits[u] & ~(1 << v)
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # For each b in N(a) \ {u, v}:
            b_mask = adj_bits[a] & ~(1 << u) & ~(1 << v)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # For each c in N(b) \ {u, a, v}:
                c_mask = adj_bits[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # Check if N(c) ∩ N(v) contains any d not in {u, a, b, c}
                    d_candidates = adj_bits[c] & adj_bits[v]
                    forbidden = (1 << u) | (1 << a) | (1 << b) | (1 << c)
                    if d_candidates & ~forbidden:
                        return True
        return False
    
    def greedy_build(order, adj_bits, edge_set):
        """
        Greedily add edges from order to adj_bits if no C6 is formed.
        Returns the list of added edges.
        """
        added_edges = []
        for u, v in order:
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            # Only add if not already adjacent
            if (adj_bits[u] >> v) & 1:
                continue
            
            if not has_c6_fast(adj_bits, u, v):
                # Add edge
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                added_edges.append([u, v])
        return added_edges
    
    def local_search(adj_bits, edge_set, max_swaps=50):
        """
        Try to improve by removing one edge and adding up to two others.
        Returns the number of edges after local search.
        """
        current_count = len(edge_set) // 2  # Each edge stored twice
        
        # Get list of current edges
        current_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) in edge_set]
        
        # Try single edge additions first
        for u, v in all_pairs:
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            if (adj_bits[u] >> v) & 1:
                continue
            if not has_c6_fast(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                current_count += 1
        
        # Try 1-for-2 swaps: remove edge (a,b), add edges (c,d) and (e,f)
        # This is expensive, so limit attempts
        random.shuffle(current_edges)
        swaps_attempted = 0
        for a, b in current_edges[:max_swaps]:
            if time.time() > deadline - 5:
                break
            # Try removing (a,b)
            adj_bits[a] &= ~(1 << b)
            adj_bits[b] &= ~(1 << a)
            edge_set.discard((a, b))
            edge_set.discard((b, a))
            count_after_removal = current_count - 1
            
            # Try adding edges
            added = []
            for u, v in all_pairs:
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if (adj_bits[u] >> v) & 1:
                    continue
                if not has_c6_fast(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edge_set.add((u, v))
                    edge_set.add((v, u))
                    added.append((u, v))
                    if len(added) >= 2:
                        break
            
            if len(added) > 1:
                current_count += len(added) - 1
            else:
                # Undo the removal
                adj_bits[a] |= (1 << b)
                adj_bits[b] |= (1 << a)
                edge_set.add((a, b))
                edge_set.add((b, a))
                # Undo additions
                for u, v in added:
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    edge_set.discard((u, v))
                    edge_set.discard((v, u))
            
            swaps_attempted += 1
        
        return current_count

    random.seed(42)
    
    # Initial baseline: 8 disjoint K5s
    # Vertices 0-4, 5-9, ..., 35-39
    base_adj = [0] * n
    base_edges = []
    for k in range(8):
        start = k * 5
        for a in range(5):
            for b in range(a + 1, 5):
                u = start + a
                v = start + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
                base_edges.append([u, v])
    
    best_count = len(base_edges)
    best_edges = base_edges[:]
    best_adj = base_adj[:]
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random restart
        current_adj = [0] * n
        current_edge_set = set()
        
        # Strategy 1: Start with baseline and add random edges
        if attempts % 3 == 1:
            # Start with baseline
            for u, v in base_edges:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edge_set.add((u, v))
                current_edge_set.add((v, u))
            # Add random non-baseline edges
            non_baseline_pairs = [(i, j) for i, j in all_pairs 
                                  if not ((i, j) in current_edge_set or (j, i) in current_edge_set)]
            random.shuffle(non_baseline_pairs)
            current_order = non_baseline_pairs
        else:
            # Start empty with random order
            current_order = all_pairs[:]
            random.shuffle(current_order)
        
        # Greedy add
        current_edges = greedy_build(current_order, current_adj, current_edge_set)
        
        # Local search to improve
        if time.time() < deadline - 10:
            current_count = local_search(current_adj, current_edge_set, max_swaps=20)
        else:
            current_count = len(current_edge_set) // 2
        
        if current_count > best_count:
            best_count = current_count
            # Reconstruct edge list
            best_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in current_edge_set:
                        best_edges.append([u, v])
            best_adj = current_adj[:]
            
    return best_edges
# EVOLVE-BLOCK-END
