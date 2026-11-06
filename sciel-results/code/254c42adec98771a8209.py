import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    N = 20
    TOTAL_EDGES = N * (N - 1) // 2  # 190
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Bitmask utilities for fast operations
    # Python 3.10+ has int.bit_count(), which is much faster than bin().count('1')
    if hasattr(int, 'bit_count'):
        def count_bits(x):
            return x.bit_count()
    else:
        def count_bits(x):
            return bin(x).count('1')

    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added without violating K2,3 constraint."""
        if adj[u] & (1 << v):
            return True  # Already present
        
        # 1. Check pair (u, v): common neighbours must be <= 2
        if count_bits(adj[u] & adj[v]) >= 3:
            return False
        
        # 2. For each w in N(u) \ {v}, check |N(v) ∩ N(w)| <= 1
        mask_u = adj[u] & ~(1 << v)
        w = mask_u
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if count_bits(adj[v] & adj[idx]) >= 2:
                return False
        
        # 3. For each w in N(v) \ {u}, check |N(u) ∩ N(w)| <= 1
        mask_v = adj[v] & ~(1 << u)
        w = mask_v
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if count_bits(adj[u] & adj[idx]) >= 2:
                return False
        
        return True
    
    def add_edge(adj, u, v):
        """Add edge (u,v) to bitmask representation."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        """Remove edge (u,v) from bitmask representation."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def count_edges(adj):
        """Count edges from bitmask representation."""
        return sum(count_bits(adj[u]) for u in range(N)) // 2
    
    def bitmask_to_edge_list(adj):
        """Convert bitmask representation to edge list."""
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    def greedy_build(seed):
        """Greedy construction using bitmask representation."""
        rng = random.Random(seed)
        adj = [0] * N
        
        edge_indices = list(range(TOTAL_EDGES))
        rng.shuffle(edge_indices)
        
        for idx in edge_indices:
            u, v = all_edges[idx]
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        
        return adj
    
    def local_search(adj, seed, time_limit):
        """Enhanced local search with multiple move types."""
        rng = random.Random(seed)
        adj = [b for b in adj]  # Copy
        
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = [b for b in adj]
        
        start_time_ls = time.time()
        
        while time.time() - start_time_ls < time_limit:
            if time.time() > deadline:
                break
            improved = False
            
            # Strategy 1: Try to add random missing edges (single add)
            missing = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
            if missing:
                rng.shuffle(missing)
                for u, v in missing[:30]:  # More attempts per iteration
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        improved = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = [b for b in adj]
                        break
            
            # Strategy 2: 1-for-2 swap (remove 1 edge, try to add 2+)
            if not improved:
                present = [(u, v) for u, v in all_edges if (adj[u] & (1 << v))]
                if present:
                    rng.shuffle(present)
                    for u_rm, v_rm in present[:8]:
                        if time.time() > deadline:
                            break
                        
                        adj_backup = [b for b in adj]
                        
                        remove_edge(adj, u_rm, v_rm)
                        current_edges -= 1
                        
                        added = 0
                        missing_now = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
                        rng.shuffle(missing_now)
                        for u_add, v_add in missing_now[:40]:
                            if can_add_edge(adj, u_add, v_add):
                                add_edge(adj, u_add, v_add)
                                current_edges += 1
                                added += 1
                                if current_edges > best_edges:
                                    best_edges = current_edges
                                    best_adj = [b for b in adj]
                                if added >= 2:
                                    break
                        
                        if added < 2:
                            adj = adj_backup
                            current_edges += 1
                        else:
                            improved = True
                            break
            
            # Strategy 3: 2-for-3 swap (remove 2 edges, try to add 3+)
            if not improved:
                present = [(u, v) for u, v in all_edges if (adj[u] & (1 << v))]
                if len(present) >= 2:
                    rng.shuffle(present)
                    num_swaps = min(5, len(present) - 1)
                    for i in range(num_swaps):
                        if time.time() > deadline:
                            break
                        u1, v1 = present[i]
                        u2, v2 = present[(i + 1) % len(present)]
                        
                        if (u1, v1) == (u2, v2): continue
                        
                        adj_backup = [b for b in adj]
                        
                        remove_edge(adj, u1, v1)
                        remove_edge(adj, u2, v2)
                        current_edges -= 2
                        
                        added = 0
                        missing_now = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
                        rng.shuffle(missing_now)
                        for u_add, v_add in missing_now[:50]:
                            if can_add_edge(adj, u_add, v_add):
                                add_edge(adj, u_add, v_add)
                                current_edges += 1
                                added += 1
                                if current_edges > best_edges:
                                    best_edges = current_edges
                                    best_adj = [b for b in adj]
                                if added >= 3:
                                    break
                        
                        if added < 3:
                            adj = adj_backup
                            current_edges += 2
                        else:
                            improved = True
                            break
            
            # Strategy 4: 3-for-4 swap (remove 3 edges, try to add 4+)
            if not improved:
                present = [(u, v) for u, v in all_edges if (adj[u] & (1 << v))]
                if len(present) >= 3:
                    rng.shuffle(present)
                    num_swaps = min(3, len(present) - 2)
                    for i in range(num_swaps):
                        if time.time() > deadline:
                            break
                        u1, v1 = present[i]
                        u2, v2 = present[(i + 1) % len(present)]
                        u3, v3 = present[(i + 2) % len(present)]
                        
                        # Check for duplicates
                        edges_to_remove = [(u1, v1), (u2, v2), (u3, v3)]
                        if len(set(edges_to_remove)) < 3: continue
                        
                        adj_backup = [b for b in adj]
                        
                        remove_edge(adj, u1, v1)
                        remove_edge(adj, u2, v2)
                        remove_edge(adj, u3, v3)
                        current_edges -= 3
                        
                        added = 0
                        missing_now = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
                        rng.shuffle(missing_now)
                        for u_add, v_add in missing_now[:60]:
                            if can_add_edge(adj, u_add, v_add):
                                add_edge(adj, u_add, v_add)
                                current_edges += 1
                                added += 1
                                if current_edges > best_edges:
                                    best_edges = current_edges
                                    best_adj = [b for b in adj]
                                if added >= 4:
                                    break
                        
                        if added < 4:
                            adj = adj_backup
                            current_edges += 3
                        else:
                            improved = True
                            break
            
            # Strategy 5: 2-for-2 swap (remove 2 edges, add exactly 2 different ones)
            if not improved:
                present = [(u, v) for u, v in all_edges if (adj[u] & (1 << v))]
                missing = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
                if len(present) >= 2 and len(missing) >= 2:
                    rng.shuffle(present)
                    rng.shuffle(missing)
                    num_swaps = min(5, len(present) - 1)
                    for i in range(num_swaps):
                        if time.time() > deadline:
                            break
                        u1, v1 = present[i]
                        u2, v2 = present[(i + 1) % len(present)]
                        
                        if (u1, v1) == (u2, v2): continue
                        
                        adj_backup = [b for b in adj]
                        
                        remove_edge(adj, u1, v1)
                        remove_edge(adj, u2, v2)
                        current_edges -= 2
                        
                        added = 0
                        # Try to add the shuffled missing edges
                        for u_add, v_add in missing[:30]:
                            if can_add_edge(adj, u_add, v_add):
                                add_edge(adj, u_add, v_add)
                                current_edges += 1
                                added += 1
                                if current_edges > best_edges:
                                    best_edges = current_edges
                                    best_adj = [b for b in adj]
                                if added >= 2:
                                    break
                        
                        if added < 2:
                            adj = adj_backup
                            current_edges += 2
                        else:
                            improved = True
                            break
            
            if not improved:
                # Small random perturbation: remove 1-2 random edges and rebuild
                present = [(u, v) for u, v in all_edges if (adj[u] & (1 << v))]
                if present:
                    rng.shuffle(present)
                    num_remove = min(2, len(present))
                    for u_rm, v_rm in present[:num_remove]:
                        remove_edge(adj, u_rm, v_rm)
                        current_edges -= 1
                    
                    # Re-add greedily
                    missing_now = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
                    rng.shuffle(missing_now)
                    for u_add, v_add in missing_now:
                        if can_add_edge(adj, u_add, v_add):
                            add_edge(adj, u_add, v_add)
                            current_edges += 1
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = [b for b in adj]
        
        return best_adj, best_edges
    
    # Multiple restarts with different strategies
    best_adj = [0] * N
    best_count = 0
    
    # We will use a mix of random seeds.
    # Given the tight time limit, we prioritize fast greedy + short local search.
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Simple random greedy is fast and effective
        adj = greedy_build(restart)
        
        cur_count = count_edges(adj)
        
        # Apply local search with a small time budget per restart
        # Total time ~ 115s. 50 restarts. ~2s each.
        adj, cur_count = local_search(adj, restart, time_limit=1.5)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = [b for b in adj]
    
    # Final polish: try to add any remaining edges multiple times
    for _ in range(3):  # Multiple passes for thoroughness
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not (best_adj[u] & (1 << v)):
                if can_add_edge(best_adj, u, v):
                    add_edge(best_adj, u, v)
                    best_count += 1
    
    return bitmask_to_edge_list(best_adj)
