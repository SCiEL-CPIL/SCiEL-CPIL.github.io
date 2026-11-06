import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_regular_mask(mask, adj, n):
        """
        Check if the induced subgraph on vertices in 'mask' is regular.
        mask: integer bitmask
        adj: list of integer bitmasks
        n: total vertices
        """
        k = popcount(mask)
        if k < 6:
            return True
        
        first_deg = -1
        # Iterate through vertices in mask
        # Optimization: iterate only over set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            u = lsb.bit_length() - 1
            deg = popcount(adj[u] & mask)
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
            m ^= lsb
        return True

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def verify_full(adj, n):
        """
        Verify no induced subgraph of size >= 6 is regular.
        """
        # Check sizes 6 to n
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular_mask(mask, adj, n):
                    return False
        return True

    def solve_n(n, time_budget):
        """
        Solve for n vertices using local search with incremental updates.
        """
        end_time = start_time + time_budget
        if time.time() > end_time:
            return None, None
            
        # Precompute 6-subsets
        subsets6 = []
        for comb in itertools.combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets6.append(mask)
        
        num_subsets = len(subsets6)
        
        # Map edge (u,v) to list of indices in subsets6 that contain both u and v
        # This allows fast updates
        # edge_map[u][v] = list of indices
        edge_map = [[[] for _ in range(n)] for __ in range(n)]
        
        for i, mask in enumerate(subsets6):
            # Extract vertices in mask
            verts = []
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                verts.append(u)
                m ^= lsb
            
            # For each pair in verts, add i to edge_map
            for a in range(len(verts)):
                for b in range(a + 1, len(verts)):
                    u = verts[a]
                    v = verts[b]
                    edge_map[u][v].append(i)
                    edge_map[v][u].append(i)
        
        # Initial random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Initialize bad status for each 6-subset
        # bad[i] = 1 if subsets6[i] is regular, 0 otherwise
        bad = [0] * num_subsets
        total_bad = 0
        for i, mask in enumerate(subsets6):
            if is_regular_mask(mask, adj, n):
                bad[i] = 1
                total_bad += 1
        
        best_total_bad = total_bad
        best_adj = adj[:]
        
        # Local Search
        no_improve_count = 0
        max_no_improve = 500 # Restart if stuck
        
        while time.time() < end_time:
            if total_bad == 0:
                # Check full condition
                if verify_full(adj, n):
                    return n, get_edges_from_adj(adj, n)
                # If it fails full check, it means there is a regular subgraph of size > 6
                # but no size 6. This is rare but possible.
                # We can try to flip a random edge to break the larger one, 
                # but we don't know which one. 
                # For now, treat it as a "stuck" state or continue searching.
                # Actually, if we have no bad 6-sets, it's a very strong candidate.
                # We might just restart to find a cleaner one.
                
            if total_bad < best_total_bad:
                best_total_bad = total_bad
                best_adj = adj[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart with new random graph
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                bad = [0] * num_subsets
                total_bad = 0
                for i, mask in enumerate(subsets6):
                    if is_regular_mask(mask, adj, n):
                        bad[i] = 1
                        total_bad += 1
                no_improve_count = 0
                continue
            
            if total_bad == 0:
                # If we have 0 bad 6-sets, but failed full check, 
                # we can't easily pick a "bad" subset to fix.
                # We pick a random edge and flip it, hoping to break the larger regular subgraph.
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v: continue
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # Pick a random bad 6-subset
                # To speed up, we can maintain a list of bad indices
                # But for now, just pick a random index and check if it's bad? 
                # Or maintain a list.
                # Let's pick a random index from 0 to num_subsets-1
                # and check if bad[i] is 1. If not, try again? 
                # Better: maintain a list of bad indices.
                
                # Optimization: keep a list of bad indices
                # We'll update it as we go.
                # For simplicity in this loop, let's just pick a random index 
                # and if it's not bad, we just wasted a step. 
                # Given num_subsets is ~50k, and total_bad might be small, 
                # this could be slow.
                
                # Let's maintain a list `bad_indices`
                # We need to update `bad_indices` when we flip an edge.
                
                # Actually, let's just pick a random edge from the whole graph 
                # and see if it reduces bad count? 
                # No, the strategy is to pick a violating set.
                
                # Let's maintain a list of bad indices.
                # This requires careful updating.
                
                # Alternative: 
                # If total_bad is small, finding a bad index is fast.
                # If total_bad is large, we can just pick a random index.
                
                # Let's try to pick a random index and check.
                idx = random.randint(0, num_subsets - 1)
                if bad[idx] == 0:
                    # Retry a few times
                    for _ in range(10):
                        idx = random.randint(0, num_subsets - 1)
                        if bad[idx] == 1:
                            break
                    if bad[idx] == 0:
                        # Just pick a random edge
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u == v: continue
                else:
                    # Found a bad subset
                    mask = subsets6[idx]
                    verts = []
                    m = mask
                    while m:
                        lsb = m & -m
                        u = lsb.bit_length() - 1
                        verts.append(u)
                        m ^= lsb
                    
                    # Pick random edge within this subset
                    u, v = random.sample(verts, 2)
            
            # Flip edge (u, v)
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Update bad status for all subsets containing (u, v)
            affected_indices = edge_map[u][v]
            for i in affected_indices:
                old_bad = bad[i]
                new_bad = 1 if is_regular_mask(subsets6[i], adj, n) else 0
                if old_bad != new_bad:
                    bad[i] = new_bad
                    if old_bad:
                        total_bad -= 1
                    else:
                        total_bad += 1
                    
        # Time up
        # Check if best_adj is valid
        if best_total_bad == 0 and verify_full(best_adj, n):
            return n, get_edges_from_adj(best_adj, n)
            
        return None, None

    # Try n=21 first
    # Give it 60 seconds
    result = solve_n(21, 60.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Try n=20
    # Give it 60 seconds
    result = solve_n(20, 60.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Fallback: n=5
    adj = [0] * 5
    for i in range(5):
        for j in range(i + 1, 5):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 5, 'edges': get_edges_from_adj(adj, 5)}
