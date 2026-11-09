import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill
    
    def is_regular_subset_fast(adj, mask):
        """Check if the subset defined by mask induces a regular subgraph."""
        if mask == 0:
            return False
        first_deg = -1
        # Iterate through set bits
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            d = (adj[v] & mask).bit_count()
            if first_deg == -1:
                first_deg = d
            elif d != first_deg:
                return False
            m ^= lsb
        return True

    def is_valid_fast(adj, n):
        """Check if graph has no regular induced subgraph of size >= 6."""
        for size in range(6, n + 1):
            for combo in combinations(range(n), size):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_regular_subset_fast(adj, mask):
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def greedy_construct(target_n):
        """Try to build a valid graph up to target_n vertices using greedy addition."""
        # Start with a small valid graph
        # n=5: any graph on 5 vertices is valid (no subset of size >= 6)
        adj = [0] * target_n
        # Build up from 0 to target_n
        for v in range(target_n):
            if v < 5:
                continue  # First 5 vertices: no constraints yet
            # Choose edges from v to previous vertices
            # Try random orderings of edge subsets
            best_edge_mask = None
            # Try several random assignments
            for attempt in range(50):
                edge_mask = 0
                for u in range(v):
                    if random.random() < 0.5:
                        edge_mask |= (1 << u)
                # Check if adding vertex v with these edges creates a regular subset >= 6
                # We need to check all subsets of size >= 6 that include v
                # And also subsets not including v (but those were already valid)
                # So we only need to check subsets that include v
                valid = True
                # Check subsets of size 6 to v+1 that include v
                # A subset S of {0,...,v} with v in S and |S| >= 6
                # The induced subgraph on S must not be regular
                for size in range(6, v + 1):
                    # Choose size-1 other vertices from {0,...,v-1}
                    for combo in combinations(range(v), size - 1):
                        mask = (1 << v)
                        for u in combo:
                            mask |= (1 << u)
                        # Compute degrees in this induced subgraph
                        # For vertex v: degree = popcount(edge_mask & mask_without_v)
                        mask_without_v = mask & ~(1 << v)
                        deg_v = (edge_mask & mask_without_v).bit_count()
                        # For each other vertex u in combo:
                        first_deg = -1
                        is_reg = True
                        # Degree of v
                        if first_deg == -1:
                            first_deg = deg_v
                        else:
                            # Check others
                            pass
                        # Check all vertices in mask
                        m = mask
                        while m:
                            lsb = m & (-m)
                            u = lsb.bit_length() - 1
                            if u == v:
                                d = deg_v
                            else:
                                d = (adj[u] & mask).bit_count()
                                # But adj[u] hasn't been updated with edges to v yet
                                # We need to account for the edge (u,v) if it exists
                                if (edge_mask >> u) & 1:
                                    d += 1
                            if first_deg == -1:
                                first_deg = d
                            elif d != first_deg:
                                is_reg = False
                                break
                            m ^= lsb
                        if is_reg:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_edge_mask = edge_mask
                    break
            
            if best_edge_mask is None:
                # Failed to add vertex v
                return None
            
            # Commit the edges
            for u in range(v):
                if (best_edge_mask >> u) & 1:
                    adj[v] |= (1 << u)
                    adj[u] |= (1 << v)
        
        # Verify the full graph is valid
        if is_valid_fast(adj, target_n):
            return adj
        return None

    def local_search(adj, n, max_time=10.0):
        """Local search to fix a graph that's close to valid."""
        start_ls = time.time()
        
        def count_bad(adj, n):
            """Count number of regular subsets of size >= 6."""
            count = 0
            for size in range(6, n + 1):
                for combo in combinations(range(n), size):
                    mask = 0
                    for v in combo:
                        mask |= (1 << v)
                    if is_regular_subset_fast(adj, mask):
                        count += 1
            return count
        
        current_bad = count_bad(adj, n)
        if current_bad == 0:
            return adj
        
        while time.time() - start_ls < max_time:
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge (u, v)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_bad = count_bad(adj, n)
            
            if new_bad < current_bad:
                current_bad = new_bad
                if current_bad == 0:
                    return adj
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        
        return adj

    # Try greedy construction for n=20
    for attempt in range(20):
        if time.time() - start_time > deadline - 20:
            break
        adj = greedy_construct(20)
        if adj is not None:
            # Verify
            if is_valid_fast(adj, 20):
                return {'n': 20, 'edges': get_edges(adj, 20)}
    
    # If greedy failed, try local search from random start
    for attempt in range(5):
        if time.time() - start_time > deadline - 5:
            break
        adj = [0] * 20
        for i in range(20):
            for j in range(i + 1, 20):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        remaining_time = deadline - 5 - (time.time() - start_time)
        if remaining_time < 1:
            break
        
        adj = local_search(adj, 20, max_time=min(remaining_time, 15.0))
        
        if is_valid_fast(adj, 20):
            return {'n': 20, 'edges': get_edges(adj, 20)}
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
