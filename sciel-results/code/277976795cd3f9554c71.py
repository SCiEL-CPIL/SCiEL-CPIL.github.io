import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0
    
    def now():
        return time.time()
    
    # Build PG(2,5) polarity graph
    q = 5
    # Points of PG(2,q): (x,y,1) for x,y in F_q; (x,1,0) for x in F_q; (1,0,0)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n0 = len(points)  # 31
    
    def is_orthogonal(p, r):
        x1, y1, z1 = p
        x2, y2, z2 = r
        return (x1 * y2 + y1 * x2 + z1 * z2) % q == 0
    
    # Build adjacency as bitsets
    adj = [0] * n0
    for i in range(n0):
        for j in range(i + 1, n0):
            if is_orthogonal(points[i], points[j]):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Fast independent set check using bitset branching
    def has_indep_set_k(adj, n, k, timeout=None):
        """Check if graph with adjacency bitsets has independent set of size k."""
        if k <= 0:
            return True
        if k == 1:
            return n >= 1
        if k > n:
            return False
        
        # Use recursive branching with bitsets
        # remaining: bitset of available vertices
        # We want to find k vertices with no edges between them
        
        def dfs(remaining, count):
            if count == k:
                return True
            # Upper bound: can we still reach k?
            if count + popcount(remaining) < k:
                return False
            if remaining == 0:
                return False
            
            # Pick a vertex to branch on
            # Use a heuristic: pick vertex with max degree in remaining subgraph
            # For speed, just pick lowest set bit
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            
            # Option 1: include v -> remove v and its neighbors from remaining
            new_rem = remaining & ~(v_bit | adj[v])
            if dfs(new_rem, count + 1):
                return True
            
            # Option 2: exclude v
            new_rem = remaining & ~v_bit
            if dfs(new_rem, count):
                return True
            
            return False
        
        all_vertices = (1 << n) - 1
        return dfs(all_vertices, 0)
    
    def popcount(x):
        return bin(x).count('1')
    
    # Conflict graph: two vertices conflict if they share a common neighbor
    # conflict[u] is bitset of vertices that share a common neighbor with u
    def build_conflict(adj, n):
        conflict = [0] * n
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                # Check if u and v have a common neighbor
                # N(u) & N(v) != 0
                if (adj[u] & adj[v]) != 0:
                    conflict[u] |= (1 << v)
                    conflict[v] |= (1 << u)
        return conflict
    
    conflict = build_conflict(adj, n0)
    
    # State
    n = n0
    best_n = n0
    best_adj = adj[:]
    
    # Verify C4-freeness and alpha
    def verify_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if popcount(common) > 1:
                    return False
        return True
    
    def compute_alpha_lower_bound(adj, n):
        """Greedy: find a large independent set."""
        remaining = (1 << n) - 1
        count = 0
        while remaining:
            # Pick vertex with min degree in remaining
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                v_bit = temp & (-temp)
                v = v_bit.bit_length() - 1
                deg = popcount(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
                temp &= temp - 1
            count += 1
            remaining &= ~((1 << best_v) | adj[best_v])
        return count
    
    # Try to add vertices
    def try_add_vertex(adj, conflict, n, S_bitset, timeout_check=None):
        """Try to add a vertex connected to vertices in S_bitset.
        Returns (success, new_adj, new_conflict, new_n) or (False, None, None, None)"""
        if timeout_check and now() > timeout_check:
            return False, None, None, None
        
        # C4 check: S must be independent in conflict graph
        # For any u,v in S, u and v must not conflict (no common neighbor)
        temp = S_bitset
        while temp:
            u_bit = temp & (-temp)
            u = u_bit.bit_length() - 1
            # Check if u conflicts with any other vertex in S
            if (conflict[u] & S_bitset) & ~u_bit:
                return False, None, None, None
            temp &= temp - 1
        
        # Build new adjacency
        new_n = n + 1
        new_adj = adj[:]
        new_adj.append(0)  # new vertex v
        
        # Connect v to all in S
        v_idx = n
        temp = S_bitset
        while temp:
            u_bit = temp & (-temp)
            u = u_bit.bit_length() - 1
            new_adj[u] |= (1 << v_idx)
            new_adj[v_idx] |= u_bit
            temp &= temp - 1
        
        # Check alpha: need no independent set of size 11
        # alpha(new_adj) = max(alpha(adj), 1 + alpha(adj without N(v) and without v))
        # More precisely: an independent set either avoids v (size ≤ alpha(adj)) 
        # or includes v (size ≤ 1 + alpha(adj restricted to V \ ({v} ∪ N(v))))
        # V \ ({v} ∪ N(v)) = V \ S (since N(v) = S)
        # So we need: alpha(adj restricted to V\S) ≤ 9
        
        # Check if adj restricted to V\S has independent set of size 10
        # If yes, then new graph has independent set of size 11
        remaining_mask = ((1 << n) - 1) & ~S_bitset
        
        # Check if there's an independent set of size 10 in the subgraph induced by remaining_mask
        # We can check by seeing if the subgraph has alpha ≥ 10
        
        # For efficiency, check if there's an independent set of size 10
        # Use the same dfs but on the subgraph
        
        def has_indep_k_subgraph(adj, n, k, mask):
            """Check if subgraph induced by mask has independent set of size k."""
            if k <= 0:
                return True
            if k == 1:
                return mask != 0
            if popcount(mask) < k:
                return False
            
            def dfs(rem, count):
                if count == k:
                    return True
                if count + popcount(rem) < k:
                    return False
                if rem == 0:
                    return False
                
                v_bit = rem & (-rem)
                v = v_bit.bit_length() - 1
                
                # Include v
                new_rem = rem & ~(v_bit | (adj[v] & rem))
                if dfs(new_rem, count + 1):
                    return True
                
                # Exclude v
                new_rem = rem & ~v_bit
                if dfs(new_rem, count):
                    return True
                
                return False
            
            return dfs(mask, 0)
        
        if has_indep_k_subgraph(adj, n, 10, remaining_mask):
            return False, None, None, None
        
        # Valid! Build new conflict
        new_conflict = [c | 0 for c in conflict]  # copy
        new_conflict.append(0)
        
        # Update conflict for new vertex v
        # v conflicts with u if N(v) ∩ N(u) ≠ 0
        # N(v) = S
        # So v conflicts with u if S ∩ N(u) ≠ 0, i.e., if any vertex in S is adjacent to u
        temp = S_bitset
        while temp:
            u_bit = temp & (-temp)
            u = u_bit.bit_length() - 1
            if (new_adj[u] & S_bitset) != 0:
                # u and v share a common neighbor (some vertex in S adjacent to u)
                new_conflict[u] |= (1 << v_idx)
                new_conflict[v_idx] |= u_bit
            temp &= temp - 1
        
        # Also, pairs within S now conflict (they share v as common neighbor)
        # Wait: do they? N(u) contains v, N(w) contains v for u,w in S.
        # So u and w share v as common neighbor. Yes, they conflict.
        # But they might have already conflicted.
        temp = S_bitset
        while temp:
            u_bit = temp & (-temp)
            u = u_bit.bit_length() - 1
            # All other vertices in S conflict with u
            new_conflict[u] |= (S_bitset & ~u_bit)
            temp &= temp - 1
        
        return True, new_adj, new_conflict, new_n
    
    # Main search loop
    adj = adj[:]
    conflict = conflict[:]
    n = n0
    best_n = n0
    best_adj = adj[:]
    
    # Try to add vertices
    while now() < deadline:
        added = False
        
        # Strategy: try various sizes of S
        # Start with small S (size 1, 2, ...) then larger
        max_S_size = min(n, 8)
        
        for S_size in range(1, max_S_size + 1):
            if now() > deadline:
                break
            
            attempts = 0
            max_attempts = 200 if S_size <= 2 else 50
            
            while attempts < max_attempts and now() < deadline:
                attempts += 1
                
                # Generate random independent set of size S_size in conflict graph
                # Greedy random
                S = set()
                available = (1 << n) - 1
                
                for _ in range(S_size):
                    if available == 0:
                        break
                    # Pick random vertex from available that doesn't conflict with S
                    candidates = available
                    # Remove vertices that conflict with any in S
                    for u in S:
                        candidates &= ~conflict[u]
                    if candidates == 0:
                        break
                    
                    # Pick random from candidates
                    # Convert to list and pick random
                    cand_list = []
                    temp = candidates
                    while temp:
                        b = temp & (-temp)
                        cand_list.append(b.bit_length() - 1)
                        temp &= temp - 1
                    
                    if not cand_list:
                        break
                    u = random.choice(cand_list)
                    S.add(u)
                    # Remove u and its conflicts from available
                    available &= ~((1 << u) | conflict[u])
                
                if len(S) != S_size:
                    continue
                
                S_bitset = 0
                for u in S:
                    S_bitset |= (1 << u)
                
                success, new_adj, new_conflict, new_n = try_add_vertex(
                    adj, conflict, n, S_bitset, deadline
                )
                
                if success:
                    adj = new_adj
                    conflict = new_conflict
                    n = new_n
                    
                    if n > best_n:
                        best_n = n
                        best_adj = adj[:]
                    
                    added = True
                    break  # Break inner while, try next S_size or continue
            
            if added:
                break  # Break S_size loop, go to next outer iteration
        
        if not added:
            # Try single vertex additions more aggressively
            found = False
            for _ in range(50):
                if now() > deadline:
                    break
                u = random.randint(0, n - 1)
                S_bitset = 1 << u
                success, new_adj, new_conflict, new_n = try_add_vertex(
                    adj, conflict, n, S_bitset, deadline
                )
                if success:
                    adj = new_adj
                    conflict = new_conflict
                    n = new_n
                    if n > best_n:
                        best_n = n
                        best_adj = adj[:]
                    found = True
                    break
            
            if not found:
                break
    
    # Build edge list from best_adj
    edges = []
    for u in range(best_n):
        for v in range(u + 1, best_n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return {'n': best_n, 'edges': edges}

construct()
