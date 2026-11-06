import itertools
import random
import time

def construct():
    # We want to maximize n such that no 6+ vertex subset induces a regular subgraph.
    # Known: 20-vertex graphs exist. Let's try to construct one via local search.
    
    def try_build(n, max_time=110):
        start_time = time.time()
        
        # 1. Precompute all 6-vertex subsets
        subsets6 = list(itertools.combinations(range(n), 6))
        num_subsets = len(subsets6)
        
        # Map subset to index for quick lookup
        subset_to_idx = {s: i for i, s in enumerate(subsets6)}
        
        # For each subset, store the 6 vertices and the 15 edges (as pairs of indices in the subset)
        # Actually, we just need the vertices.
        # We also need to know, for each edge (u, v), which subsets contain it.
        
        # Initialize adjacency matrix
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Precompute affected subsets for each edge
        # edge_to_subsets: dict mapping (u, v) to list of subset indices
        edge_to_subsets = {}
        for i in range(n):
            for j in range(i+1, n):
                edge_to_subsets[(i, j)] = []
        
        for idx, subset in enumerate(subsets6):
            # Get all pairs within this subset
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = subset[i], subset[j]
                    edge_to_subsets[(u, v)].append(idx)
                    # Note: (v, u) is same as (u, v) in undirected graph
                    # We store canonical (min, max)
        
        # State for each subset:
        # degs: list of 6 degrees (within the subset)
        # is_reg: 1 if regular, 0 otherwise
        # We need to initialize these.
        
        subset_degs = [None] * num_subsets
        subset_is_reg = [0] * num_subsets
        total_violations = 0
        
        def update_subset_state(idx):
            """Calculate degrees and regularity for subset idx."""
            s = subsets6[idx]
            degs = [0]*6
            # Map vertex to index in subset
            # Since subset is sorted, we can just use the values directly?
            # No, we need to count neighbors within s.
            for i in range(6):
                u = s[i]
                for j in range(6):
                    if i != j:
                        v = s[j]
                        if adj[u][v]:
                            degs[i] += 1
            
            # Check if regular
            first = degs[0]
            reg = 1
            for d in degs[1:]:
                if d != first:
                    reg = 0
                    break
            
            subset_degs[idx] = degs
            subset_is_reg[idx] = reg
            return reg
        
        # Initialize all subset states
        for idx in range(num_subsets):
            r = update_subset_state(idx)
            if r:
                total_violations += 1
        
        # Helper to check if a specific subset is regular
        def is_subset_regular(idx):
            return subset_is_reg[idx] == 1
        
        # Helper to flip edge and update states
        def flip_edge(u, v):
            nonlocal total_violations
            # Canonicalize
            if u > v: u, v = v, u
            
            old_val = adj[u][v]
            adj[u][v] = 1 - old_val
            adj[v][u] = 1 - old_val
            
            # Get affected subsets
            affected = edge_to_subsets[(u, v)]
            
            for idx in affected:
                s = subsets6[idx]
                # Find indices of u and v in s
                # s is a tuple of 6 sorted vertices
                # u and v are in s
                # We need to update degrees of u and v in this subset
                # degs[i] is degree of s[i]
                
                degs = subset_degs[idx]
                
                # Find position of u and v in s
                # Since s is small, linear search is fine
                pos_u = -1
                pos_v = -1
                for i in range(6):
                    if s[i] == u:
                        pos_u = i
                    elif s[i] == v:
                        pos_v = i
                
                # Update degrees
                # If we added an edge, degrees increase by 1. If removed, decrease by 1.
                delta = 1 if old_val == 0 else -1
                degs[pos_u] += delta
                degs[pos_v] += delta
                
                # Check regularity
                first = degs[0]
                reg = 1
                for d in degs[1:]:
                    if d != first:
                        reg = 0
                        break
                
                old_reg = subset_is_reg[idx]
                subset_is_reg[idx] = reg
                
                if old_reg != reg:
                    if reg:
                        total_violations += 1
                    else:
                        total_violations -= 1
                        
            # Return the change in violations
            # We need to know the violation count BEFORE the flip to calculate delta?
            # No, we just updated total_violations.
            # To get delta, we should have stored the old total_violations.
            # Let's modify flip_edge to return delta.
            
            # Actually, let's just return the new total_violations?
            # No, we need delta to compare candidates.
            # Let's refactor: 
            # 1. Calculate delta by simulating on a copy? Too slow.
            # 2. Calculate delta by looking at the affected subsets' old and new status.
            
            # Let's re-implement the delta calculation inside the loop properly.
            pass

        # Let's do a simpler loop:
        # 1. Calculate delta for each edge.
        # 2. Pick best.
        # 3. Apply.
        
        def calc_delta(u, v):
            """Calculate change in total violations if we flip edge (u, v)."""
            if u > v: u, v = v, u
            old_val = adj[u][v]
            delta = 0
            affected = edge_to_subsets[(u, v)]
            
            for idx in affected:
                s = subsets6[idx]
                degs = subset_degs[idx]
                pos_u = s.index(u)
                pos_v = s.index(v)
                
                # Simulate flip
                d_u = degs[pos_u] + (1 if old_val == 0 else -1)
                d_v = degs[pos_v] + (1 if old_val == 0 else -1)
                
                # Create new degs list for check
                new_degs = degs[:]
                new_degs[pos_u] = d_u
                new_degs[pos_v] = d_v
                
                # Check old reg
                old_reg = subset_is_reg[idx]
                
                # Check new reg
                first = new_degs[0]
                new_reg = 1
                for d in new_degs[1:]:
                    if d != first:
                        new_reg = 0
                        break
                
                if old_reg != new_reg:
                    if new_reg:
                        delta += 1
                    else:
                        delta -= 1
            return delta

        def apply_flip(u, v):
            nonlocal total_violations
            if u > v: u, v = v, u
            old_val = adj[u][v]
            adj[u][v] = 1 - old_val
            adj[v][u] = 1 - old_val
            
            affected = edge_to_subsets[(u, v)]
            for idx in affected:
                s = subsets6[idx]
                degs = subset_degs[idx]
                pos_u = s.index(u)
                pos_v = s.index(v)
                
                delta_deg = 1 if old_val == 0 else -1
                degs[pos_u] += delta_deg
                degs[pos_v] += delta_deg
                
                first = degs[0]
                reg = 1
                for d in degs[1:]:
                    if d != first:
                        reg = 0
                        break
                
                old_reg = subset_is_reg[idx]
                subset_is_reg[idx] = reg
                if old_reg != reg:
                    if reg:
                        total_violations += 1
                    else:
                        total_violations -= 1

        # Main loop
        for iteration in range(50000):
            if time.time() - start_time > max_time:
                break
            
            if total_violations == 0:
                # Check for larger violations
                if _verify_large(n, adj):
                    return adj, True
                else:
                    # Find a large violation and try to break it
                    # This is a bit tricky. We can just do random flips for a bit.
                    # Or pick a random edge in a known large violation.
                    # For simplicity, let's just do 10 random flips.
                    for _ in range(10):
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u != v:
                            apply_flip(u, v)
                    continue
            
            # Find best edge to flip
            best_delta = 0
            best_edge = None
            # Sample a subset of edges to speed up if n is large?
            # For n=21, ~210 edges. Checking all is C(21,2) * C(19,4) ~ 210 * 3876 ~ 800k ops.
            # This is too slow for 50000 iterations.
            # We need to be smarter.
            # Only check edges that are part of current violations?
            # If an edge is not in any violation, flipping it might create violations.
            # If an edge is in many violations, flipping it might fix them.
            
            # Let's collect edges that appear in violations.
            viol_edges = set()
            # We don't have a direct list of violating subsets.
            # We can iterate through all subsets? 3876 subsets.
            # For each violating subset, add its 15 edges to viol_edges.
            for idx in range(num_subsets):
                if subset_is_reg[idx]:
                    s = subsets6[idx]
                    for i in range(6):
                        for j in range(i+1, 6):
                            u, v = s[i], s[j]
                            if u > v: u, v = v, u
                            viol_edges.add((u, v))
            
            if not viol_edges:
                # No violations? Should have returned.
                # Maybe numerical error?
                break
                
            # Try flipping edges in viol_edges
            # Also try some random edges to escape local minima
            candidates = list(viol_edges)
            random.shuffle(candidates)
            # Add some random edges
            for _ in range(10):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    if u > v: u, v = v, u
                    candidates.append((u, v))
            
            improved = False
            for (u, v) in candidates:
                d = calc_delta(u, v)
                if d < best_delta:
                    best_delta = d
                    best_edge = (u, v)
                    
            if best_edge is not None and best_delta < 0:
                apply_flip(best_edge[0], best_edge[1])
                improved = True
            else:
                # No improving move. Do random flips.
                for _ in range(5):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        apply_flip(u, v)
                        
        return adj, False

    def _verify_large(n, adj):
        """Check if any subset of size > 6 is regular."""
        # Check sizes 7 to n
        # This is expensive. We'll do it carefully.
        for k in range(7, n+1):
            for subset in itertools.combinations(range(n), k):
                # Quick check: degrees
                degs = []
                for v in subset:
                    d = 0
                    for u in subset:
                        if u != v and adj[v][u]:
                            d += 1
                    degs.append(d)
                if len(set(degs)) == 1:
                    return False
        return True

    # Try n=21 first
    adj, success = try_build(21, max_time=50)
    if success:
        edges = []
        for i in range(21):
            for j in range(i+1, 21):
                if adj[i][j]:
                    edges.append([i, j])
        return {'n': 21, 'edges': edges}
    
    # Try n=20
    adj, success = try_build(20, max_time=50)
    if success:
        edges = []
        for i in range(20):
            for j in range(i+1, 20):
                if adj[i][j]:
                    edges.append([i, j])
        return {'n': 20, 'edges': edges}
    
    # Fallback
    return {'n': 5, 'edges': []}
