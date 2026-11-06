"""Rewritten program for R(C4,K11) construction."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a C4-free graph with no independent set of size 11.
    Strategy: Use a finite geometry-inspired base from GF(7) polarity graph,
    select 39 vertices, then perform deterministic local search to reduce independence number.
    """
    
    def is_c4_free(adj, n):
        """Check if graph is C4-free using adjacency sets."""
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return False
        return True
    
    def find_max_independent_set(adj, n, num_trials=200):
        """Find maximum independent set size using greedy with multiple orderings."""
        best = 0
        best_set = set()
        for _ in range(num_trials):
            order = list(range(n))
            random.shuffle(order)
            independent = set()
            for v in order:
                if all(u not in independent for u in adj[v]):
                    independent.add(v)
            if len(independent) > best:
                best = len(independent)
                best_set = independent.copy()
        return best, best_set
    
    # Step 1: Build polarity graph of PG(2,7) using GF(7)
    # GF(7) is a field since 7 is prime
    q = 7
    p = q  # prime
    
    # Points of PG(2,q): 57 points total
    # Represent as: affine points (x,y) for x,y in GF(q), plus points at infinity
    # Total: q^2 + q + 1 = 49 + 7 + 1 = 57
    
    # Define GF(7) arithmetic (mod 7)
    def gf7_add(a, b):
        return (a + b) % p
    
    def gf7_sub(a, b):
        return (a - b) % p
    
    def gf7_mul(a, b):
        return (a * b) % p
    
    def gf7_inv(a):
        if a == 0:
            return 0
        # Fermat's little theorem: a^(p-2) = a^(-1) mod p
        return pow(a, p-2, p)
    
    # Create list of all 57 points
    # Points 0-48: affine (x, y) for x,y in 0..6
    # Points 49-55: points at infinity with slope m in 0..6
    # Point 56: point at infinity for vertical lines
    
    all_points = []
    for x in range(q):
        for y in range(q):
            all_points.append(('affine', x, y))
    for m in range(q):
        all_points.append(('infinity', m))
    all_points.append(('infinity_vert',))
    
    n_total = len(all_points)  # 57
    
    # Define the polarity: two points are orthogonal if they satisfy the polarity condition
    # For a standard polarity on PG(2,q), use: x*x' + y*y' + z*z' = 0 in homogeneous coords
    # For affine point (x,y,1) and (x',y',1): orthogonal if x*x' + y*y' + 1 = 0
    # For affine (x,y,1) and infinity (1,m,0): orthogonal if x + y*m = 0
    # For affine (x,y,1) and infinity_vert (0,1,0): orthogonal if y = 0
    # For infinity (1,m,0) and (1,m',0): orthogonal if m*m' = 0 (so m=0 or m'=0)
    # For infinity (1,m,0) and infinity_vert (0,1,0): orthogonal if 1*0 + m*1 = m = 0
    # For infinity_vert (0,1,0) and itself: 0*0 + 1*1 = 1 != 0 (not self-orthogonal)
    
    def are_orthogonal(p1, p2):
        """Check if two points are orthogonal w.r.t. the polarity."""
        if p1[0] == 'affine' and p2[0] == 'affine':
            x1, y1 = p1[1], p1[2]
            x2, y2 = p2[1], p2[2]
            return gf7_add(gf7_mul(x1, x2), gf7_add(gf7_mul(y1, y2), 1)) == 0
        elif p1[0] == 'affine' and p2[0] == 'infinity':
            x1, y1 = p1[1], p1[2]
            m2 = p2[1]
            return gf7_add(x1, gf7_mul(y1, m2)) == 0
        elif p1[0] == 'affine' and p2[0] == 'infinity_vert':
            y1 = p1[2]
            return y1 == 0
        elif p1[0] == 'infinity' and p2[0] == 'affine':
            return are_orthogonal(p2, p1)
        elif p1[0] == 'infinity' and p2[0] == 'infinity':
            m1, m2 = p1[1], p2[1]
            return gf7_mul(m1, m2) == 0
        elif p1[0] == 'infinity' and p2[0] == 'infinity_vert':
            m1 = p1[1]
            return m1 == 0
        elif p1[0] == 'infinity_vert' and p2[0] == 'affine':
            return are_orthogonal(p2, p1)
        elif p1[0] == 'infinity_vert' and p2[0] == 'infinity':
            return are_orthogonal(p2, p1)
        elif p1[0] == 'infinity_vert' and p2[0] == 'infinity_vert':
            return False  # 0*0 + 1*1 = 1 != 0
        return False
    
    # Build the full polarity graph adjacency
    full_adj = [set() for _ in range(n_total)]
    for i in range(n_total):
        for j in range(i+1, n_total):
            if are_orthogonal(all_points[i], all_points[j]):
                full_adj[i].add(j)
                full_adj[j].add(i)
    
    # Step 2: Select 39 vertices from the 57 that form a good subgraph
    # We want a subset of 39 vertices where the induced subgraph is C4-free
    # (it will be, since it's a subgraph of a C4-free graph) and has small independence number
    
    n_target = 39
    
    # Try multiple random subsets and pick the one with smallest independence number
    random.seed(12345)
    best_subset = None
    best_alpha = float('inf')
    best_adj_subset = None
    
    for trial in range(50):
        # Randomly select 39 vertices
        selected = random.sample(range(n_total), n_target)
        selected_set = set(selected)
        
        # Build induced subgraph
        adj_subset = [set() for _ in range(n_target)]
        idx_map = {v: i for i, v in enumerate(selected)}
        
        for i in range(n_target):
            for j in range(i+1, n_target):
                vi, vj = selected[i], selected[j]
                if vj in full_adj[vi]:
                    adj_subset[i].add(j)
                    adj_subset[j].add(i)
        
        # Check independence number
        alpha, _ = find_max_independent_set(adj_subset, n_target, 100)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_subset = selected
            best_adj_subset = adj_subset
        
        if best_alpha <= 10:
            break
    
    if best_subset is None:
        # Fallback: use first 39 vertices
        best_subset = list(range(n_target))
        best_adj_subset = [set() for _ in range(n_target)]
        for i in range(n_target):
            for j in range(i+1, n_target):
                if j in full_adj[i]:
                    best_adj_subset[i].add(j)
                    best_adj_subset[j].add(i)
    
    adj = best_adj_subset
    n = n_target
    
    # Step 3: Deterministic local search to reduce independence number
    # Add edges that don't create C4, prioritizing those that break large independent sets
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) preserves C4-freeness."""
        if u == v or v in adj[u]:
            return False
        common = adj[u] & adj[v]
        return len(common) < 2
    
    # Repeatedly find max independent set and add edges within it
    for iteration in range(2000):
        alpha, max_iset = find_max_independent_set(adj, n, 100)
        
        if alpha <= 10:
            break
        
        # Get the independent set
        indep_list = list(max_iset)
        
        # Try to add edges within the independent set
        # Prioritize pairs that have few common neighbors (easier to add)
        candidates = []
        for i in range(len(indep_list)):
            for j in range(i+1, len(indep_list)):
                u, v = indep_list[i], indep_list[j]
                if v not in adj[u]:
                    common_count = len(adj[u] & adj[v])
                    if common_count < 2:
                        candidates.append((common_count, u, v))
        
        if not candidates:
            break
        
        # Sort by common neighbor count (prefer pairs with 0 common neighbors)
        candidates.sort()
        
        # Add the best edge
        _, u, v = candidates[0]
        adj[u].add(v)
        adj[v].add(u)
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    
    # Verify C4-freeness
    if not is_c4_free(adj, n):
        # Fallback: return a valid smaller graph
        # Use 10 disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Final verification of independence number
    alpha_final, _ = find_max_independent_set(adj, n, 500)
    if alpha_final > 10:
        # Try more aggressive local search
        for _ in range(1000):
            alpha, max_iset = find_max_independent_set(adj, n, 100)
            if alpha <= 10:
                break
            indep_list = list(max_iset)
            candidates = []
            for i in range(len(indep_list)):
                for j in range(i+1, len(indep_list)):
                    u, v = indep_list[i], indep_list[j]
                    if v not in adj[u]:
                        common_count = len(adj[u] & adj[v])
                        if common_count < 2:
                            candidates.append((common_count, u, v))
            if not candidates:
                break
            candidates.sort()
            _, u, v = candidates[0]
            adj[u].add(v)
            adj[v].add(u)
        
        alpha_final, _ = find_max_independent_set(adj, n, 500)
        if alpha_final > 10:
            # Fallback
            edges = []
            for t in range(10):
                a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
                edges += [[a, b], [a, c], [b, c]]
            return {'n': 30, 'edges': edges}
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
