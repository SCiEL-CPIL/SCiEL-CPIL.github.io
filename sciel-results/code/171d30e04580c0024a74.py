import time
import random
import numpy as np

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # 1. Construct Polarity Graph of PG(2,5) - This is the unique (31,90) C4-free graph.
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    key = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)

    assert len(points) == 31, f"Expected 31 points, got {len(points)}"

    def is_on_polar(p, qpt):
        x, y, z = p
        X, Y, Z = qpt
        return (x * Y + y * X + z * Z) % q == 0

    # Build base graph (90 edges) using bitmask representation
    # adj_mask[i] is a 31-bit integer where bit j is set if (i,j) is an edge
    adj_mask = [0] * n
    edge_list = []  # List of (i, j) with i < j for all edges
    
    for i in range(n):
        for j in range(i + 1, n):
            if is_on_polar(points[i], points[j]):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edge_list.append((i, j))
    
    assert len(edge_list) == 90, f"Expected 90 edges, got {len(edge_list)}"
    
    # Precompute common neighbor counts for all pairs in the base graph
    # common[u][v] = number of common neighbors of u and v in current graph
    # We'll maintain this as a 2D array and update incrementally
    
    def binom2(x):
        return x * (x - 1) // 2

    def count_c4_from_adj(adj_m):
        """Count C4s from adjacency masks."""
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_m[i] >> j) & 1:
                    # Count common neighbors of i and j
                    common = bin_count(adj_m[i] & adj_m[j])
                    total += binom2(common)
        return total // 4

    def bin_count(x):
        return bin(x).count('1')
    
    # Better: use a lookup table for bin_count on 31-bit values
    # Or use int.bit_count() available in Python 3.10+
    def popcount(x):
        return x.bit_count()
    
    def count_c4_fast(adj_m):
        """Count C4s from adjacency masks using bit operations."""
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_m[i] >> j) & 1:
                    common = popcount(adj_m[i] & adj_m[j])
                    total += common * (common - 1) // 2
        return total // 4

    # Find best single edge to add
    best_add = None
    min_common = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            if not ((adj_mask[i] >> j) & 1):
                common = popcount(adj_mask[i] & adj_mask[j])
                if common < min_common:
                    min_common = common
                    best_add = (i, j)
    
    # Initialize with base + best_add
    cur_adj = adj_mask[:]
    u0, v0 = best_add
    cur_adj[u0] |= (1 << v0)
    cur_adj[v0] |= (1 << u0)
    
    cur_edges = set(edge_list) | {best_add}
    cur_c4 = count_c4_fast(cur_adj)
    
    best_c4 = cur_c4
    best_adj = cur_adj[:]
    best_edges = cur_edges.copy()
    
    random.seed(42)
    
    # Precompute all possible edges as a list
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Local search with bitmask representation
    # Key insight: when we add edge (a,b) and remove edge (c,d),
    # only the common-neighbor counts for edges incident to {a,b,c,d} change.
    # We can compute the delta in O(1) amortized time.
    
    def compute_c4_delta(remove_u, remove_v, add_u, add_v, adj_m):
        """
        Compute the change in C4 count when removing (remove_u, remove_v) 
        and adding (add_u, add_v).
        
        Returns the new C4 count.
        """
        involved = {remove_u, remove_v, add_u, add_v}
        
        # Build temporary adjacency with the swap applied
        tmp_adj = adj_m[:]
        tmp_adj[remove_u] &= ~(1 << remove_v)
        tmp_adj[remove_v] &= ~(1 << remove_u)
        tmp_adj[add_u] |= (1 << add_v)
        tmp_adj[add_v] |= (1 << add_u)
        
        # Find all edges affected by the change
        # An edge (x,y) is affected if x or y is in involved
        affected = set()
        for v in involved:
            m = tmp_adj[v]
            # Get all neighbors of v in the new graph
            while m:
                lb = m & (-m)
                u = lb.bit_length() - 1
                if u < v:
                    affected.add((u, v))
                else:
                    affected.add((v, u))
                m ^= lb
        
        # Also need to include the removed edge in the "old" affected set
        # and the added edge in the "new" affected set
        
        # Old affected edges: edges incident to involved nodes in OLD graph
        old_affected = set()
        for v in involved:
            m = adj_m[v]
            while m:
                lb = m & (-m)
                u = lb.bit_length() - 1
                if u < v:
                    old_affected.add((u, v))
                else:
                    old_affected.add((v, u))
                m ^= lb
        
        # New affected edges: edges incident to involved nodes in NEW graph
        new_affected = set()
        for v in involved:
            m = tmp_adj[v]
            while m:
                lb = m & (-m)
                u = lb.bit_length() - 1
                if u < v:
                    new_affected.add((u, v))
                else:
                    new_affected.add((v, u))
                m ^= lb
        
        # Sum of binom2(common) for old affected edges (excluding the one being removed)
        old_sum = 0
        for u, v in old_affected:
            if (u, v) == (remove_u, remove_v) or (u, v) == (remove_v, remove_u):
                continue
            common = popcount(adj_m[u] & adj_m[v])
            old_sum += common * (common - 1) // 2
        
        # Sum of binom2(common) for new affected edges
        new_sum = 0
        for u, v in new_affected:
            common = popcount(tmp_adj[u] & tmp_adj[v])
            new_sum += common * (common - 1) // 2
        
        # new_c4 = (4 * cur_c4 - old_sum + new_sum) // 4
        return (4 * cur_c4 - old_sum + new_sum) // 4
    
    # Simulated annealing / hill climbing with restarts
    iterations = 0
    max_iterations = 200000
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5 and iterations < max_iterations:
        iterations += 1
        
        # Pick a random edge to remove
        u1, v1 = random.choice(list(cur_edges))
        
        # Try several random non-edges to add
        improved = False
        for _ in range(15):
            u2, v2 = random.choice(all_pairs)
            if (u2, v2) in cur_edges or (v2, u2) in cur_edges:
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            
            new_c4 = compute_c4_delta(u1, v1, u2, v2, cur_adj)
            
            if new_c4 < cur_c4:
                # Accept the move
                cur_adj[u1] &= ~(1 << v1)
                cur_adj[v1] &= ~(1 << u1)
                cur_adj[u2] |= (1 << v2)
                cur_adj[v2] |= (1 << u2)
                
                cur_edges.discard((u1, v1))
                cur_edges.add((u2, v2))
                
                cur_c4 = new_c4
                
                if cur_c4 < best_c4:
                    best_c4 = cur_c4
                    best_adj = cur_adj[:]
                    best_edges = cur_edges.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                improved = True
                break
        
        if not improved:
            no_improve_count += 1
        
        # Periodic restart if stuck
        if no_improve_count > max_no_improve and time.time() < deadline - 20:
            # Restart from best solution with random perturbation
            cur_adj = best_adj[:]
            cur_edges = best_edges.copy()
            cur_c4 = best_c4
            
            # Apply 3-5 random swaps to perturb
            for _ in range(random.randint(3, 5)):
                # Remove random edge
                ru, rv = random.choice(list(cur_edges))
                # Add random non-edge
                for attempt in range(20):
                    au, av = random.choice(all_pairs)
                    if (au, av) not in cur_edges and (au, av) != (ru, rv):
                        cur_adj[ru] &= ~(1 << rv)
                        cur_adj[rv] &= ~(1 << ru)
                        cur_adj[au] |= (1 << av)
                        cur_adj[av] |= (1 << au)
                        cur_edges.discard((ru, rv))
                        cur_edges.add((au, av))
                        break
            
            cur_c4 = count_c4_fast(cur_adj)
            no_improve_count = 0

    return list(best_edges)
