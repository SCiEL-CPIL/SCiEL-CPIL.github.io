"""
Hexagon-free graph construction on 40 vertices - Genetic Algorithm approach.
Uses bitsets for fast C6 detection and a population-based search strategy.
"""
import time
import random
import numpy as np

N = 40
MAX_EDGES = N * (N - 1) // 2  # 780

# Precompute edge index mapping
EDGE_IDX = {}
ALL_EDGES = []
for i in range(N):
    for j in range(i + 1, N):
        idx = len(ALL_EDGES)
        ALL_EDGES.append((i, j))
        EDGE_IDX[(i, j)] = idx
        EDGE_IDX[(j, i)] = idx

# Bitset representation for adjacency
# adj[i] is an integer where bit j is set if edge (i, j) exists

def _edge_to_bit(u, v):
    return (1 << u) | (1 << v)

def _check_c6_fast(adj):
    """
    Check if graph with bitset adjacency has a C6.
    A C6 is v0-v1-v2-v3-v4-v5-v0.
    We iterate over pairs (v0, v1) and (v3, v4) which are non-adjacent? No.
    A C6 consists of 6 vertices. 
    Efficient check: For every pair of non-adjacent vertices (u, v), 
    check if there is a path of length 4 between them.
    Path of length 4: u-w1-w2-w3-v.
    This is equivalent to checking if (A^4)[u][v] > 0 and not (A[u][v] or A^2[u][v] etc?)
    Actually, simpler: 
    For each edge (u, v), check if there is a path of length 4 from v to u 
    that does not use u or v internally? No, that's not right.
    
    Let's use the standard enumeration but optimized with bitsets.
    For each u:
      For each v in adj[u]:
        For each w in adj[v] (w != u):
          For each x in adj[w] (x not in {u, v}):
            # Now we have path u-v-w-x. We need y such that x-y-z-u? No, x-y-z-u is length 3 from x to u.
            # We need path u-v-w-x-y-z-u.
            # So from x, we need to reach u in 3 steps via distinct vertices y, z.
            # i.e., there exists y in adj[x] (y not in {u,v,w}) such that 
            # there exists z in adj[y] (z not in {u,v,w,x}) such that z in adj[u].
            
            # Let's iterate:
            # Candidates for y: adj[x] & ~(bit u | bit v | bit w)
            # For each y, candidates for z: adj[y] & ~(bit u | bit v | bit w | bit x)
            # If any such z is in adj[u], we have a C6.
            
    This is O(N * d^4) roughly. With d ~ 5-6, 40 * 6^4 = 51840 ops. Fast enough.
    """
    # Precompute complement masks? No, just use bit ops.
    all_ones = (1 << N) - 1
    
    for u in range(N):
        # Neighbors of u
        neighbors_u = adj[u]
        # Iterate v in neighbors_u
        # To avoid duplicates, we can just iterate all, or enforce v > u? 
        # Since it's undirected, checking all (u, v) pairs is fine, just redundant.
        # Let's just iterate v in neighbors_u.
        
        # We want to find path u-v-w-x-y-z-u
        # v is neighbor of u.
        # w is neighbor of v, w != u
        # x is neighbor of w, x not in {u, v}
        # y is neighbor of x, y not in {u, v, w}
        # z is neighbor of y, z not in {u, v, w, x}
        # u is neighbor of z
        
        # Optimization: 
        # For a fixed u, v:
        #   mask_uv = ~(1<<u) | ~(1<<v) -- careful with bit logic
        #   mask_uv = all_ones ^ ((1<<u) | (1<<v))
        
        # Let's just use the nested loops with bitset intersections.
        
        # Step 1: v in adj[u]
        # Step 2: w in adj[v] \ {u}
        # Step 3: x in adj[w] \ {u, v}
        # Step 4: y in adj[x] \ {u, v, w}
        # Step 5: z in adj[y] \ {u, v, w, x}
        # Step 6: check if u in adj[z]
        
        # This is still potentially slow if degrees are high.
        # Alternative: 
        # A C6 exists iff there are two vertices u, v with dist(u,v)=4? No.
        
        # Let's stick to the enumeration but break early.
        for v in range(N):
            if not (adj[u] & (1 << v)):
                continue
            if v <= u: continue # Optional symmetry break? No, u,v is directed in loop.
            
            # Path u-v
            # Next vertex w must be neighbor of v, not u
            mask_w = adj[v] & ~(1 << u)
            # Iterate w
            w_bit = mask_w
            while w_bit:
                lsb = w_bit & -w_bit
                w = lsb.bit_length() - 1
                w_bit ^= lsb
                
                # Path u-v-w
                # Next vertex x must be neighbor of w, not u, v
                mask_x = adj[w] & ~(1 << u) & ~(1 << v)
                x_bit = mask_x
                while x_bit:
                    lsb = x_bit & -x_bit
                    x = lsb.bit_length() - 1
                    x_bit ^= lsb
                    
                    # Path u-v-w-x
                    # Next vertex y must be neighbor of x, not u, v, w
                    mask_y = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    y_bit = mask_y
                    while y_bit:
                        lsb = y_bit & -y_bit
                        y = lsb.bit_length() - 1
                        y_bit ^= lsb
                        
                        # Path u-v-w-x-y
                        # Next vertex z must be neighbor of y, not u, v, w, x
                        mask_z = adj[y] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                        # z must also be neighbor of u to close the cycle
                        # So z in (mask_z & adj[u])
                        if mask_z & adj[u]:
                            return True
    return False

def _count_edges(adj):
    count = 0
    for i in range(N):
        count += bin(adj[i]).count('1')
    return count // 2

def _init_adj():
    return [0] * N

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges."""
    adj = _init_adj()
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                _add_edge(adj, base + a, base + b)
    return adj

def _random_valid_graph(target_density=0.3, seed=0):
    """Generate a random graph and remove C6s greedily."""
    random.seed(seed)
    adj = _init_adj()
    edges = list(ALL_EDGES)
    random.shuffle(edges)
    
    for u, v in edges:
        if _check_c6_fast(adj):
            continue
        # Check if adding (u,v) creates C6
        # We need a function to check if adding edge (u,v) to adj creates C6.
        # Let's implement _creates_c6_adj(adj, u, v)
        
        # Temporarily add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if _check_c6_fast(adj):
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            # Keep
            pass
            
    return adj

def _creates_c6(adj, u, v):
    """Check if adding edge (u,v) creates a C6."""
    # A C6 using (u,v) must look like u-v-w-x-y-z-u or u-z-y-x-w-v-u
    # So we need a path of length 4 from v to u in the current graph, 
    # using vertices distinct from u and v.
    
    # Path v-w1-w2-w3-u
    # w1 in adj[v] \ {u}
    # w2 in adj[w1] \ {u, v}
    # w3 in adj[w2] \ {u, v, w1}
    # u in adj[w3]
    
    mask_v = adj[v] & ~(1 << u)
    w1_bit = mask_v
    while w1_bit:
        lsb = w1_bit & -w1_bit
        w1 = lsb.bit_length() - 1
        w1_bit ^= lsb
        
        mask_w1 = adj[w1] & ~(1 << u) & ~(1 << v)
        w2_bit = mask_w1
        while w2_bit:
            lsb = w2_bit & -w2_bit
            w2 = lsb.bit_length() - 1
            w2_bit ^= lsb
            
            mask_w2 = adj[w2] & ~(1 << u) & ~(1 << v) & ~(1 << w1)
            # w3 must be neighbor of w2, and neighbor of u
            # So w3 in (mask_w2 & adj[u])
            if mask_w2 & adj[u]:
                return True
    return False

def _greedy_fill(adj, max_time=1.0):
    """Try to add edges greedily until no more can be added or time limit."""
    start = time.time()
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        # Try all edges
        random.shuffle(ALL_EDGES)
        for u, v in ALL_EDGES:
            if adj[u] & (1 << v):
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
            if time.time() - start > max_time:
                break
    return _count_edges(adj)

def _local_search(adj, max_time=2.0):
    """2-opt style local search: swap edges to increase count or maintain."""
    start = time.time()
    best_adj = adj[:]
    best_count = _count_edges(adj)
    
    # We want to increase edge count. 
    # Strategy: try to add an edge. If it creates C6, try to remove an edge from the C6 to allow it?
    # Or: try to swap an existing edge for a non-edge that increases "potential"?
    # Simplest: try to add any edge. If C6, find a vertex in the C6 and try to remove an edge incident to it?
    # This is complex. Let's stick to: 
    # 1. Try to add random non-edges.
    # 2. If stuck, try to remove a random edge and see if we can add 2 edges? (Net +1)
    
    while time.time() - start < max_time:
        # Phase 1: Add
        added = False
        for _ in range(100):
            u, v = random.choice(ALL_EDGES)
            if adj[u] & (1 << v):
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            added = True
            if _count_edges(adj) > best_count:
                best_count = _count_edges(adj)
                best_adj = adj[:]
            break
        
        if not added:
            # Phase 2: Swap (Remove 1, Add 1 or 2)
            # Try to remove a random edge
            edges = [(u, v) for u, v in ALL_EDGES if adj[u] & (1 << v)]
            if edges:
                u_rem, v_rem = random.choice(edges)
                _remove_edge(adj, u_rem, v_rem)
                
                # Now try to add 2 edges
                added_count = 0
                for _ in range(50):
                    u_add, v_add = random.choice(ALL_EDGES)
                    if adj[u_add] & (1 << v_add):
                        continue
                    if _creates_c6(adj, u_add, v_add):
                        continue
                    _add_edge(adj, u_add, v_add)
                    added_count += 1
                    if added_count >= 2:
                        break
                
                if added_count >= 1:
                    if _count_edges(adj) > best_count:
                        best_count = _count_edges(adj)
                        best_adj = adj[:]
                else:
                    # Revert
                    _add_edge(adj, u_rem, v_rem)
                    
    return best_adj, best_count

def construct():
    deadline = 145  # Safety margin
    start_time = time.time()
    
    # Initialize population with diverse seeds
    pop_size = 10
    population = []
    
    # Seed 1: 8-K5
    adj1 = _seed_graph_8K5()
    population.append(adj1)
    
    # Seeds 2-5: Random valid graphs
    for i in range(4):
        adj_rand = _random_valid_graph(seed=100 + i)
        population.append(adj_rand)
    
    # Seeds 6-10: Perturbations of 8-K5
    for i in range(5):
        adj_pert = _seed_graph_8K5()
        # Add some random edges
        random.seed(200 + i)
        for _ in range(20):
            u, v = random.choice(ALL_EDGES)
            if not (adj_pert[u] & (1 << v)):
                if not _creates_c6(adj_pert, u, v):
                    _add_edge(adj_pert, u, v)
        population.append(adj_pert)
    
    best_adj = max(population, key=lambda a: _count_edges(a))
    best_count = _count_edges(best_adj)
    
    # Run local search on all initial members
    for i in range(pop_size):
        if time.time() > deadline - 5:
            break
        adj, cnt = _local_search(population[i], max_time=1.0)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
        population[i] = adj
    
    # Genetic Algorithm Loop
    while time.time() < deadline - 10:
        # Selection: Keep top half
        population.sort(key=lambda a: _count_edges(a), reverse=True)
        population = population[:pop_size // 2]
        
        # Crossover: Create offspring
        new_pop = []
        while len(new_pop) < pop_size and time.time() < deadline - 5:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
            # Uniform crossover on edges
            child = _init_adj()
            for u, v in ALL_EDGES:
                # Bit from parent1 or parent2
                # We need to check if edge exists in parent1 or parent2
                # If in parent1, use parent1's bit. Else use parent2's bit.
                # This is not a valid graph operation directly, we must reconstruct.
                
                # Simpler: Start with parent1, then randomly swap some edges with parent2
                pass
            
            # Let's use a simpler crossover: 
            # Take parent1. For each edge in parent1, with prob 0.5, keep it.
            # For each edge in parent2 not in parent1, with prob 0.5, add it.
            child = parent1[:]
            # Remove some edges
            for u, v in ALL_EDGES:
                if child[u] & (1 << v):
                    if random.random() < 0.3:
                        _remove_edge(child, u, v)
                        
            # Add some edges from parent2
            for u, v in ALL_EDGES:
                if not (child[u] & (1 << v)):
                    if parent2[u] & (1 << v):
                        if random.random() < 0.3:
                            if not _creates_c6(child, u, v):
                                _add_edge(child, u, v)
            
            # Repair: Remove any C6s? 
            # The crossover might have created C6s. 
            # We must ensure validity.
            # If _check_c6_fast(child) is True, we need to fix it.
            # Simplest: If invalid, revert to parent1 or parent2.
            if _check_c6_fast(child):
                # Try to fix by removing random edges from C6?
                # Or just discard this child and use a mutated parent.
                child = parent1[:]
            
            # Mutation: Add/remove random edges
            for _ in range(5):
                u, v = random.choice(ALL_EDGES)
                if child[u] & (1 << v):
                    if random.random() < 0.5:
                        _remove_edge(child, u, v)
                else:
                    if not _creates_c6(child, u, v):
                        _add_edge(child, u, v)
            
            # Final validity check
            if _check_c6_fast(child):
                # Fallback: use parent1
                child = parent1[:]
                
            new_pop.append(child)
        
        population = new_pop
        
        # Local search on best offspring
        for i in range(min(3, len(population))):
            if time.time() > deadline - 5:
                break
            adj, cnt = _local_search(population[i], max_time=0.5)
            population[i] = adj
            if cnt > best_count:
                best_count = cnt
                best_adj = adj
                
    # Final greedy fill on best
    if time.time() < deadline - 2:
        _greedy_fill(best_adj, max_time=2.0)
        best_count = _count_edges(best_adj)
        
    # Convert best_adj to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    # Final verification
    if _check_c6_fast(best_adj):
        # Fallback to 8-K5
        return _seed_graph_8K5_edges()
        
    return edges

def _seed_graph_8K5_edges():
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges

# Helper to convert adj to edges for fallback
def _adj_to_edges(adj):
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u] & (1 << v):
                edges.append([u, v])
    return edges

# Wrap construct to ensure it returns list of edges
_original_construct = construct

def construct():
    edges = _original_construct()
    return edges
