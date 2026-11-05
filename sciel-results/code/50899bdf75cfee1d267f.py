import numpy as np
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # Try n=47 first, then 46, etc.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        required_min_deg = n - 39
        
        # Try to construct for this n
        result = _construct_with_conflict(n, required_min_deg, deadline)
        
        if result is not None:
            return {'n': n, 'edges': result}
            
    # Fallback: return a valid small graph
    # n=39, min_deg=0 is always valid (empty graph)
    return {'n': 39, 'edges': []}

def _construct_with_conflict(n, target_deg, deadline):
    """
    Use a conflict-graph-based approach to find a C4-free graph with min degree >= target_deg.
    We model this as selecting a subset of edges from K_n such that:
      1. The selected edges form a C4-free graph.
      2. Each vertex has degree >= target_deg in the selected graph.
    
    We use a greedy heuristic with backtracking on the conflict structure.
    """
    # Total number of possible edges in K_n
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Precompute edge index for O(1) lookup
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
    
    # Precompute for each vertex, which edges are incident to it
    vertex_edges = [[] for _ in range(n)]
    for i, (u, v) in enumerate(all_edges):
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)
    
    # Precompute C4 conflicts: two edges (e1, e2) conflict if they share a vertex or if adding both would create a C4
    # Actually, for C4-freeness, the constraint is more complex. Let's think differently.
    # A set of edges is C4-free iff for any two edges (a,b) and (c,d) in the set, the four vertices a,b,c,d are not all distinct and forming a C4.
    # This is a 4-uniform hypergraph constraint, which is hard to precompute.
    
    # Alternative approach: Use the fact that for C4-free graphs, the number of edges is bounded by O(n^(3/2)).
    # For n=47, the maximum number of edges in a C4-free graph is at most around 47 * sqrt(47) / 2 ≈ 160.
    # We need min degree >= 8 (for n=47, target=8), so we need at least 47*8/2 = 187 edges.
    # This is already close to the upper bound, so the graph must be very dense.
    
    # Let's use a different strategy: Start with a known C4-free graph structure and try to optimize.
    # Polar graph / Reiman graph for n=47? Or use a finite geometry construction?
    
    # Actually, let's use a simpler approach: greedy edge addition with C4 check, but with a better ordering.
    
    # Strategy: 
    # 1. Start with empty graph.
    # 2. Sort edges by some heuristic (e.g., how many vertices they help satisfy degree constraints).
    # 3. Greedily add edges if they don't create a C4.
    # 4. Use local search to improve.
    
    # For speed, use bitmask-based adjacency for fast C4 checking.
    
    # Bitmask approach: represent adjacency as list of bitmasks
    # For n=47, we can use Python integers as bitmasks (arbitrary precision)
    
    def _try_greedy_with_order(n, target_deg, deadline, order_seed=0):
        rng = random.Random(order_seed)
        adj_mask = [0] * n  # adj_mask[i] is a bitmask of neighbors of i
        
        # Precompute all edges
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Order edges by a heuristic: prefer edges that connect vertices with currently lower degree
        # Since we start empty, all degrees are 0, so let's just randomize
        rng.shuffle(edges)
        
        added_count = 0
        for u, v in edges:
            if time.time() > deadline:
                break
            
            # Check if adding (u, v) creates a C4
            # C4 exists if there exist x, y such that:
            # x is neighbor of u, y is neighbor of v, x != y, and (x, y) is an edge
            # In bitmask terms: 
            # common = adj_mask[u] & adj_mask[v]  -- no wait, that's not right
            # We need: for some x in N(u)\{v}, y in N(v)\{u}, with x != y and (x,y) in E
            # This is equivalent to: (adj_mask[u] & adj_mask[v]) == 0 is not the right check either.
            
            # Correct check: 
            # Let A = adj_mask[u] (neighbors of u)
            # Let B = adj_mask[v] (neighbors of v)
            # Adding (u,v) creates a C4 iff there exists x in A and y in B such that x != y and (x,y) is an edge.
            # This is equivalent to: (A & B) == 0 is sufficient but not necessary.
            # Actually, we need to check if the subgraph induced by A ∪ B contains an edge between A\{v} and B\{u}.
            
            # Simpler check: 
            # For each x in N(u)\{v}, check if N(x) intersects N(v)\{u}.
            # In bitmask: for each x where adj_mask[u] has bit x set (and x != v), check if (adj_mask[x] & adj_mask[v]) has any bit set (excluding u and x).
            
            # Let's do it efficiently:
            # temp = adj_mask[u]  # neighbors of u
            # We need to check if any x in temp (x != v) has adj_mask[x] & adj_mask[v] != 0 (excluding u and x themselves)
            
            # For speed, we can precompute for each pair (u,v) whether adding (u,v) would create a C4 given the current graph.
            # But that's O(n^3) per check, which is too slow.
            
            # Let's just do the direct check for now.
            creates_c4 = False
            mask_u = adj_mask[u]
            mask_v = adj_mask[v]
            
            # Iterate over bits in mask_u (neighbors of u)
            m = mask_u
            while m:
                # Get lowest set bit
                x_bit = m & (-m)
                m = m & (m - 1)
                x = x_bit.bit_length() - 1
                if x == v:
                    continue
                # Check if adj_mask[x] has any common neighbor with v (other than u and x)
                common = adj_mask[x] & mask_v
                # Remove u and x from common if present
                common &= ~adj_mask[u]  # Remove u (since u is in adj_mask[x] iff x is in adj_mask[u])
                common &= ~(1 << x)  # Remove x
                common &= ~(1 << v)  # Remove v
                if common:
                    creates_c4 = True
                    break
            
            if not creates_c4:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                added_count += 1
        
        # Check if min degree is satisfied
        min_deg = min(bin(adj_mask[i]).count('1') for i in range(n))
        if min_deg >= target_deg:
            # Convert to edge list
            result_edges = []
            for i in range(n):
                m = adj_mask[i]
                while m:
                    x_bit = m & (-m)
                    m = m & (m - 1)
                    j = x_bit.bit_length() - 1
                    if i < j:
                        result_edges.append([i, j])
            return result_edges
        
        return None
    
    # Try multiple random orderings
    best_result = None
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        result = _try_greedy_with_order(n, target_deg, deadline, order_seed=attempt)
        if result is not None:
            return result
    
    return None

# EVOLVE-BLOCK-END
