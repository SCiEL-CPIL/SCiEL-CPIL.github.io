"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 256
    
    # Precompute conflict graph: two elements conflict if they form a 3-AP with some third element in [1, N]
    # For MWIS, we need: edge between u and v if there exists w in [1,N] such that {u,v,w} is a 3-AP
    # This means: |u - v| = |v - w| or |u - v| = |u - w| or |v - w| = |u - w|
    # Actually, {u,v,w} is a 3-AP iff one of them is the midpoint of the other two.
    # So u and v conflict if:
    #   - There exists w such that u + w = 2v (v is midpoint), i.e., w = 2v - u, and 1 <= w <= N, w != u, w != v
    #   - There exists w such that u + w = 2v... wait, let me be more careful.
    #   {u,v,w} is a 3-AP iff (after sorting) the middle is the average of the other two.
    #   So for pair (u,v) with u < v:
    #     - u is midpoint: v + w = 2u => w = 2u - v. Need 1 <= w <= N, w != u, w != v. Since u < v, w = 2u-v < u, so w < u < v. Need w >= 1.
    #     - v is midpoint: u + w = 2v => w = 2v - u. Need 1 <= w <= N, w != u, w != v. Since u < v, w = 2v-u > v, so u < v < w. Need w <= N.
    #     - w is midpoint: u + v = 2w => w = (u+v)/2. Need u+v even, and w != u, w != v (which is true since u < v).
    
    # Build conflict adjacency list
    conflict_adj = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            # Check if u and v conflict (i.e., there exists w in [1,N], w != u, w != v, such that {u,v,w} is a 3-AP)
            conflicts = False
            
            # Case 1: u is midpoint. w = 2*u - v
            w = 2 * u - v
            if 1 <= w <= N and w != u and w != v:
                conflicts = True
            
            # Case 2: v is midpoint. w = 2*v - u
            if not conflicts:
                w = 2 * v - u
                if 1 <= w <= N and w != u and w != v:
                    conflicts = True
            
            # Case 3: w is midpoint. w = (u+v)/2
            if not conflicts and (u + v) % 2 == 0:
                w = (u + v) // 2
                if 1 <= w <= N and w != u and w != v:
                    conflicts = True
            
            if conflicts:
                conflict_adj[u].append(v)
                conflict_adj[v].append(u)
    
    # Precompute reciprocal weights
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Branch and bound for Maximum Weight Independent Set
    # Use bitset representation for the remaining candidate set
    # Sort vertices by weight (descending) for better pruning
    
    # Order vertices by decreasing weight
    vertex_order = list(range(1, N + 1))
    vertex_order.sort(key=lambda x: -recip[x])
    
    # Map from vertex to position in vertex_order
    pos_in_order = [0] * (N + 1)
    for i, v in enumerate(vertex_order):
        pos_in_order[v] = i
    
    # For each vertex, precompute which other vertices (in order) it conflicts with
    # conflict_mask[v] = bitmask of positions in vertex_order that conflict with v
    conflict_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        mask = 0
        for u in conflict_adj[v]:
            mask |= (1 << pos_in_order[u])
        conflict_mask[v] = mask
    
    # Precompute for each position in vertex_order, the cumulative weight from that position to the end
    # This gives an upper bound on the weight we can still add
    suffix_weights = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_weights[i] = suffix_weights[i + 1] + recip[vertex_order[i]]
    
    best_weight = 0.0
    best_solution = [False] * (N + 1)
    
    def branch_and_bound(candidates_mask, current_weight, current_solution, depth=0):
        """
        candidates_mask: bitmask of vertices (by position in vertex_order) still available
        current_weight: sum of reciprocals of vertices chosen so far
        current_solution: list of booleans, True if vertex is in solution
        """
        nonlocal best_weight, best_solution
        
        if (time.time() - start_time) > DEADLINE:
            return
        
        # Update best if current is better
        if current_weight > best_weight:
            best_weight = current_weight
            best_solution = current_solution[:]
        
        # If no candidates left, return
        if candidates_mask == 0:
            return
        
        # Upper bound: current_weight + sum of all remaining weights
        # Find the highest position set in candidates_mask
        # We can use a tighter bound: for each remaining vertex, we can only take it if it doesn't conflict with chosen vertices
        # But for simplicity, use the suffix bound from the lowest available position
        
        # Find lowest set bit position
        # Actually, let's find the highest index i such that bit i is set in candidates_mask
        # The upper bound is current_weight + suffix_weights[i] where i is the min position with a candidate
        # Wait, suffix_weights[i] = sum of weights from position i to N-1
        # So if the lowest available position is i, upper bound = current_weight + suffix_weights[i]
        
        # Find lowest set bit
        if candidates_mask == 0:
            return
        
        # Get the position of the lowest set bit
        # We want to try vertices in order of decreasing weight, so position 0 is highest weight
        # Let's pick the vertex with highest weight among candidates
        # Find highest weight vertex in candidates
        best_pos = -1
        for i in range(N):
            if (candidates_mask >> i) & 1:
                best_pos = i
                break
        
        if best_pos == -1:
            return
        
        v = vertex_order[best_pos]
        
        # Upper bound check
        ub = current_weight + suffix_weights[best_pos]
        if ub <= best_weight + 1e-15:
            return
        
        # Option 1: Include vertex v
        # New candidates = candidates_mask without v and without all vertices that conflict with v
        new_mask = candidates_mask & ~(1 << best_pos)
        new_mask &= ~conflict_mask[v]
        
        new_solution = current_solution[:]
        new_solution[v] = True
        branch_and_bound(new_mask, current_weight + recip[v], new_solution, depth + 1)
        
        # Option 2: Exclude vertex v
        new_mask2 = candidates_mask & ~(1 << best_pos)
        branch_and_bound(new_mask2, current_weight, current_solution, depth + 1)
    
    # Initial candidates: all vertices
    initial_mask = (1 << N) - 1
    initial_solution = [False] * (N + 1)
    
    # Run branch and bound
    branch_and_bound(initial_mask, 0.0, initial_solution)
    
    # Extract result
    result = [v for v in range(1, N + 1) if best_solution[v]]
    
    # If B&B didn't find anything or timed out badly, fall back to greedy
    if not result:
        # Simple greedy fallback
        bitmask = 0
        for x in range(1, N + 1):
            # Check if x can be added
            valid = True
            for u in conflict_adj[x]:
                if (bitmask >> (u - 1)) & 1:
                    valid = False
                    break
            if valid:
                bitmask |= (1 << (x - 1))
        result = [x for x in range(1, N + 1) if (bitmask >> (x - 1)) & 1]
    
    return result
# EVOLVE-BLOCK-END
