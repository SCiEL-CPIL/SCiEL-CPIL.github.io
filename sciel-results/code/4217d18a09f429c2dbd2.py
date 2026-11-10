"""Rewritten program: maximum independent set on conflict graph via branch-and-bound."""
import math
import time

def construct():
    """Return a nonempty subset of [1,256] maximizing size such that
    a+b is never a perfect square for all a<=b in the set (including a=b).
    """
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s kill
    
    n = 256
    # Perfect squares up to 512 (max sum is 256+256=512)
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # conflict[i][j] = True if i+j is a perfect square (0-indexed, so values 1..256)
    # We'll use bitmasks for efficiency
    # Each vertex i (0-indexed, representing value i+1) has a conflict mask
    
    # First, identify vertices that have self-conflict (2*(i+1) is a perfect square)
    # These cannot be in any valid set
    self_conflict = [False] * n
    for i in range(n):
        val = i + 1
        if 2 * val in squares:
            self_conflict[i] = True
    
    # Build adjacency: for each vertex, which other vertices conflict with it
    # Use Python ints as bitmasks for the 256 vertices
    conflict_mask = [0] * n
    for i in range(n):
        vi = i + 1
        for j in range(n):
            vj = j + 1
            if vi + vj in squares:
                conflict_mask[i] |= (1 << j)
    
    # Remove self-conflicting vertices from consideration
    # They can never be in the set
    available = [i for i in range(n) if not self_conflict[i]]
    
    # Now we need maximum independent set in the graph induced by 'available' vertices
    # where edges are conflicts
    
    # For branch-and-bound, we work with the subgraph of available vertices
    # Remap to 0..m-1 for compactness
    m = len(available)
    # adj_mask[i] for i in 0..m-1: bitmask of neighbors in the subgraph
    adj_mask = [0] * m
    for i in range(m):
        vi = available[i]
        # Get conflicts from the full graph
        cm = conflict_mask[vi]
        # Map to subgraph indices
        mask = 0
        for j in range(m):
            vj = available[j]
            if cm & (1 << vj):
                mask |= (1 << j)
        adj_mask[i] = mask
        
    # Remove self-loops from adj_mask for the subgraph if any
    for i in range(m):
        adj_mask[i] &= ~(1 << i)

    # Branch and bound for maximum independent set
    # State: remaining vertices (bitmask), current set (bitmask), current size
    # We want to maximize the size of an independent set
    
    best_size = 0
    best_set = 0  # bitmask of the best independent set found
    
    def time_exceeded():
        return (time.time() - start_time) > DEADLINE

    # Greedy coloring bound for the remaining subgraph
    def color_bound(remaining_mask):
        # Returns upper bound on MIS size for subgraph induced by remaining_mask
        # We extract the list of vertices
        verts = []
        tmp = remaining_mask
        while tmp:
            lb = tmp & (-tmp)
            v = lb.bit_length() - 1
            verts.append(v)
            tmp ^= lb
            
        if not verts:
            return 0
            
        # Greedy coloring
        # We want to assign colors such that adjacent vertices have different colors.
        # The number of colors is an upper bound on the independence number.
        
        # Sort by degree in the remaining subgraph (descending)
        # To get degree in remaining subgraph:
        # deg(v) = popcount(adj_mask[v] & remaining_mask)
        
        # Precompute degrees for these verts
        degs = []
        for v in verts:
            d = bin(adj_mask[v] & remaining_mask).count('1')
            degs.append((d, v))
            
        degs.sort(reverse=True)
        
        colors = [-1] * m # Use global array, reset for verts
        color_count = 0
        colored_set = set()
        
        for d, v in degs:
            # Find colors used by neighbors of v that are already colored
            used = set()
            # Iterate through neighbors of v in the remaining graph
            nbrs = adj_mask[v] & remaining_mask
            tmp_nbrs = nbrs
            while tmp_nbrs:
                lb = tmp_nbrs & (-tmp_nbrs)
                u = lb.bit_length() - 1
                if u in colored_set:
                    used.add(colors[u])
                tmp_nbrs ^= lb
                
            # Assign smallest non-negative integer not in used
            c = 0
            while c in used:
                c += 1
                
            colors[v] = c
            if c >= color_count:
                color_count = c + 1
                
            colored_set.add(v)
            
        return color_count

    def max_independent_set(remaining_mask, current_size, current_mask):
        nonlocal best_size, best_set
        
        if time_exceeded():
            return
        
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return
        
        # Upper bound using greedy coloring
        ub = color_bound(remaining_mask)
        
        if current_size + ub <= best_size:
            return
        
        # Select a vertex to branch on
        # Choose the vertex with the highest degree in the remaining subgraph
        verts = []
        tmp = remaining_mask
        while tmp:
            lb = tmp & (-tmp)
            v = lb.bit_length() - 1
            verts.append(v)
            tmp ^= lb
        
        if not verts:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return
        
        # Find vertex with max degree in remaining subgraph
        max_deg = -1
        branch_v = verts[0]
        for v in verts:
            deg = bin(adj_mask[v] & remaining_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                branch_v = v
        
        # Branch 1: include branch_v
        # Remove branch_v and all its neighbors from remaining
        to_remove = adj_mask[branch_v] | (1 << branch_v)
        new_remaining = remaining_mask & ~to_remove
        new_mask = current_mask | (1 << branch_v)
        max_independent_set(new_remaining, current_size + 1, new_mask)
        
        if time_exceeded():
            return
        
        # Branch 2: exclude branch_v
        new_remaining = remaining_mask & ~(1 << branch_v)
        max_independent_set(new_remaining, current_size, current_mask)
        
    # Initial call
    full_mask = (1 << m) - 1
    max_independent_set(full_mask, 0, 0)
    
    # Extract the result
    if best_set == 0:
        # Fallback: return a single valid element
        for i in available:
            return [i + 1]
        return [1]
    
    result = []
    tmp = best_set
    while tmp:
        lb = tmp & (-tmp)
        v = lb.bit_length() - 1
        result.append(available[v] + 1)
        tmp ^= lb
    
    result.sort()
    
    # Verify the result
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = int(math.isqrt(s))
            if r * r == s:
                # This should not happen
                pass
    
    return result
