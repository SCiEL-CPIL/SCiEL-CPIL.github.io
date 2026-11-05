"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
TOTAL_VERTS = 1 << N  # 512

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a snake (induced path) in Q_9 using bit-parallel exclusion masks.
    
    Key idea: maintain an exclusion mask E (512-bit integer) where bit i is set
    iff vertex i cannot be the next vertex in the path. A vertex v can be next
    iff:
      - v is not already in the path (tracked via used_mask)
      - v is not adjacent to any non-consecutive vertex in the path
    
    We maintain E as the union of:
      - used_mask (vertices already in path)
      - For each vertex u in the path (except the last), all neighbors of u
    
    When we extend the path by adding vertex w as new last:
      - The old last vertex 'old_last' is no longer 'last', so its neighbors
        must now be added to the exclusion set (they become non-consecutive to w).
      - Actually wait: when we add w, the constraint is that w cannot be adjacent
        to any vertex in path[:-1]. The vertices in path[:-1] are fixed.
      - When we move on and add a new vertex after w, then w becomes part of 
        path[:-1] for that next step, so w's neighbors get added to exclusion.
    
    Let me re-think: 
    - At step k, we have path = [p0, p1, ..., pk]. We want to find p_{k+1} such that:
      - p_{k+1} not in {p0, ..., pk}
      - p_{k+1} not adjacent to p0, p1, ..., p_{k-1}  (non-consecutive vertices)
      - p_{k+1} IS adjacent to pk (consecutive)
    
    So the exclusion set at step k is:
      excluded = used_mask OR (union of neighbors of p0, p1, ..., p_{k-1})
    
    Note: pk's neighbors are NOT excluded (we WANT to pick one of them as p_{k+1}),
    but pk itself is in used_mask so it's excluded.
    
    When we add p_{k+1}:
      - New used_mask includes p_{k+1}
      - For the NEXT step, the exclusion will include neighbors of pk (since pk 
        becomes non-consecutive to p_{k+2})
      - So we add neighbors of pk to our running exclusion
    
    Let me maintain:
      - used_mask: bitmask of all vertices in path so far
      - non_last_exclusion: bitmask of all vertices that are neighbors of any 
        vertex in path except the last one. This is the "forbidden" set from 
        non-consecutive adjacency.
      - total_excluded = used_mask | non_last_exclusion
    
    When we extend by w:
      - old_last = path[-1]
      - Before adding w, we check: w must be adjacent to old_last, and 
        w not in total_excluded
      - After adding w: 
        - used_mask |= (1 << w)
        - non_last_exclusion |= neighbor_mask(old_last)  # old_last is now non-last
        - path.append(w)
    """
    used_mask = 1  # vertex 0 is in path
    non_last_exclusion = 0  # no non-last vertices yet
    path = [0]
    
    def get_neighbor_mask(v):
        """Return 512-bit mask with bits set for all neighbors of v (distance 1)."""
        m = 0
        for b in range(N):
            m |= 1 << (v ^ (1 << b))
        return m
    
    # Precompute neighbor masks for all 512 vertices for speed
    neighbor_masks = [0] * TOTAL_VERTS
    for v in range(TOTAL_VERTS):
        m = 0
        for b in range(N):
            m |= 1 << (v ^ (1 << b))
        neighbor_masks[v] = m
    
    while True:
        cur = path[-1]
        total_excluded = used_mask | non_last_exclusion
        
        # Find a neighbor of cur that is not in total_excluded
        candidates = neighbor_masks[cur] & ~total_excluded
        
        if candidates == 0:
            return path
        
        # Pick the lowest-set bit as our next vertex
        # candidates is a 512-bit integer; find lowest set bit
        # Use: lowest_bit = candidates & (-candidates), then index = log2(lowest_bit)
        lowest_bit = candidates & (-candidates)
        # Compute index: we can use bit_length - 1
        w = lowest_bit.bit_length() - 1
        
        # Extend the path
        used_mask |= (1 << w)
        # Now cur becomes a non-last vertex, so add its neighbors to non_last_exclusion
        non_last_exclusion |= neighbor_masks[cur]
        path.append(w)
    
# EVOLVE-BLOCK-END
