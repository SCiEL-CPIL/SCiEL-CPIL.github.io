import time
import random
import sys

N = 9
NUM_VERTICES = 1 << N

# Precompute neighbor masks for all vertices
# nbr_masks[v] is a bitmask where bit i is 1 if v and i are neighbors
nbr_masks = [0] * NUM_VERTICES
for v in range(NUM_VERTICES):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    nbr_masks[v] = m

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Beam width: balance between exploration and memory/time
    BEAM_WIDTH = 100
    
    best_path = []
    
    def beam_search(seed):
        random.seed(seed)
        
        # State: (path_list, used_mask, disallowed_mask)
        # used_mask: bitmask of vertices in the path
        # disallowed_mask: bitmask of vertices that are neighbors of non-consecutive vertices in the path
        #               i.e., neighbors of path[0], path[1], ..., path[-2]
        
        # Initialize with vertex 0
        # path = [0]
        # used = 1 << 0
        # disallowed = 0 (no non-consecutive vertices yet, path[-2] doesn't exist)
        
        beam = [([0], 1, 0)]
        
        current_best_len = 1
        
        while beam and time.time() < deadline:
            next_beam = []
            
            for path, used, disallowed in beam:
                last_v = path[-1]
                
                # Candidates are neighbors of last_v that are not used and not disallowed
                cand_mask = nbr_masks[last_v] & ~used & ~disallowed
                
                if cand_mask == 0:
                    # Dead end
                    continue
                
                # Expand all candidates
                cm = cand_mask
                while cm:
                    lsb = cm & -cm
                    n = lsb.bit_length() - 1
                    
                    new_path = path + [n]
                    new_used = used | lsb
                    # The new disallowed set includes neighbors of the new 'last_v' (which was last_v in old state)
                    # Because last_v becomes non-consecutive to the vertex after n
                    new_disallowed = disallowed | nbr_masks[last_v]
                    
                    next_beam.append((new_path, new_used, new_disallowed))
                    cm ^= lsb
            
            if not next_beam:
                break
            
            # Score and prune
            # Heuristic: number of available moves from the new last vertex
            scored_beam = []
            for path, used, disallowed in next_beam:
                last_v = path[-1]
                avail_mask = nbr_masks[last_v] & ~used & ~disallowed
                score = avail_mask.bit_count()
                
                # Add small random noise to break ties and encourage diversity
                noise = random.random() * 0.5
                scored_beam.append((score + noise, path, used, disallowed))
            
            # Sort by score descending
            scored_beam.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top BEAM_WIDTH
            beam = [(p, u, d) for s, p, u, d in scored_beam[:BEAM_WIDTH]]
            
            # Update best length
            if len(beam[0][0]) > current_best_len:
                current_best_len = len(beam[0][0])
                
        # Return the longest path found in this seed
        if beam:
            # The beam contains paths of the same length (the last expanded level)
            # But some might have died earlier? No, we only keep alive states.
            # So all paths in beam have length current_best_len (if beam is not empty)
            # Actually, if beam is empty, we return the last non-empty state?
            # In the loop, if next_beam is empty, we break. The previous beam had the longest paths.
            # Let's just return the first path in the final beam if it exists, else the last known best.
            # To be safe, we track the best path explicitly.
            
            # Re-tracing:
            # If we broke because next_beam was empty, the current `beam` holds the longest paths.
            # If we broke because time ran out, `beam` holds the current state.
            return beam[0][0] if beam else []
            
        return []

    # Run multiple seeds
    # We can run many seeds if the beam search is fast.
    # With BEAM_WIDTH=100 and depth ~200, each step is ~100*9 ops.
    # 200 steps * 900 ops = 180,000 ops per seed.
    # Python can do ~10M simple ops/sec. So ~0.02s per seed.
    # 115s / 0.02s = ~5000 seeds possible.
    # Let's run as many as we can.
    
    seed = 0
    while time.time() < deadline:
        p = beam_search(seed)
        if len(p) > len(best_path):
            best_path = p
        seed += 1
        
        # If we found a very long path, we might stop early?
        # The theoretical max is 512. Current best is ~107 (1.0288 * 104).
        # If we hit 150+, we can continue.
        
    return best_path
