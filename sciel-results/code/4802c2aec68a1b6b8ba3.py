import time
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container, placed):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    # Check bounds
    if x + w > W or y + h > H or z + d > D:
        return False
    # Check overlap
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _greedy_pack(boxes, container, sort_key):
    W, H, D = container
    # Sort boxes based on the provided key
    sorted_boxes = sorted(boxes, key=sort_key, reverse=True)
    
    placed = []
    # Candidate points for placement
    # Start with origin
    candidates = [(0, 0, 0)]
    
    total_volume = 0
    
    for w, h, d in sorted_boxes:
        best_pos = None
        best_rot = None
        
        rots = _rotations(w, h, d)
        
        # Try all candidate points
        # We iterate through a copy of candidates because we might add new ones, 
        # but we only want to consider the current set for this box to avoid 
        # placing two boxes in the same "wave" incorrectly? 
        # Actually, standard extreme point method:
        # We pick the "best" position among all current candidates.
        
        # To find the "best" position, we usually prefer lowest x, then lowest y, then lowest z.
        # We iterate all candidates and all rotations to find the first valid one 
        # according to the Bottom-Left-Back order.
        
        # For efficiency, we can sort candidates by (x, y, z) and break early?
        # Yes, if we sort candidates, the first valid one is the BLB position.
        
        # However, the "best" position depends on the rotation.
        # A box might fit in a "higher" spot with a "smaller" rotation footprint?
        # No, BLB is about the position.
        # Let's just find ALL valid (pos, rot) pairs and pick the one with min x, then min y, then min z.
        
        valid_options = []
        
        for cx, cy, cz in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), (cx, cy, cz), container, placed):
                    # Calculate a score for this position to pick the "best"
                    # BLB: min x, then min y, then min z
                    valid_options.append((cx, cy, cz, rw, rh, rd))
        
        if not valid_options:
            continue # Box cannot be placed
        
        # Sort valid options by x, y, z to find the most "bottom-left-back"
        # This ensures we fill the bottom-left corner first
        valid_options.sort(key=lambda opt: (opt[0], opt[1], opt[2]))
        
        # Pick the first one
        bx, by, bz, bw, bh, bd = valid_options[0]
        
        # Place the box
        placed.append((bx, by, bz, bw, bh, bd))
        total_volume += bw * bh * bd
        
        # Generate new candidates
        new_candidates = [
            (bx + bw, by, bz),
            (bx, by + bh, bz),
            (bx, by, bz + bd)
        ]
        
        # Add to candidates if inside container
        for nc in new_candidates:
            if nc[0] <= W and nc[1] <= H and nc[2] <= D:
                candidates.append(nc)
                
        # Optional: Deduplicate candidates to save memory/time?
        # For N=60, duplicates are not a huge issue, but let's use a set for uniqueness.
        # Converting to set every time is O(N).
        # Let's just let them be a list and check duplicates in _fits? No, _fits checks placed boxes.
        # Duplicate candidates just mean we check the same position multiple times.
        # We can clean up candidates occasionally.
        if len(candidates) > 100:
            candidates = list(set(candidates))

    # Construct result
    # We need to map back to original box indices?
    # The task asks for placement per box.
    # The input `boxes` is a list. The output must be a list of same length.
    # My greedy loop iterates `sorted_boxes`. I need to keep track of original indices.
    
    # Let's refactor _greedy_pack to return the placements for the ORIGINAL order.
    
    return None # Placeholder, will rewrite below

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Define sorting strategies
    # 1. Volume desc
    # 2. Max dimension desc
    # 3. Surface area desc
    # 4. Min dimension asc (compactness)
    
    strategies = [
        lambda b: b[0]*b[1]*b[2],
        lambda b: max(b),
        lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]),
        lambda b: min(b)
    ]
    
    best_placements = None
    best_volume = 0
    
    # Time guard
    start_time = time.time()
    time_limit = 140 # Leave 10s margin
    
    for i, key_func in enumerate(strategies):
        if time.time() - start_time > time_limit:
            break
            
        # Sort boxes with indices
        indexed_boxes = [(boxes[j], j) for j in range(n)]
        indexed_boxes.sort(key=lambda x: key_func(x[0]), reverse=(i != 3)) # i==3 is min dim, sort asc
        
        placed_indices = {}
        placed_boxes = [] # (x, y, z, w, h, d)
        candidates = [(0, 0, 0)]
        total_vol = 0
        
        for box_dims, orig_idx in indexed_boxes:
            w, h, d = box_dims
            best_pos = None
            best_rot = None
            
            rots = _rotations(w, h, d)
            valid_options = []
            
            for cx, cy, cz in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), (cx, cy, cz), container, placed_boxes):
                        valid_options.append((cx, cy, cz, rw, rh, rd))
                        
            if not valid_options:
                continue
                
            valid_options.sort(key=lambda opt: (opt[0], opt[1], opt[2]))
            bx, by, bz, bw, bh, bd = valid_options[0]
            
            placed_indices[orig_idx] = (bx, by, bz, bw, bh, bd)
            placed_boxes.append((bx, by, bz, bw, bh, bd))
            total_vol += bw * bh * bd
            
            new_candidates = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd)
            ]
            for nc in new_candidates:
                if nc[0] <= W and nc[1] <= H and nc[2] <= D:
                    candidates.append(nc)
                    
            if len(candidates) > 200:
                candidates = list(set(candidates))
                
        if total_vol > best_volume:
            best_volume = total_vol
            # Store the placements for this strategy
            best_placements = placed_indices.copy()
            
    # Construct final output
    # If no boxes were placed, best_placements is None
    if best_placements is None:
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    result = []
    for i in range(n):
        if i in best_placements:
            x, y, z, w, h, d = best_placements[i]
            result.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
            
    return result
