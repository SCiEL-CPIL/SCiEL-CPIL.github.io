import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending
    indexed_boxes = [(w * h * d, i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(reverse=True)
    
    placed = [] # List of (x, y, z, w, h, d)
    candidate_set = set()
    candidates = [(0, 0, 0)]
    candidate_set.add((0, 0, 0))
    results = [None] * len(boxes)
    
    MAX_CANDIDATES = 800
    
    def get_orientations(w, h, d):
        # Unique orientations, sorted by z-height ascending for layer packing
        orients = list(set(itertools.permutations([w, h, d])))
        orients.sort(key=lambda o: (o[2], o[1], o[0]))
        return orients
    
    def fits(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with placed boxes
        for px, py, pz, pw, ph, pd in placed:
            # No overlap if separated in any dimension
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def prune_candidates():
        """Remove dominated candidates: if A has x<=x', y<=y', z<=z' and at least one strict, keep A."""
        nonlocal candidates, candidate_set
        if len(candidates) <= MAX_CANDIDATES:
            return
        # Sort by (z, y, x) ascending
        candidates.sort(key=lambda c: (c[2], c[1], c[0]))
        kept = []
        kept_set = set()
        for c in candidates:
            dominated = False
            for kc in kept:
                if kc[0] <= c[0] and kc[1] <= c[1] and kc[2] <= c[2] and (kc[0], kc[1], kc[2]) != c:
                    dominated = True
                    break
            if not dominated:
                kept.append(c)
                kept_set.add(c)
            if len(kept) >= MAX_CANDIDATES:
                break
        candidates = kept
        candidate_set = kept_set
    
    def cost_function(x, y, z, w, h, d):
        """Cost: prioritize low z, then compact in y and x. Penalize gaps."""
        # Primary: minimize z to fill layers
        # Secondary: minimize y, then x to pack tightly
        # Also penalize if box is floating (not touching bottom or another box below)
        cost = z * 10000 + y * 100 + x
        return cost
    
    def generate_new_candidates(bx, by, bz, bw, bh, bd):
        """Generate new candidate positions including projections onto surfaces."""
        new_cands = [
            (bx + bw, by, bz),
            (bx, by + bh, bz),
            (bx, by, bz + bd),
            (bx + bw, by + bh, bz),
            (bx + bw, by, bz + bd),
            (bx, by + bh, bz + bd),
            (bx + bw, by + bh, bz + bd),
        ]
        
        # Add projection-based candidates to fill gaps
        # Project onto each face of the placed box
        for px, py, pz, pw, ph, pd in placed:
            # If placed box is to the left of new box, project right face
            if px + pw <= bx and abs(px + pw - bx) < 1e-9:
                pass  # Already covered by (bx, py, pz) type logic
            
            # Add positions at the corners of the placed box that face the new box
            # This helps with filling irregular gaps
            pass
        
        return new_cands
    
    for vol, idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
            
        best_placement = None
        best_cost = float('inf')
        
        orientations = get_orientations(w, h, d)
        
        # Prune candidates if too many
        if len(candidates) > MAX_CANDIDATES:
            prune_candidates()
        
        for cx, cy, cz in candidates:
            for ow, oh, od in orientations:
                if not fits(cx, cy, cz, ow, oh, od):
                    continue
                cost = cost_function(cx, cy, cz, ow, oh, od)
                if cost < best_cost:
                    best_cost = cost
                    best_placement = (cx, cy, cz, ow, oh, od)
        
        if best_placement:
            bx, by, bz, bw, bh, bd = best_placement
            placed.append((bx, by, bz, bw, bh, bd))
            results[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Generate new candidates: positions adjacent to the placed box
            new_cands = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd),
                (bx + bw, by + bh, bz),
                (bx + bw, by, bz + bd),
                (bx, by + bh, bz + bd),
                (bx + bw, by + bh, bz + bd),
            ]
            
            # Add more diverse candidates by projecting onto other boxes' surfaces
            for px, py, pz, pw, ph, pd in placed:
                # Generate candidates at the "shadow" edges
                # Right face of placed box
                if px + pw < W:
                    c1 = (px + pw, py, pz)
                    c2 = (px + pw, py + ph, pz)
                    c3 = (px + pw, py, pz + pd)
                    c4 = (px + pw, py + ph, pz + pd)
                    for c in [c1, c2, c3, c4]:
                        if c not in candidate_set:
                            new_cands.append(c)
                
                # Top face of placed box
                if pz + pd < D:
                    c5 = (px, py, pz + pd)
                    c6 = (px + pw, py, pz + pd)
                    c7 = (px, py + ph, pz + pd)
                    c8 = (px + pw, py + ph, pz + pd)
                    for c in [c5, c6, c7, c8]:
                        if c not in candidate_set:
                            new_cands.append(c)
                
                # Front face of placed box
                if py + ph < H:
                    c9 = (px, py + ph, pz)
                    c10 = (px + pw, py + ph, pz)
                    c11 = (px, py + ph, pz + pd)
                    c12 = (px + pw, py + ph, pz + pd)
                    for c in [c9, c10, c11, c12]:
                        if c not in candidate_set:
                            new_cands.append(c)
            
            for nc in new_cands:
                if nc not in candidate_set:
                    candidates.append(nc)
                    candidate_set.add(nc)
        else:
            results[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill remaining results with -1 if not processed due to timeout
    for i in range(len(boxes)):
        if results[i] is None:
            w, h, d = boxes[i]
            results[i] = (-1, 0, 0, 0, w, h, d)
            
    return results
