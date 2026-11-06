"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    deadline = time.time() + 115.0  # Safety margin for 120s limit

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]

    # --- Packing Engine ---

    def is_valid_placement(placed, x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        
        # Check overlap with all placed boxes
        for (idx, px, py, pz, pw, ph, pd) in placed:
            # Overlap condition: NOT (x >= px+pw OR x+bw <= px OR y >= py+ph OR y+bh <= py OR z >= pz+pd OR z+bd <= pz)
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        return True

    def pack_with_order(order):
        placed = []
        candidates = [(0, 0, 0)]
        
        for idx in order:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            # Try all orientations for this box
            for (bw, bh, bd) in orient_cache[idx]:
                # Try all current candidates
                # Note: candidates list might contain points that are now invalid (inside new boxes)
                # but we check is_valid_placement which handles that.
                # We also prune candidates that fall inside the newly placed box after placement.
                
                for (cx, cy, cz) in candidates:
                    if is_valid_placement(placed, cx, cy, cz, bw, bh, bd):
                        # Score: prefer positions that are "lower" in the container
                        # Standard BFD (Bottom-Front-Left) heuristic: minimize z, then y, then x
                        # Using a weighted sum to prioritize z most, then y, then x
                        score = 10000 * cz + 100 * cy + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
                
                # Update candidates: add the 8 corners of the new box
                new_corners = [
                    (x, y, z), (x+bw, y, z), (x, y+bh, z), (x+bw, y+bh, z),
                    (x, y, z+bd), (x+bw, y, z+bd), (x, y+bh, z+bd), (x+bw, y+bh, z+bd)
                ]
                
                # Filter new corners: must be inside container and not inside any existing box (including the one just placed? 
                # A corner of the new box is on its boundary. It might be inside ANOTHER box if they are adjacent? 
                # No, if they are adjacent, the corner is on the boundary. If they overlap, placement was invalid.
                # So a corner of a validly placed box is never strictly inside another box.
                # However, it might be inside a box that was placed *before*? 
                # Wait, if I place Box A, then Box B next to it, a corner of B might be on the boundary of A.
                # The condition for a point P to be a valid candidate is that a box starting at P does not overlap.
                # If P is inside an existing box, any box starting at P will overlap.
                # So we should prune candidates that are strictly inside any existing box.
                
                for nc in new_corners:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H and 0 <= nc[2] < D:
                        # Check if this point is inside any existing box
                        inside = False
                        for (idx2, px, py, pz, pw, ph, pd) in placed:
                            if px <= nc[0] < px + pw and py <= nc[1] < py + ph and pz <= nc[2] < pz + pd:
                                inside = True
                                break
                        if not inside:
                            # Avoid duplicates
                            if nc not in candidates:
                                candidates.append(nc)
                
                # Prune candidates that are now inside the newly placed box
                # This is an optimization to keep the candidate list small.
                new_box = placed[-1]
                nx, ny, nz, nw, nh, nd = new_box[1], new_box[2], new_box[3], new_box[4], new_box[5], new_box[6]
                filtered_candidates = []
                for c in candidates:
                    if not (nx <= c[0] < nx + nw and ny <= c[1] < ny + nh and nz <= c[2] < nz + nd):
                        filtered_candidates.append(c)
                candidates = filtered_candidates

        return placed

    # --- Search Strategy ---
    
    indexed = list(range(n))
    
    # 1. Generate initial diverse orders
    # Volume descending
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    
    # Max dimension descending
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    
    # Sum of dimensions descending
    order_sum = sorted(indexed, key=lambda i: sum(boxes[i]), reverse=True)
    
    # Aspect ratio (cubic first)
    def aspect_ratio(i):
        w, h, d = boxes[i]
        m = min(w, h, d)
        if m == 0: return 0
        return max(w, h, d) / m
    order_aspect = sorted(indexed, key=aspect_ratio)
    
    # Random orders
    random_orders = []
    for _ in range(5):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        random_orders.append(rand_order)
        
    all_orders = [order_vol, order_max, order_sum, order_aspect] + random_orders
    
    best_placed = []
    best_vol = 0
    best_order = None
    
    start_time = time.time()
    
    # Phase 1: Greedy BFD with multiple orders
    for order in all_orders:
        if time.time() > deadline:
            break
        p = pack_with_order(order)
        vol = sum(w*h*d for (_, x, y, z, w, h, d) in p)
        if vol > best_vol:
            best_vol = vol
            best_placed = p
            best_order = order
            
        # If we hit 100% fill, stop
        if vol >= W*H*D:
            break

    # Phase 2: Local Search / Perturbation
    # We can try to swap boxes in the order or re-insert them.
    # Given time constraints, we might do a few random restarts with perturbed orders.
    
    if time.time() < deadline - 10:
        for _ in range(20): # Limited number of perturbations
            if time.time() > deadline:
                break
            
            # Perturb the best order
            perturbed_order = best_order[:]
            # Randomly swap a few elements
            for _ in range(5):
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
            
            p = pack_with_order(perturbed_order)
            vol = sum(w*h*d for (_, x, y, z, w, h, d) in p)
            if vol > best_vol:
                best_vol = vol
                best_placed = p
                best_order = perturbed_order

    # Construct final result
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    for (idx, x, y, z, w, h, d) in best_placed:
        placements[idx] = (0, x, y, z, w, h, d)
        
    return placements
# EVOLVE-BLOCK-END
