import time
import random
import math

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in ((w,h,d),(w,d,h),(h,w,d),(h,d,w),(d,w,h),(d,h,w)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _shelf_volume(order, rots, container):
    """
    Compute total volume of boxes that fit into the container using a 2D shelf
    packing (layers along D, shelves along H, items along W).
    order: list of box indices in placement order
    rots:  list of rotation index per box (same length as boxes)
    container: (W, H, D)
    """
    W, H, D = container
    n = len(order)
    # Precompute rotated dims per box
    # We'll compute on the fly using rots
    
    total_vol = 0
    z = 0
    while z < D:
        # Start a new layer
        y = 0
        while y < H:
            # Start a new shelf
            x = 0
            shelf_h = 0
            # Place boxes in order that fit in this shelf
            for idx in order:
                rw, rh, rd = rots[idx]
                if x + rw > W:
                    continue
                if y + rh > H:
                    continue
                # Check remaining depth in layer
                if z + rd > D:
                    continue
                # Place it
                x += rw
                shelf_h = max(shelf_h, rh)
                total_vol += rw * rh * rd
            y += shelf_h
            if shelf_h == 0:
                break
        z += max(rots[idx][2] for idx in order) if False else 0
        # Actually, layer depth is max rd of boxes in this layer
        # We need to track which boxes were placed in this layer
        # Let me restructure: we need to know layer depth after placing
        # Let me redo this more carefully
        break  # placeholder, will fix below
    
    # Let me rewrite the shelf packing properly
    total_vol = 0
    z = 0
    # We need to process boxes in order, assigning them to shelves/layers
    # A simpler shelf approach: process boxes in order, maintain current shelf
    # When shelf overflows W, move to next shelf (y += shelf_h)
    # When shelf overflows H, move to next layer (z += layer_d)
    
    # Actually, for a proper shelf method that respects the order:
    # We place boxes sequentially. Each box goes in the first shelf where it fits.
    # But that's not quite right for 3D. Let's use a simpler model:
    # Layers are determined by the max depth of boxes in that layer.
    # Within a layer, shelves are determined by max height of boxes on that shelf.
    # This is complex to do incrementally. Let's use a simpler approach:
    # Just try to pack in order using a shelf within a layer, and when we can't
    # fit more in a layer, start a new one.
    
    # Simplified: iterate through boxes in order. Maintain current layer's boxes.
    # When a box doesn't fit in current layer's shelves, start new layer.
    # Actually, let's just use a very simple shelf method:
    # Place boxes one by one. Keep track of current x, y, z.
    # If box doesn't fit in current position, try next shelf/layer.
    
    # Even simpler: use the "next-fit decreasing" style shelf packing
    # Process boxes in the given order. For each box, try to place it in the
    # current shelf. If it doesn't fit, start a new shelf. If the shelf height
    # exceeds H, start a new layer. If layer depth exceeds D, stop.
    
    # Let me just do a straightforward simulation:
    x, y, z = 0, 0, 0
    shelf_h = 0
    layer_d = 0
    total_vol = 0
    for idx in order:
        rw, rh, rd = rots[idx]
        # Try to place in current shelf
        if x + rw <= W and y + rh <= H and z + rd <= D:
            # Fits in current position
            x += rw
            shelf_h = max(shelf_h, rh)
            layer_d = max(layer_d, rd)
            total_vol += rw * rh * rd
        else:
            # Try to move to next shelf in same layer
            x = 0
            y += shelf_h
            shelf_h = rh
            if y + rh <= H and z + rd <= D:
                x = rw
                layer_d = max(layer_d, rd)
                total_vol += rw * rh * rd
            else:
                # Try to move to next layer
                x = 0
                y = 0
                shelf_h = rh
                z += layer_d
                if y + rh <= H and z + rd <= D:
                    x = rw
                    layer_d = rd
                    total_vol += rw * rh * rd
                # else: box doesn't fit, skip it (not placed)
    return total_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Precompute rotations for each box
    all_rots = [_rotations(b[0], b[1], b[2]) for b in boxes]
    
    # Best solution tracking
    best_vol = -1
    best_order = list(range(n))
    best_rots = [0] * n  # rotation index per box
    
    def evaluate(order, rot_choices):
        """Evaluate a permutation + rotation choice using shelf packing."""
        # Build rots array
        rots = [all_rots[i][rot_choices[i] % len(all_rots[i])] for i in range(n)]
        return _shelf_volume(order, rots, container)
    
    def random_init():
        """Generate a random initial solution."""
        order = list(range(n))
        random.shuffle(order)
        rot_choices = [random.randrange(len(all_rots[i])) for i in range(n)]
        return order, rot_choices
    
    # Simulated Annealing
    def simulated_annealing(order, rot_choices, t_start, t_end, n_steps, deadline):
        """Run SA starting from given solution."""
        current_vol = evaluate(order, rot_choices)
        best_local_vol = current_vol
        best_local_order = order[:]
        best_local_rots = rot_choices[:]
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            
            # Choose a move: swap two boxes in order, or change rotation of one box
            move_type = random.random()
            if move_type < 0.7:
                # Swap two positions in order
                i, j = random.sample(range(n), 2)
                order[i], order[j] = order[j], order[i]
            else:
                # Change rotation of one box
                i = random.randrange(n)
                rot_choices[i] = (rot_choices[i] + 1) % len(all_rots[i])
            
            new_vol = evaluate(order, rot_choices)
            delta = new_vol - current_vol
            
            if delta >= 0 or random.random() < math.exp(delta / max(t_start * 0.001, 1e-10)):
                current_vol = new_vol
                if new_vol > best_local_vol:
                    best_local_vol = new_vol
                    best_local_order = order[:]
                    best_local_rots = rot_choices[:]
            else:
                # Revert
                if move_type < 0.7:
                    order[i], order[j] = order[j], order[i]
                else:
                    rot_choices[i] = (rot_choices[i] - 1) % len(all_rots[i])
            
            # Cool down
            t_start *= 0.9995
            
        return best_local_vol, best_local_order, best_local_rots
    
    # Run multiple SA restarts
    n_restarts = 0
    while time.time() < deadline:
        order, rot_choices = random_init()
        vol, order, rot_choices = simulated_annealing(
            order, rot_choices,
            t_start=100.0,
            t_end=1.0,
            n_steps=2000,
            deadline=deadline
        )
        n_restarts += 1
        
        if vol > best_vol:
            best_vol = vol
            best_order = order[:]
            best_rots = rot_choices[:]
        
        # Early stop if perfect
        if best_vol >= container_volume:
            break
    
    # Convert best solution to placements
    placements = [None] * n
    x, y, z = 0, 0, 0
    shelf_h = 0
    layer_d = 0
    
    for idx in best_order:
        rw, rh, rd = all_rots[idx][best_rots[idx] % len(all_rots[idx])]
        if x + rw <= W and y + rh <= H and z + rd <= D:
            x += rw
            shelf_h = max(shelf_h, rh)
            layer_d = max(layer_d, rd)
            placements[idx] = (0, x - rw, y, z, rw, rh, rd)
        else:
            x = 0
            y += shelf_h
            shelf_h = rh
            if y + rh <= H and z + rd <= D:
                x = rw
                layer_d = max(layer_d, rd)
                placements[idx] = (0, 0, y - rh, z, rw, rh, rd)
            else:
                x = 0
                y = 0
                shelf_h = rh
                z += layer_d
                if y + rh <= H and z + rd <= D:
                    x = rw
                    layer_d = rd
                    placements[idx] = (0, 0, 0, z - rd, rw, rh, rd)
                else:
                    placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
    
    # Fill in any None placements
    for i in range(n):
        if placements[i] is None:
            placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return placements
