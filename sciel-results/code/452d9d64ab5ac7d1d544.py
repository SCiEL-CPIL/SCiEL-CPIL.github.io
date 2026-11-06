"""
3D Container Loading with Iterated Local Search
Combines greedy heuristics with ILS to explore box ordering space.
"""
import time
import itertools
import random
import sys

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Edge cases
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]

    start_time = time.time()
    # Safety margin: 150s kill limit, aim for 140s max execution
    deadline = start_time + 138.0 

    # Precompute orientations for each box
    # orient[i] is a list of unique (w, h, d) tuples for box i
    orient = []
    for w, h, d in boxes:
        opts = set()
        for p in itertools.permutations((w, h, d)):
            opts.add(p)
        orient.append(list(opts))
    
    # Precompute box volumes
    box_vol = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    # Helper: Check if a box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap
    # placed: list of (x, y, z, w, h, d)
    def fits(x, y, z, w, h, d, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            # Separation test
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def get_contact_score(x, y, z, w, h, d, placed):
        """
        Calculate a score based on how many faces touch existing boxes.
        Higher is better (more contact = tighter packing).
        """
        score = 0
        for (px, py, pz, pw, ph, pd) in placed:
            # Check if faces are touching (distance is 0 and they overlap in the other 2 dims)
            # X-face touch
            if x == px + pw or px == x + w:
                if not (y + h <= py or py + ph <= y) and not (z + d <= pz or pz + pd <= z):
                    score += 1
            # Y-face touch
            if y == py + ph or py == y + h:
                if not (x + w <= px or px + pw <= x) and not (z + d <= pz or pz + pd <= z):
                    score += 1
            # Z-face touch
            if z == pz + pd or pz == z + d:
                if not (x + w <= px or px + pw <= x) and not (y + h <= py or py + ph <= y):
                    score += 1
        return score

    def try_pack(order, time_limit):
        """
        Packs boxes according to the given order.
        Returns (filled_volume, placements_dict)
        placements_dict: {box_idx: (bin, x, y, z, w, h, d)}
        """
        placed = []  # List of (x, y, z, w, h, d)
        placements = {}
        filled_vol = 0
        
        # Anchors: potential positions for the next box
        # We start with (0,0,0) and expand as we place boxes
        anchors = [(0, 0, 0)]
        anchor_set = {(0, 0, 0)}
        
        for idx in order:
            if time.time() > time_limit:
                break
                
            w, h, d = boxes[idx]
            best_pos = None
            best_key = None
            # Key will be a tuple: (z, y, x, -contact_score)
            # We want min z, min y, min x, max contact.
            # So we minimize (z, y, x) and maximize contact.
            # We can combine into a single minimization key:
            # key = (z * 1e9 + y * 1e6 + x * 1e3 - contact * 100) 
            # Or just compare tuples: (z, y, x, -contact)
            
            found_origin = False
            
            for (ow, oh, od) in orient[idx]:
                if ow > W or oh > H or od > D:
                    continue
                
                for (ax, ay, az) in anchors:
                    if ax + ow > W or ay + oh > H or az + od > D:
                        continue
                    
                    # Quick reject if we already have a position at (0,0,0)
                    # (0,0,0) is the absolute best position.
                    if found_origin:
                        break
                    
                    if fits(ax, ay, az, ow, oh, od, placed):
                        # Calculate contact score
                        contact = get_contact_score(ax, ay, az, ow, oh, od, placed)
                        # We want to minimize z, then y, then x.
                        # And maximize contact.
                        # Let's use a tuple key for comparison.
                        # Python compares tuples lexicographically.
                        # We want:
                        # 1. Lower z
                        # 2. Lower y
                        # 3. Lower x
                        # 4. Higher contact
                        # So key = (az, ay, ax, -contact)
                        key = (az, ay, ax, -contact)
                        
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (ax, ay, az, ow, oh, od)
                            if az == 0 and ay == 0 and ax == 0:
                                found_origin = True
                                # If we found origin, we can't do better on position.
                                # But maybe another orientation has more contact at origin?
                                # Unlikely to matter much, but let's check other orientations
                                # if we have time. For now, break out of anchors loop.
                                break
                
                if found_origin:
                    # If we found a box at origin, it's the best possible position.
                    # We still need to check if another orientation at origin has higher contact?
                    # Usually, just taking the first fit at origin is fine.
                    # But to be safe, we can break out of orientations loop too if we are confident.
                    # Let's stick to breaking anchors, and let orientations loop finish or break if found_origin.
                    # Actually, if found_origin is True, we break the anchor loop.
                    # We should probably break the orientation loop too if we are at (0,0,0)
                    # because no other position can be better than (0,0,0).
                    # The only question is orientation.
                    # Let's just break the orientation loop as well for speed, assuming (0,0,0) is best.
                    break

            if best_pos is not None:
                bx, by, bz, bw, bh, bd = best_pos
                placed.append((bx, by, bz, bw, bh, bd))
                placements[idx] = (0, bx, by, bz, bw, bh, bd)
                filled_vol += bw * bh * bd
                
                # Generate new anchors
                # Faces:
                # Top (z+bd)
                if bz + bd <= D:
                    new_a = (bx, by, bz + bd)
                    if new_a not in anchor_set:
                        anchor_set.add(new_a)
                        anchors.append(new_a)
                # Right (x+bw)
                if bx + bw <= W:
                    new_a = (bx + bw, by, bz)
                    if new_a not in anchor_set:
                        anchor_set.add(new_a)
                        anchors.append(new_a)
                # Front (y+bh)
                if by + bh <= H:
                    new_a = (bx, by + bh, bz)
                    if new_a not in anchor_set:
                        anchor_set.add(new_a)
                        anchors.append(new_a)
                # Corners (combinations)
                if bx + bw <= W and by + bh <= H:
                    new_a = (bx + bw, by + bh, bz)
                    if new_a not in anchor_set:
                        anchor_set.add(new_a)
                        anchors.append(new_a)
                if bx + bw <= W and bz + bd <= D:
                    new_a = (bx + bw, by, bz + bd)
                    if new_a not in anchor_set:
                        anchor_set.add(new_a)
                        anchors.append(new_a)
                if by + bh <= H and bz + bd <= D:
                    new_a = (bx, by + bh, bz + bd)
                    if new_a not in anchor_set:
                        anchor_set.add(new_a)
                        anchors.append(new_a)
                if bx + bw <= W and by + bh <= H and bz + bd <= D:
                    new_a = (bx + bw, by + bh, bz + bd)
                    if new_a not in anchor_set:
                        anchor_set.add(new_a)
                        anchors.append(new_a)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        return filled_vol, placements

    # Define sorting keys for initial heuristics
    def key_volume(i): return -box_vol[i]
    def key_max_dim(i): return -max(boxes[i])
    def key_min_dim(i): return -min(boxes[i])
    def key_surf(i):
        a,b,c = boxes[i]
        return -(2*(a*b + b*c + a*c))
    def key_compact(i):
        a,b,c = boxes[i]
        m = max(a,b,c)
        return -(a*b*c / m) if m > 0 else 0
    def key_sum(i):
        return -sum(boxes[i])
        
    heuristics = [key_volume, key_max_dim, key_min_dim, key_surf, key_compact, key_sum]
    
    best_filled = 0
    best_placements = None
    best_order = None
    
    # Phase 1: Try initial heuristics
    for h_func in heuristics:
        if time.time() > deadline:
            break
        order = sorted(range(n), key=h_func)
        filled, placements = try_pack(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placements = placements
            best_order = list(order)
            
    # Phase 2: Iterated Local Search
    # We perturb the best order found and re-pack.
    rng = random.Random(42)
    
    # If no box was placed, we can't perturb.
    if best_order is None:
        # Fallback
        if best_placements is None:
            best_placements = [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
        return best_placements

    # Convert best_placements to a list format for return
    # The function expects a list of tuples.
    # best_placements is a dict {idx: tuple}.
    
    # ILS Loop
    # We will run as many iterations as possible until deadline.
    # To avoid getting stuck in a local optimum, we occasionally reset to a random heuristic or accept worse solutions with low probability (Simulated Annealing style).
    # Given the short time, simple hill climbing on the order permutation is effective.
    
    current_order = list(best_order)
    current_filled = best_filled
    current_placements = best_placements
    
    # We need to be careful with time. try_pack takes some time.
    # Let's estimate time per pack.
    # If we have 100s left, and pack takes 0.05s, we can do 2000 packs.
    
    while time.time() < deadline - 2.0: # Keep 2s for final return
        # Perturb current_order
        # Strategies:
        # 1. Swap two elements
        # 2. Insert element at random position
        # 3. Reverse a segment
        
        r = rng.random()
        new_order = list(current_order)
        
        if r < 0.4:
            # Swap
            i, j = rng.randrange(n), rng.randrange(n)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif r < 0.7:
            # Insert
            i = rng.randrange(n)
            j = rng.randrange(n)
            elem = new_order.pop(i)
            new_order.insert(j, elem)
        else:
            # Reverse segment
            i, j = rng.randrange(n), rng.randrange(n)
            if i > j: i, j = j, i
            new_order[i:j+1] = reversed(new_order[i:j+1])
        
        filled, placements = try_pack(new_order, deadline)
        
        # Acceptance Criteria
        # If better, accept.
        # If worse, accept with small probability (to escape local optima)?
        # Or just strict hill climbing?
        # Strict hill climbing is safer for stability.
        if filled > current_filled:
            current_filled = filled
            current_placements = placements
            current_order = new_order
            
            # Update global best
            if current_filled > best_filled:
                best_filled = current_filled
                best_placements = current_placements
                best_order = current_order
        else:
            # If we are stuck for a long time, maybe try a larger perturbation or reset?
            # For now, just continue.
            pass

    # Finalize result
    # best_placements is a dict. We need a list.
    result = [None] * n
    for i in range(n):
        if i in best_placements:
            result[i] = best_placements[i]
        else:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
            
    return result

