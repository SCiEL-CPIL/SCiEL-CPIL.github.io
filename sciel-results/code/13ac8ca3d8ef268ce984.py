"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 150s limit, keep 10s margin for safety/overhead
    deadline = start_time + 135.0
    
    # Preprocess: store box dimensions
    box_data = [(w, h, d) for w, h, d in boxes]
    
    # Initial result: all boxes left out
    final_placements = [(-1, 0, 0, 0, box_data[i][0], box_data[i][1], box_data[i][2]) for i in range(n)]
    
    def get_rotations(w, h, d):
        # Generate unique rotations
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)
    
    # Pre-compute all possible rotations for each box
    all_rotations = [get_rotations(w, h, d) for w, h, d in box_data]
    
    def pack_with_order(order, rotations_choice):
        """
        Packs boxes in the given order with the given rotation choices.
        Uses a fast Shelf-like heuristic for speed within SA.
        Returns (filled_volume, placements_list)
        """
        placements = [None] * n
        filled_vol = 0.0
        
        # State for shelf packing:
        # We maintain a list of "shelves" (y_level, current_x, max_h)
        # But standard shelf packing is 2D. For 3D, we can use "layers" in Z.
        # Let's use a simpler "Block" approach or just a fast BFD-like.
        # Given the speed requirement, let's try a simple "First Fit Decreasing Height" with shelves.
        
        # Actually, a very fast heuristic is "Greedy Shelf" where shelves are horizontal planes.
        # However, 3D is tricky. Let's use a simple "Corner" search but limited.
        
        # To be safe and fast, let's use a simple 2D projection per Z-layer?
        # Or just a very simple greedy:
        # Sort boxes by height descending? No, order is fixed by SA.
        
        # Let's use a simple "Best Fit" with a small set of candidate positions.
        # Candidates: (0,0,0) and corners of placed boxes.
        
        placed_boxes = [] # list of (x, y, z, w, h, d)
        
        for idx in order:
            rw, rh, rd = rotations_choice[idx]
            
            # Generate candidates
            candidates = [(0, 0, 0)]
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                # Faces
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
                # Corners (optional, might be too many)
                # candidates.append((px + pw, py + ph, pz))
                # candidates.append((px + pw, py, pz + pd))
                # candidates.append((px, py + ph, pz + pd))
                # candidates.append((px + pw, py + ph, pz + pd))
                
            best_pos = None
            best_score = float('inf')
            
            for cx, cy, cz in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                
                # Check overlap
                overlap = False
                for (px, py, pz, pw, ph, pd) in placed_boxes:
                    if not (cx + rw <= px or px + pw <= cx or
                            cy + rh <= py or py + ph <= cy or
                            cz + rd <= pz or pz + pd <= cz):
                        overlap = True
                        break
                
                if not overlap:
                    # Score: prefer lower z, then lower y, then lower x
                    score = cz * W * H + cy * W + cx
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
            
            if best_pos is not None:
                x, y, z = best_pos
                placements[idx] = (0, x, y, z, rw, rh, rd)
                placed_boxes.append((x, y, z, rw, rh, rd))
                filled_vol += rw * rh * rd
            else:
                placements[idx] = (-1, 0, 0, 0, box_data[idx][0], box_data[idx][1], box_data[idx][2])
                
        return filled_vol, placements

    def simulate_annealing():
        """
        Runs SA to optimize order and rotations.
        """
        # Initial order: random
        order = list(range(n))
        random.shuffle(order)
        
        # Initial rotations: random
        rotations_choice = [all_rotations[i][random.randint(0, len(all_rotations[i])-1)] for i in range(n)]
        
        # Evaluate initial
        curr_vol, curr_placements = pack_with_order(order, rotations_choice)
        best_vol = curr_vol
        best_placements = curr_placements
        best_order = order[:]
        best_rotations = rotations_choice[:]
        
        T = 1000.0
        cooling = 0.995
        steps = 0
        
        while T > 1.0 and time.time() < deadline - 5.0:
            # Perturb: swap two boxes in order
            i, j = random.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]
            
            # Or change rotation
            if random.random() < 0.5:
                idx = random.randint(0, n-1)
                rotations_choice[idx] = all_rotations[idx][random.randint(0, len(all_rotations[idx])-1)]
            
            new_vol, new_placements = pack_with_order(order, rotations_choice)
            
            delta = new_vol - curr_vol
            if delta > 0 or random.random() < math.exp(delta / T):
                curr_vol = new_vol
                curr_placements = new_placements
                if new_vol > best_vol:
                    best_vol = new_vol
                    best_placements = new_placements
                    best_order = order[:]
                    best_rotations = rotations_choice[:]
            
            T *= cooling
            steps += 1
            
        return best_vol, best_placements

    # Run SA
    best_vol, best_placements = simulate_annealing()
    
    # If SA didn't finish or found nothing, fallback to simple greedy
    if best_vol == 0 or best_placements is None:
        # Simple greedy fallback
        placements = [None] * n
        for i in range(n):
            placements[i] = (-1, 0, 0, 0, box_data[i][0], box_data[i][1], box_data[i][2])
        return placements
    
    return best_placements
# EVOLVE-BLOCK-END
