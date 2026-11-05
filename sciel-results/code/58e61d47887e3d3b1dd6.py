"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    n = len(boxes)
    best_results = None
    best_vol = 0.0

    # Pre-compute unique orientations for each box to avoid repeated calculation
    box_orientations = []
    for w, h, d in boxes:
        seen = set()
        orients = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        box_orientations.append(orients)

    # Define different sorting strategies to diversify the search space
    # 1. Volume descending
    # 2. Max dimension descending
    # 3. Sum of dimensions descending
    # 4. Randomized weighted by volume (to break ties and explore different orderings)
    
    def run_strategy(sort_key_func, noise_level=0.0):
        """
        Runs a single greedy pass with a specific sorting strategy.
        Returns the total volume placed and the results array.
        """
        placed = [] # List of (x, y, z, w, h, d, orig_index)
        results = [None] * n
        total_vol = 0.0
        
        # Create list of (index, box) and sort
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=sort_key_func, reverse=True)
        
        # Candidate points: Start with origin
        # Using a list for efficiency in iteration, but we need to avoid duplicates.
        # A set is good for uniqueness but iteration order matters for determinism/noise.
        # We will maintain a list of unique points. For N=60, points are < 200.
        candidate_points = [(0, 0, 0)]
        
        for orig_idx, (w, h, d) in indexed_boxes:
            if time.time() > deadline:
                break
                
            best_pos = None
            best_score = float('inf')
            best_orient = None
            found_placement = False
            
            # To introduce noise, we can shuffle candidate points or add random jitter to scores
            # However, shuffling candidates changes the "first found" best fit.
            # A better noise mechanism: slightly perturb the score.
            
            for cx, cy, cz in candidate_points:
                if time.time() > deadline:
                    break
                    
                for ow, oh, od in box_orientations[orig_idx]:
                    # Check bounds
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Check overlap with all placed boxes
                    overlap = False
                    for px, py, pz, pw, ph, pd, _ in placed:
                        # Separation check:
                        # If any of these is true, they do NOT overlap
                        if not (cx + ow <= px or px + pw <= cx or 
                                cy + oh <= py or py + ph <= cy or 
                                cz + od <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Score: Minimize max extent (tightness)
                        # Add small random noise to break ties and explore different local minima
                        score = max(cx + ow, cy + oh, cz + od)
                        if noise_level > 0:
                            score += random.random() * noise_level
                        
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_orient = (ow, oh, od)
                            found_placement = True
            
            if found_placement:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                
                # Place the box
                placed.append((bx, by, bz, bw, bh, bd, orig_idx))
                results[orig_idx] = (0, bx, by, bz, bw, bh, bd)
                total_vol += bw * bh * bd
                
                # Generate new candidate positions
                new_candidates = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd)
                ]
                
                # Add to candidate list if unique and within bounds
                for nc in new_candidates:
                    if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                        if nc not in candidate_points:
                            candidate_points.append(nc)
            else:
                results[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any remaining results if deadline hit
        for i in range(n):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)
                
        return total_vol, results

    # Strategy 1: Volume Descending (Deterministic)
    vol, res1 = run_strategy(lambda x: x[1][0]*x[1][1]*x[1][2])
    if vol > best_vol:
        best_vol = vol
        best_results = res1

    # Strategy 2: Max Dimension Descending (Deterministic)
    if time.time() < deadline:
        vol, res2 = run_strategy(lambda x: max(x[1]))
        if vol > best_vol:
            best_vol = vol
            best_results = res2

    # Strategy 3: Sum of Dimensions Descending (Deterministic)
    if time.time() < deadline:
        vol, res3 = run_strategy(lambda x: sum(x[1]))
        if vol > best_vol:
            best_vol = vol
            best_results = res3

    # Strategy 4+: Randomized Multi-Start
    # Use remaining time to run multiple randomized passes with different noise levels
    # This helps escape local minima of the greedy heuristic.
    
    # We will run as many randomized passes as time allows.
    # Each pass uses a slightly different random seed (implicit via random module state)
    # and adds noise to the scoring function.
    
    # To ensure we don't just repeat the same noise, we vary the noise level.
    # Also, we can shuffle the order of candidate points evaluation to break symmetry.
    
    # Let's define a strategy that shuffles candidates
    def run_strategy_shuffled(sort_key_func, noise_level=0.0):
        placed = []
        results = [None] * n
        total_vol = 0.0
        
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=sort_key_func, reverse=True)
        
        candidate_points = [(0, 0, 0)]
        
        for orig_idx, (w, h, d) in indexed_boxes:
            if time.time() > deadline:
                break
                
            best_pos = None
            best_score = float('inf')
            best_orient = None
            found_placement = False
            
            # Shuffle candidates to break ties and explore different spots
            shuffled_candidates = candidate_points[:]
            random.shuffle(shuffled_candidates)
            
            for cx, cy, cz in shuffled_candidates:
                if time.time() > deadline:
                    break
                    
                for ow, oh, od in box_orientations[orig_idx]:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    overlap = False
                    for px, py, pz, pw, ph, pd, _ in placed:
                        if not (cx + ow <= px or px + pw <= cx or 
                                cy + oh <= py or py + ph <= cy or 
                                cz + od <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    
                    if not overlap:
                        score = max(cx + ow, cy + oh, cz + od)
                        if noise_level > 0:
                            score += random.random() * noise_level
                        
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_orient = (ow, oh, od)
                            found_placement = True
            
            if found_placement:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                
                placed.append((bx, by, bz, bw, bh, bd, orig_idx))
                results[orig_idx] = (0, bx, by, bz, bw, bh, bd)
                total_vol += bw * bh * bd
                
                new_candidates = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd)
                ]
                
                for nc in new_candidates:
                    if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                        if nc not in candidate_points:
                            candidate_points.append(nc)
            else:
                results[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        for i in range(n):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)
                
        return total_vol, results

    # Run randomized passes
    # We vary the noise level to explore different landscapes
    noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # We'll cycle through different sorting keys and noise levels
    sort_keys = [
        lambda x: x[1][0]*x[1][1]*x[1][2],
        lambda x: max(x[1]),
        lambda x: sum(x[1])
    ]
    
    i = 0
    while time.time() < deadline - 5.0: # Keep 5s buffer
        sort_key = sort_keys[i % len(sort_keys)]
        noise = noise_levels[i % len(noise_levels)]
        
        vol, res = run_strategy_shuffled(sort_key, noise)
        
        if vol > best_vol:
            best_vol = vol
            best_results = res
        
        i += 1

    # If no results were found (should not happen), return all -1
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_results
# EVOLVE-BLOCK-END
