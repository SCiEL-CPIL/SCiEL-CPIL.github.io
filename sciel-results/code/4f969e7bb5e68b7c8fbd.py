"""
3D Bin Packing: Simulated Annealing with Maximal Empty Rectangles (MER) representation.
Strategy:
1. Initialize a valid packing using a fast greedy heuristic (Best-Fit Decreasing Volume).
2. Use Simulated Annealing to refine the solution.
3. Moves: 
   - Relocate a box to a different position in the same bin.
   - Relocate a box to a different bin.
   - Swap positions of two boxes (if in same bin) or bin assignments (if compatible).
4. Representation: Maintain a list of placed boxes per bin. Overlap checks are O(N) per bin 
   (N is small, <=40), which is fast enough for SA iterations within the time limit.
5. Objective: Minimize B + 0.1 * (1 - last_bin_fill).
"""
import time
import random
import math

DEADLINE = time.time() + 145.0  # Safety margin

def _get_initial_packing(boxes, container):
    """
    Greedy Best-Fit Decreasing Volume (BFDV) initialization.
    Returns: list of bins, each bin is a list of (idx, x, y, z, w, h, d)
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort indices by volume descending
    idxs = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    bins = []
    
    for idx in idxs:
        w, h, d = boxes[idx]
        placed = False
        
        for bi in range(len(bins)):
            # Try to place in existing bin
            # Candidate positions: Corners defined by existing boxes
            candidates = [(0, 0, 0)]
            for (_, px, py, pz, pw, ph, pd) in bins[bi]:
                if px + pw < W: candidates.append((px + pw, py, pz))
                if py + ph < H: candidates.append((px, py + ph, pz))
                if pz + pd < D: candidates.append((px, py, pz + pd))
                # Also try combinations for better fit? Keep it simple for init.
                if px + pw < W and py + ph < H: candidates.append((px + pw, py + ph, pz))
                if px + pw < W and pz + pd < D: candidates.append((px + pw, py, pz + pd))
                if py + ph < H and pz + pd < D: candidates.append((px, py + ph, pz + pd))
                if px + pw < W and py + ph < H and pz + pd < D: candidates.append((px + pw, py + ph, pz + pd))
            
            # Remove duplicates
            candidates = list(set(candidates))
            
            best_pos = None
            best_score = float('inf')
            
            for (cx, cy, cz) in candidates:
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                # Check overlap
                valid = True
                for (_, px, py, pz, pw, ph, pd) in bins[bi]:
                    if not (cx + w <= px or px + pw <= cx or
                            cy + h <= py or py + ph <= cy or
                            cz + d <= pz or pz + pd <= cz):
                        valid = False
                        break
                if valid:
                    # Score: prefer lower z, then y, then x
                    s = cz * 10000 + cy * 100 + cx
                    if s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz)
            
            if best_pos:
                cx, cy, cz = best_pos
                bins[bi].append((idx, cx, cy, cz, w, h, d))
                placed = True
                break
        
        if not placed:
            # New bin
            bins.append([(idx, 0, 0, 0, w, h, d)])
            
    return bins

def _score(bins, container):
    """Calculate the official score."""
    W, H, D = container
    C = W * H * D
    B = len(bins)
    if B == 0:
        return 0.0
    last_bin = bins[-1]
    if not last_bin:
        return B
    last_vol = sum(w * h * d for (_, _, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    return B + 0.1 * (1.0 - fill_ratio)

def _is_valid_bin(bin_content, container):
    """Check if a bin's content has no overlaps and fits in container."""
    W, H, D = container
    for i in range(len(bin_content)):
        idx1, x1, y1, z1, w1, h1, d1 = bin_content[i]
        if x1 < 0 or y1 < 0 or z1 < 0:
            return False
        if x1 + w1 > W or y1 + h1 > H or z1 + d1 > D:
            return False
        for j in range(i+1, len(bin_content)):
            idx2, x2, y2, z2, w2, h2, d2 = bin_content[j]
            if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1):
                return False
    return True

def _get_candidate_positions(bin_content, container, box_idx):
    """
    Get candidate positions for a box in a specific bin.
    Uses anchors from other boxes.
    """
    W, H, D = container
    w, h, d = boxes[box_idx]
    
    candidates = [(0, 0, 0)]
    for (_, px, py, pz, pw, ph, pd) in bin_content:
        if px == box_idx: continue
        # Faces
        if px + pw < W: candidates.append((px + pw, py, pz))
        if py + ph < H: candidates.append((px, py + ph, pz))
        if pz + pd < D: candidates.append((px, py, pz + pd))
        
        # Corners (combinations)
        if px + pw < W and py + ph < H: candidates.append((px + pw, py + ph, pz))
        if px + pw < W and pz + pd < D: candidates.append((px + pw, py, pz + pd))
        if py + ph < H and pz + pd < D: candidates.append((px, py + ph, pz + pd))
        if px + pw < W and py + ph < H and pz + pd < D: candidates.append((px + pw, py + ph, pz + pd))
        
        # Also allow "sliding" along axes? No, anchors are usually sufficient for SA.
        
    # Remove duplicates and out of bounds
    valid_candidates = set()
    for (cx, cy, cz) in candidates:
        if 0 <= cx and cx + w <= W and 0 <= cy and cy + h <= H and 0 <= cz and cz + d <= D:
            valid_candidates.add((cx, cy, cz))
            
    return list(valid_candidates)

def _fits_in_bin(bin_content, container, box_idx, pos):
    """Check if box at box_idx fits at pos in bin_content without overlap."""
    W, H, D = container
    cx, cy, cz = pos
    w, h, d = boxes[box_idx]
    
    for (_, px, py, pz, pw, ph, pd) in bin_content:
        if px == box_idx: continue
        if not (cx + w <= px or px + pw <= cx or
                cy + h <= py or py + ph <= cy or
                cz + d <= pz or pz + pd <= cz):
            return False
    return True

def _sa_optimize(bins, boxes, container, rng):
    """
    Simulated Annealing optimization.
    """
    W, H, D = container
    n = len(boxes)
    
    # Deep copy current best
    current_bins = [list(b) for b in bins]
    current_score = _score(current_bins, container)
    
    best_bins = [list(b) for b in current_bins]
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    max_iter = 50000 # Cap iterations, time guard is primary
    
    # Precompute volumes
    vols = [w*h*d for w,h,d in boxes]
    
    while T > T_min and time.time() < DEADLINE:
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Relocate a box to a new position in the same bin
            if len(current_bins) == 0: break
            # Pick a random non-empty bin
            non_empty_bins = [bi for bi in range(len(current_bins)) if current_bins[bi]]
            if not non_empty_bins: break
            bi = rng.choice(non_empty_bins)
            # Pick a random box in that bin
            bi_box_idx_in_bin = rng.randint(0, len(current_bins[bi])-1)
            box_item = current_bins[bi][bi_box_idx_in_bin]
            box_idx = box_item[0]
            
            # Get candidates
            candidates = _get_candidate_positions(current_bins[bi], container, box_idx)
            if not candidates:
                T *= alpha
                continue
                
            new_pos = rng.choice(candidates)
            
            # Check validity
            if not _fits_in_bin(current_bins[bi], container, box_idx, new_pos):
                T *= alpha
                continue
                
            # Apply move
            current_bins[bi][bi_box_idx_in_bin] = (box_idx, new_pos[0], new_pos[1], new_pos[2], 
                                                   boxes[box_idx][0], boxes[box_idx][1], boxes[box_idx][2])
            
            new_score = _score(current_bins, container)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_bins = [list(b) for b in current_bins]
            else:
                # Revert
                current_bins[bi][bi_box_idx_in_bin] = box_item
                
        elif move_type < 0.9:
            # Move a box to a different bin
            if len(current_bins) < 2:
                T *= alpha
                continue
            non_empty_bins = [bi for bi in range(len(current_bins)) if current_bins[bi]]
            if len(non_empty_bins) < 2:
                T *= alpha
                continue
                
            src_bi = rng.choice(non_empty_bins)
            src_box_idx_in_bin = rng.randint(0, len(current_bins[src_bi])-1)
            box_item = current_bins[src_bi][src_box_idx_in_bin]
            box_idx = box_item[0]
            
            # Pick a target bin (could be empty or non-empty)
            dst_bi = rng.randint(0, len(current_bins)-1)
            if dst_bi == src_bi:
                T *= alpha
                continue
                
            # Remove from source
            current_bins[src_bi].pop(src_box_idx_in_bin)
            
            # Try to place in dest
            candidates = _get_candidate_positions(current_bins[dst_bi], container, box_idx)
            if not candidates:
                # Revert
                current_bins[src_bi].insert(src_box_idx_in_bin, box_item)
                T *= alpha
                continue
                
            # Try a few random candidates in dest
            rng.shuffle(candidates)
            placed = False
            for pos in candidates[:5]: # Limit trials
                if _fits_in_bin(current_bins[dst_bi], container, box_idx, pos):
                    current_bins[dst_bi].append((box_idx, pos[0], pos[1], pos[2], 
                                                 boxes[box_idx][0], boxes[box_idx][1], boxes[box_idx][2]))
                    placed = True
                    break
                    
            if not placed:
                # Revert
                current_bins[src_bi].insert(src_box_idx_in_bin, box_item)
                T *= alpha
                continue
                
            new_score = _score(current_bins, container)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_bins = [list(b) for b in current_bins]
            else:
                # Revert
                # Remove from dest
                # Find the index we just added
                last_idx = len(current_bins[dst_bi]) - 1
                current_bins[dst_bi].pop(last_idx)
                # Re-insert in src
                current_bins[src_bi].insert(src_box_idx_in_bin, box_item)
                
        else:
            # Swap positions of two boxes in the same bin
            non_empty_bins = [bi for bi in range(len(current_bins)) if len(current_bins[bi]) > 1]
            if not non_empty_bins:
                T *= alpha
                continue
            bi = rng.choice(non_empty_bins)
            i1, i2 = rng.sample(range(len(current_bins[bi])), 2)
            box1 = current_bins[bi][i1]
            box2 = current_bins[bi][i2]
            
            # Swap positions
            current_bins[bi][i1] = (box1[0], box2[1], box2[2], box2[3], box1[4], box1[5], box1[6])
            current_bins[bi][i2] = (box2[0], box1[1], box1[2], box1[3], box2[4], box2[5], box2[6])
            
            # Check validity
            if _is_valid_bin(current_bins[bi], container):
                new_score = _score(current_bins, container)
                delta = new_score - current_score
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_bins = [list(b) for b in current_bins]
                else:
                    # Revert
                    current_bins[bi][i1] = box1
                    current_bins[bi][i2] = box2
            else:
                # Revert
                current_bins[bi][i1] = box1
                current_bins[bi][i2] = box2
                
        T *= alpha
        
    return best_bins

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
        
    t_start = time.time()
    rng = random.Random(42)
    
    best_bins = None
    best_score = float('inf')
    
    # Multiple restarts with different initializations
    # 1. BFDV
    init1 = _get_initial_packing(boxes, container)
    best_bins = init1
    best_score = _score(init1, container)
    
    # 2. Random Order Greedy (a few times)
    for _ in range(3):
        if time.time() > DEADLINE - 5: break
        idxs = list(range(n))
        rng.shuffle(idxs)
        # Simple greedy with random order
        bins = []
        for idx in idxs:
            w, h, d = boxes[idx]
            placed = False
            for bi in range(len(bins)):
                # Try (0,0,0) and some random anchors? Just (0,0,0) for simplicity in random init
                # Actually, let's reuse _get_candidate_positions
                candidates = _get_candidate_positions(bins[bi], container, idx)
                if not candidates: continue
                rng.shuffle(candidates)
                for pos in candidates:
                    if _fits_in_bin(bins[bi], container, idx, pos):
                        bins[bi].append((idx, pos[0], pos[1], pos[2], w, h, d))
                        placed = True
                        break
                if placed: break
            if not placed:
                bins.append([(idx, 0, 0, 0, w, h, d)])
                
        s = _score(bins, container)
        if s < best_score:
            best_score = s
            best_bins = bins
            
    # Run SA on the best initial solution
    if time.time() < DEADLINE - 10:
        optimized = _sa_optimize(best_bins, boxes, container, rng)
        if optimized:
            s = _score(optimized, container)
            if s < best_score:
                best_score = s
                best_bins = optimized
                
    # Build result
    placement_map = {}
    for bi, bin_content in enumerate(best_bins):
        for item in bin_content:
            idx, x, y, z, w, h, d = item
            placement_map[idx] = (bi, x, y, z, w, h, d)
            
    result = []
    for i in range(n):
        if i in placement_map:
            result.append(placement_map[i])
        else:
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result
