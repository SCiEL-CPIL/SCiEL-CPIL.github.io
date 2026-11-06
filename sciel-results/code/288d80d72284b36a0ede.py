"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe within 150s
    
    # Pre-compute box volumes
    vols = [w * h * d for w, h, d in boxes]
    max_vol = max(vols)
    
    # Initialize with a good greedy solution (Best Fit Decreasing)
    # Sort indices by volume descending
    order = sorted(range(n), key=lambda i: -vols[i])
    
    # Greedy packing to get initial state
    # State representation: 
    # placement[i] = (bin_id, x, y, z)
    # bins is a list of lists of box indices
    bins = []
    placement = [None] * n
    
    def check_overlap(bin_idx, x, y, z, w, h, d, ignore_idx=-1):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any box in bin_idx"""
        if x + w > W or y + h > H or z + d > D:
            return False
        for j in bins[bin_idx]:
            if j == ignore_idx:
                continue
            bx, by, bz, bw, bh, bd = placement[j]
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return False
        return True

    def find_spot(bin_idx, w, h, d, ignore_idx=-1):
        """Find a valid position in bin_idx for box with dims w,h,d"""
        # Candidate points: (0,0,0) and corners of existing boxes
        candidates = [(0, 0, 0)]
        for j in bins[bin_idx]:
            if j == ignore_idx:
                continue
            bx, by, bz, bw, bh, bd = placement[j]
            # Top face
            candidates.append((bx, by, bz + bd))
            # Right face
            candidates.append((bx + bw, by, bz))
            # Front face
            candidates.append((bx, by + bh, bz))
            # Other corners might help
            candidates.append((bx + bw, by + bh, bz))
            candidates.append((bx + bw, by, bz + bd))
            candidates.append((bx, by + bh, bz + bd))
            candidates.append((bx + bw, by + bh, bz + bd))
        
        # Sort candidates to prefer lower Z, then Y, then X
        candidates.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        
        for cx, cy, cz in candidates:
            if check_overlap(bin_idx, cx, cy, cz, w, h, d, ignore_idx):
                return (cx, cy, cz)
        
        return None

    # 1. Initial Greedy Packing
    for i in order:
        w, h, d = boxes[i]
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        for b_idx in range(len(bins)):
            pos = find_spot(b_idx, w, h, d)
            if pos is not None:
                # Best Fit: minimize empty space? Or just first fit? 
                # Let's use a heuristic: prefer bins that are already "fuller" or where box fits "lower"
                # For now, simple Best Fit by Z coordinate (pack low)
                score = pos[2]
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        bins[best_bin].append(i)
        placement[i] = (best_bin, best_pos[0], best_pos[1], best_pos[2])

    def evaluate():
        """Calculate score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        occupied_bins = [b for b in bins if b]
        B = len(occupied_bins)
        if B == 0:
            return 0
        # Find last bin (largest index with boxes)
        last_bin_idx = max(i for i, b in enumerate(bins) if b)
        last_bin = bins[last_bin_idx]
        last_vol = sum(vols[j] for j in last_bin)
        container_vol = W * H * D
        if container_vol == 0:
            return B
        fraction = last_vol / container_vol
        return B + 0.1 * (1 - fraction)

    current_score = evaluate()
    best_score = current_score
    best_placement = [p.copy() for p in placement]
    best_bins = [b.copy() for b in bins]

    # 2. Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Pre-allocate some structures if needed, but dynamic lists are fine for n=40
    
    while time.time() < deadline - 2:
        if T < min_T:
            break
            
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Move a box to a random position in a random bin
            i = random.randint(0, n - 1)
            old_bin, ox, oy, oz = placement[i]
            w, h, d = boxes[i]
            
            # Try to move to a random bin
            target_bin = random.randint(0, len(bins) - 1)
            
            # Try random position
            # To make it efficient, try a few random spots or reuse find_spot logic
            # For SA, we need a valid position to accept the move, or we reject if invalid.
            # Actually, standard SA on discrete spaces usually requires the new state to be valid.
            # So we try to find a spot. If none found, reject.
            
            # Remove from current bin
            bins[old_bin].remove(i)
            
            # Try to place in target_bin
            pos = find_spot(target_bin, w, h, d)
            if pos is None:
                # Revert
                bins[old_bin].append(i)
                placement[i] = (old_bin, ox, oy, oz)
                T *= cooling_rate
                continue
            
            # Calculate new score
            new_score = evaluate()
            
            # Acceptance
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                placement[i] = (target_bin, pos[0], pos[1], pos[2])
                bins[target_bin].append(i)
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_placement = [p.copy() for p in placement]
                    best_bins = [b.copy() for b in bins]
            else:
                # Revert
                bins[old_bin].append(i)
                placement[i] = (old_bin, ox, oy, oz)
                
        elif move_type < 0.7:
            # Swap two boxes
            i, j = random.sample(range(n), 2)
            bin_i, xi, yi, zi = placement[i]
            bin_j, xj, yj, zj = placement[j]
            wi, hi, di = boxes[i]
            wj, hj, dj = boxes[j]
            
            # Remove both
            bins[bin_i].remove(i)
            bins[bin_j].remove(j)
            
            # Try to place i in bin_j and j in bin_i
            # Note: find_spot ignores the box being placed, but we removed them, so it's safe.
            # However, we need to check if they fit in each other's spots? 
            # No, we need to find ANY spot in the other bin.
            
            pos_i = find_spot(bin_j, wi, hi, di)
            pos_j = find_spot(bin_i, wj, hj, dj)
            
            if pos_i is None or pos_j is None:
                # Revert
                bins[bin_i].append(i)
                bins[bin_j].append(j)
                placement[i] = (bin_i, xi, yi, zi)
                placement[j] = (bin_j, xj, yj, zj)
                T *= cooling_rate
                continue
            
            new_score = evaluate()
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                placement[i] = (bin_j, pos_i[0], pos_i[1], pos_i[2])
                placement[j] = (bin_i, pos_j[0], pos_j[1], pos_j[2])
                bins[bin_j].append(i)
                bins[bin_i].append(j)
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_placement = [p.copy() for p in placement]
                    best_bins = [b.copy() for b in bins]
            else:
                # Revert
                bins[bin_i].append(i)
                bins[bin_j].append(j)
                placement[i] = (bin_i, xi, yi, zi)
                placement[j] = (bin_j, xj, yj, zj)
        else:
            # Re-pack a random subset (restart local area)
            # Pick 5 random boxes, remove them, re-pack them greedily
            subset = random.sample(range(n), min(5, n))
            old_states = {}
            for i in subset:
                b, x, y, z = placement[i]
                old_states[i] = (b, x, y, z)
                bins[b].remove(i)
                
            # Re-pack subset
            # Sort subset by volume desc for better packing
            subset.sort(key=lambda i: -vols[i])
            
            for i in subset:
                w, h, d = boxes[i]
                best_bin = -1
                best_pos = None
                best_score_local = float('inf')
                
                for b_idx in range(len(bins)):
                    pos = find_spot(b_idx, w, h, d)
                    if pos is not None:
                        score = pos[2] # Heuristic
                        if score < best_score_local:
                            best_score_local = score
                            best_bin = b_idx
                            best_pos = pos
                            
                if best_bin == -1:
                    # If no space found in existing bins, put it back where it was? 
                    # Or open new bin? 
                    # For simplicity in SA, if we can't place it, revert the whole move.
                    # Revert all
                    for k in subset:
                        b, x, y, z = old_states[k]
                        bins[b].append(k)
                        placement[k] = (b, x, y, z)
                    break # Break out of re-pack loop
                else:
                    bins[best_bin].append(i)
                    placement[i] = (best_bin, best_pos[0], best_pos[1], best_pos[2])
            else:
                # If we successfully re-packed all
                new_score = evaluate()
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_placement = [p.copy() for p in placement]
                        best_bins = [b.copy() for b in bins]
                else:
                    # Revert
                    for k in subset:
                        b, x, y, z = old_states[k]
                        bins[b].append(k)
                        placement[k] = (b, x, y, z)
                    
        T *= cooling_rate

    # Return best solution
    if best_placement is not None:
        result = []
        for i in range(n):
            b, x, y, z = best_placement[i]
            w, h, d = boxes[i]
            result.append((b, x, y, z, w, h, d))
        return result
    else:
        # Fallback to initial greedy
        result = []
        for i in range(n):
            b, x, y, z = placement[i]
            w, h, d = boxes[i]
            result.append((b, x, y, z, w, h, d))
        return result
# EVOLVE-BLOCK-END
