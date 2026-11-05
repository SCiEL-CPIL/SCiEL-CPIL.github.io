"""Model-and-optimize 3D bin packing with deadline guard."""
import time
import random
import math
from typing import List, Tuple

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 115.0  # safety margin before 120s
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    container_vol = W * H * D
    
    # Sort boxes by volume descending
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)
    
    def find_position(existing_items, w, h, d, W, H, D):
        """
        Find a position (x, y, z) where box (w, h, d) fits without overlap.
        Uses a grid of candidate positions based on existing item corners.
        """
        candidates = [(0, 0, 0)]
        for (ex, ey, ez, ew, eh, ed) in existing_items:
            candidates.append((ex + ew, ey, ez))
            candidates.append((ex, ey + eh, ez))
            candidates.append((ex, ey, ez + ed))
            candidates.append((ex + ew, ey + eh, ez))
            candidates.append((ex + ew, ey, ez + ed))
            candidates.append((ex, ey + eh, ez + ed))
            candidates.append((ex + ew, ey + eh, ez + ed))
        
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        # Sort candidates by (z, y, x) to prefer lower positions (compactness)
        unique_candidates.sort(key=lambda c: (c[2], c[1], c[0]))
        
        for (cx, cy, cz) in unique_candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in existing_items:
                if (cx < ex + ew and ex < cx + w and
                    cy < ey + eh and ey < cy + h and
                    cz < ez + ed and ez < cz + d):
                    overlap = True
                    break
            if not overlap:
                return (cx, cy, cz)
        
        return None

    def try_place(placement_state, box_idx):
        """
        Try to place box_idx in existing bins first (best fit), then new bin.
        Returns (success, new_state, bin_id, pos)
        """
        w, h, d = boxes[box_idx]
        
        best_bin = None
        best_pos = None
        best_score = float('inf')
        
        for bin_id in sorted(placement_state.keys()):
            if time.time() > deadline:
                break
            pos = find_position(placement_state[bin_id], w, h, d, W, H, D)
            if pos is not None:
                # Score: prefer positions closer to origin
                score = pos[0] + pos[1] + pos[2]
                if score < best_score:
                    best_score = score
                    best_bin = bin_id
                    best_pos = pos
        
        if best_bin is not None:
            new_bin_list = placement_state[best_bin] + [(best_pos[0], best_pos[1], best_pos[2], w, h, d)]
            new_state = dict(placement_state)
            new_state[best_bin] = new_bin_list
            return True, new_state, best_bin, best_pos
        
        # Try new bin
        new_bin_id = max(placement_state.keys()) + 1 if placement_state else 0
        new_state = dict(placement_state)
        new_state[new_bin_id] = [(0, 0, 0, w, h, d)]
        return True, new_state, new_bin_id, (0, 0, 0)

    def greedy_pack(ordering):
        state = {}
        placements = [None] * n
        for box_idx in ordering:
            if time.time() > deadline:
                break
            success, state, bin_id, pos = try_place(state, box_idx)
            if success:
                placements[box_idx] = (bin_id, pos[0], pos[1], pos[2], boxes[box_idx][0], boxes[box_idx][1], boxes[box_idx][2])
        
        if all(p is not None for p in placements):
            return placements
        return None

    # Initial greedy pass
    initial_placements = greedy_pack(indices_sorted)
    
    if initial_placements is None:
        # Fallback
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements

    best_placements = initial_placements
    best_score = compute_score(best_placements, W, H, D)

    # Local Search: Try to empty the last bin
    # We represent the solution as a list of placements.
    # We need to reconstruct the state (bin -> items) to try moving items.
    
    def get_state_from_placements(placements):
        state = {}
        for p in placements:
            bin_id = p[0]
            if bin_id not in state:
                state[bin_id] = []
            # p is (bin, x, y, z, w, h, d)
            state[bin_id].append((p[1], p[2], p[3], p[4], p[5], p[6]))
        return state

    def try_move_box(state, box_idx, from_bin):
        """
        Try to move box_idx from from_bin to any other bin.
        Returns (success, new_state, to_bin)
        """
        w, h, d = boxes[box_idx]
        
        # Remove from source bin
        source_items = state[from_bin]
        # Find the item corresponding to box_idx in source_items
        # Note: state items are (x,y,z,w,h,d). We need to identify which one is box_idx.
        # This is tricky because we don't track box_idx in the state items.
        # We should track it or reconstruct carefully.
        
        # Let's modify the state representation to include box_idx?
        # No, let's just find the item by matching dimensions and position? 
        # That's fragile. 
        # Better: In the local search loop, we maintain a mapping of box_idx -> (bin, x, y, z).
        
        return False, state, None

    # Re-implementation of Local Search with explicit tracking
    # We will maintain a list of (box_idx, bin_id, x, y, z)
    
    current_placements = best_placements[:]
    
    # Build a helper structure for local search
    # box_to_placement: box_idx -> (bin_id, x, y, z)
    # bin_to_boxes: bin_id -> list of box_idx
    
    def build_structures(placements):
        box_to_placement = {}
        bin_to_boxes = {}
        for i, p in enumerate(placements):
            bin_id, x, y, z, w, h, d = p
            box_to_placement[i] = (bin_id, x, y, z)
            if bin_id not in bin_to_boxes:
                bin_to_boxes[bin_id] = []
            bin_to_boxes[bin_id].append(i)
        return box_to_placement, bin_to_boxes

    def get_bin_items(bin_to_boxes, bin_id, boxes):
        """Return list of (x,y,z,w,h,d) for a bin."""
        items = []
        for box_idx in bin_to_boxes.get(bin_id, []):
            w, h, d = boxes[box_idx]
            _, x, y, z = box_to_placement[box_idx] # Need box_to_placement
            # This function signature is getting messy. 
            # Let's pass box_to_placement as well.
            pass
        return items

    # Let's do a simpler local search:
    # 1. Identify last bin.
    # 2. For each box in last bin, try to remove it and re-insert into other bins.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        box_to_placement, bin_to_boxes = build_structures(current_placements)
        
        if not bin_to_boxes:
            break
            
        last_bin = max(bin_to_boxes.keys())
        boxes_in_last_bin = bin_to_boxes[last_bin]
        
        if not boxes_in_last_bin:
            break
            
        # Try to move each box in the last bin to another bin
        moved_any = False
        # We need to be careful: if we move box A out, bin items change.
        # We will try to move ONE box at a time, update state, and if successful, mark improved.
        
        # To make this efficient, we can try to move the *largest* box first? Or any?
        # Let's just iterate.
        
        for box_idx in boxes_in_last_bin:
            if time.time() > deadline:
                break
                
            # Get current position of box_idx
            cur_bin, cx, cy, cz = box_to_placement[box_idx]
            w, h, d = boxes[box_idx]
            
            # Remove box_idx from last_bin in our tracking structures
            # We will simulate the removal
            new_bin_to_boxes = {k: v[:] for k, v in bin_to_boxes.items()}
            new_box_to_placement = box_to_placement.copy()
            
            # Remove from last_bin
            if box_idx in new_bin_to_boxes[last_bin]:
                new_bin_to_boxes[last_bin].remove(box_idx)
            del new_box_to_placement[box_idx]
            
            # Try to place box_idx in other bins
            placed = False
            for target_bin in sorted(new_bin_to_boxes.keys()):
                if target_bin == last_bin:
                    continue
                    
                # Get items in target_bin
                items = []
                for b_idx in new_bin_to_boxes[target_bin]:
                    tw, th, td = boxes[b_idx]
                    _, tx, ty, tz = new_box_to_placement[b_idx]
                    items.append((tx, ty, tz, tw, th, td))
                
                pos = find_position(items, w, h, d, W, H, D)
                if pos is not None:
                    # Place it there
                    new_box_to_placement[box_idx] = (target_bin, pos[0], pos[1], pos[2])
                    if target_bin not in new_bin_to_boxes:
                        new_bin_to_boxes[target_bin] = []
                    new_bin_to_boxes[target_bin].append(box_idx)
                    placed = True
                    break
            
            if placed:
                # Update current_placements
                # We need to reconstruct the full placements list from new_box_to_placement
                # This is O(N) per move, which is fine for N=40.
                new_placements = [None] * n
                for b_idx, (b_id, x, y, z) in new_box_to_placement.items():
                    w, h, d = boxes[b_idx]
                    new_placements[b_idx] = (b_id, x, y, z, w, h, d)
                
                # Check if all are placed (should be, since we only moved one)
                if all(p is not None for p in new_placements):
                    new_score = compute_score(new_placements, W, H, D)
                    if new_score < best_score:
                        best_score = new_score
                        current_placements = new_placements
                        best_placements = new_placements
                        improved = True
                        moved_any = True
                        break # Restart loop with new state

    return best_placements

def compute_score(placements, W, H, D):
    """
    Score: mean B + 0.1 * (1 - last-bin occupied volume / container volume)
    """
    if not placements:
        return 0.0
    
    bins_used = set(p[0] for p in placements)
    B = len(bins_used)
    
    last_bin = max(bins_used)
    
    container_vol = W * H * D
    last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
    
    score = B + 0.1 * (1 - last_bin_vol / container_vol)
    return score
