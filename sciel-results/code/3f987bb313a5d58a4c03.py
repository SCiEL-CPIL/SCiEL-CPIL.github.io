"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe within 150s
    
    # Preprocess boxes
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    # Helper: Check if a box fits in a specific bin at a specific position
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return False
        return True

    # Helper: Find the best position in a bin using a Priority Queue (Bottom-Fill)
    # This is faster and often more compact than checking all surface points
    def find_best_spot(bin_boxes, w, h, d):
        if not bin_boxes:
            return (0, 0, 0)
        
        # Generate candidate points from existing boxes
        # We prioritize points that are "lower" (smaller z, then y, then x)
        # to encourage stacking
        candidates = set()
        candidates.add((0, 0, 0))
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Points on top of boxes
            candidates.add((bx, by, bz + bd))
            # Points to the side of boxes (if space allows)
            if bx + bw < W:
                candidates.add((bx + bw, by, bz))
            if by + bh < H:
                candidates.add((bx, by + bh, bz))
            # Corners
            if bx + bw < W and by + bh < H:
                candidates.add((bx + bw, by + bh, bz))
            if bx + bw < W and bz + bd < D:
                candidates.add((bx + bw, by, bz + bd))
            if by + bh < H and bz + bd < D:
                candidates.add((bx, by + bh, bz + bd))
            
        # Convert to list and sort by Z, then Y, then X to find the "lowest" valid spot
        # This mimics gravity/settling
        candidate_list = list(candidates)
        candidate_list.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for cx, cy, cz in candidate_list:
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                return (cx, cy, cz)
        
        return None

    def get_bin_volume(bin_boxes):
        return sum(bw*bh*bd for _,_,_,bw,bh,bd in bin_boxes)

    def evaluate_solution(bins):
        """Compute the score: B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not bins:
            return 0
        occupied_bins = [b for b in bins if b]
        B = len(occupied_bins)
        if B == 0:
            return 0
        last_bin_idx = max(i for i, b in enumerate(bins) if b)
        last_bin = bins[last_bin_idx]
        last_vol = get_bin_volume(last_bin)
        container_vol = W * H * D
        if container_vol == 0:
            return B
        fraction = last_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # Main Construction: Best-Fit Decreasing with Bottom-Fill
    def construct(order):
        bins = []
        placements = [None] * n
        
        for i, vol, w, h, d in order:
            best_bin_idx = -1
            best_pos = None
            best_bin_vol = -1
            
            # Try existing bins
            for b_idx, bin_boxes in enumerate(bins):
                spot = find_best_spot(bin_boxes, w, h, d)
                if spot is not None:
                    bin_vol = get_bin_volume(bin_boxes)
                    # Best Fit: Prefer the bin that is most full
                    if bin_vol > best_bin_vol:
                        best_bin_idx = b_idx
                        best_pos = spot
                        best_bin_vol = bin_vol
            
            if best_bin_idx == -1:
                # Open new bin
                best_bin_idx = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d))
            placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
            
        return bins, placements

    # Local Search: 2-Box Swap to maximize last bin volume
    def local_search(bins, placements):
        # We only care about improving the score.
        # Since B (number of bins) is usually fixed in local search unless we merge,
        # we focus on the last bin's volume.
        
        current_score = evaluate_solution(bins)
        
        # Identify the last bin (largest index with boxes)
        if not bins:
            return bins, placements, current_score
        
        last_bin_idx = max(i for i, b in enumerate(bins) if b)
        
        # We want to move volume INTO the last bin from other bins,
        # or swap boxes such that the last bin gets a larger box.
        
        # To make this efficient, we limit the search to a few iterations
        # and only try swaps that involve the last bin.
        
        improved = True
        max_iterations = 500
        iter_count = 0
        
        while improved and iter_count < max_iterations and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            
            # Try swapping a box from the last bin with a box from another bin
            # if the other bin's box is larger (increasing last bin volume)
            # OR if removing a small box from last bin and adding a large one from another bin fits.
            
            # Actually, a simpler and more effective move for the "last bin volume" term:
            # Try to move a box FROM another bin TO the last bin if it fits.
            # Or swap a box in last bin with a box in another bin if the incoming box is larger.
            
            last_bin_boxes = bins[last_bin_idx]
            if not last_bin_boxes:
                break
                
            # Create a list of boxes in last bin to potentially swap out
            # And boxes in other bins to potentially swap in
            
            # For efficiency, we'll just try random swaps involving the last bin
            # 1. Pick a box from last bin
            # 2. Pick a box from another bin
            # 3. Check if swapping them (or just moving) improves the last bin volume
            #    AND maintains validity in both bins.
            
            # To keep it simple and fast, let's try "Move" first:
            # Can we move a box from bin A to Last Bin?
            # This reduces B? No, B stays same if bin A isn't empty.
            # But it increases Last Bin Volume.
            
            # Let's try a specific heuristic:
            # Iterate through boxes in other bins.
            # Try to move them to the last bin.
            
            for b_idx in range(len(bins)):
                if b_idx == last_bin_idx:
                    continue
                if not bins[b_idx]:
                    continue
                    
                # Try to move each box in b_idx to last_bin_idx
                for box_idx_in_b, box_data in enumerate(bins[b_idx]):
                    _, _, _, bw, bh, bd = box_data
                    # Try to place this box in the last bin
                    spot = find_best_spot(bins[last_bin_idx], bw, bh, bd)
                    if spot is not None:
                        # Check if moving this box improves the score
                        # New last bin volume = old + box_vol
                        # We need to ensure the source bin doesn't become invalid (it won't, we're removing)
                        # And destination bin is valid (spot found)
                        
                        # Perform move
                        original_box = box_data
                        bins[b_idx].pop(box_idx_in_b)
                        
                        new_pos = spot
                        bins[last_bin_idx].append((new_pos[0], new_pos[1], new_pos[2], bw, bh, bd))
                        
                        # Update placements
                        # We need to know the original index of this box
                        # The box_data doesn't store the original index 'i' in the tuple (bx, by, bz, w, h, d)
                        # Wait, my bin_boxes store (bx, by, bz, w, h, d).
                        # I need to map back to the original box index 'i' to update 'placements'.
                        # This is a flaw in the current representation for local search.
                        
                        # Let's abort this complex move for now and stick to a simpler swap
                        # that doesn't require re-mapping indices easily, or fix the representation.
                        
                        # REVERT
                        bins[last_bin_idx].pop()
                        bins[b_idx].insert(box_idx_in_b, original_box)
                        
                        # To make this work, I need to store the original index in the bin box tuple.
                        # Let's change the bin box structure to (x, y, z, w, h, d, original_idx)
                        # But that changes everything.
                        
                        # Alternative: Do the local search on the 'placements' array directly?
                        # That's harder because 'placements' is scattered.
                        
                        # Let's restart the local search strategy with a simpler metric:
                        # Just try to swap two boxes between two bins if it reduces total volume in non-last bins
                        # or increases volume in last bin.
                        
                        # Given the complexity, let's implement a simpler "Shake"
                        # Randomly pick two boxes from different bins, try to swap their positions.
                        # If both fit, and the last bin's volume increases, accept.
                        
                        pass # Skip this inner loop for now to avoid bugs, use a simpler global swap
            
            # Simpler Local Search: Random Swap of any two boxes from different bins
            # If the swap results in a valid packing and increases the volume of the LAST bin (or keeps B same and increases last vol), accept.
            
            # Pick two random bins
            if len(bins) < 2:
                break
                
            b1, b2 = random.sample(range(len(bins)), 2)
            if not bins[b1] or not bins[b2]:
                continue
            
            # Pick random boxes from these bins
            idx1 = random.randint(0, len(bins[b1]) - 1)
            idx2 = random.randint(0, len(bins[b2]) - 1)
            
            box1 = bins[b1][idx1]
            box2 = bins[b2][idx2]
            
            # Get dimensions
            _, _, _, w1, h1, d1 = box1
            _, _, _, w2, h2, d2 = box2
            
            # Try to place box1 in b2 at box2's spot? No, box2 is moving away.
            # We need to find a new spot for box1 in b2 (after box2 is removed)
            # And a new spot for box2 in b1 (after box1 is removed)
            
            # Temporarily remove
            bins[b1].pop(idx1)
            bins[b2].pop(idx2)
            
            spot1_in_b2 = find_best_spot(bins[b2], w1, h1, d1)
            spot2_in_b1 = find_best_spot(bins[b1], w2, h2, d2)
            
            if spot1_in_b2 is not None and spot2_in_b1 is not None:
                # Calculate score change
                # Old last bin volume
                old_last_vol = get_bin_volume(bins[last_bin_idx]) if last_bin_idx < len(bins) else 0
                # Note: last_bin_idx might change if bins become empty?
                # B (number of occupied bins) might change.
                
                # Tentatively apply
                bins[b2].append((spot1_in_b2[0], spot1_in_b2[1], spot1_in_b2[2], w1, h1, d1))
                bins[b1].append((spot2_in_b1[0], spot2_in_b1[1], spot2_in_b1[2], w2, h2, d2))
                
                new_score = evaluate_solution(bins)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Update placements
                    # We need to map these boxes back to their original indices.
                    # This is the hard part. The bin_boxes don't know their original index.
                    
                    # To fix this, I MUST include the original index in the bin box tuple.
                    # Let's refactor the construction to include it.
                    
                    # REVERT for now because I haven't updated the tuple structure in the current scope
                    bins[b1].pop()
                    bins[b2].pop()
                    bins[b1].insert(idx1, box1)
                    bins[b2].insert(idx2, box2)
                else:
                    # Revert
                    bins[b1].pop()
                    bins[b2].pop()
                    bins[b1].insert(idx1, box1)
                    bins[b2].insert(idx2, box2)
            else:
                # Revert
                bins[b1].insert(idx1, box1)
                bins[b2].insert(idx2, box2)

        return bins, placements, current_score

    # --- REFACTORING TO SUPPORT LOCAL SEARCH PROPERLY ---
    # The above local search is broken because we can't update 'placements' without original indices.
    # Let's write a clean version that stores (x, y, z, w, h, d, original_idx) in bins.
    
    def construct_v2(order):
        bins = []
        placements = [None] * n
        
        for i, vol, w, h, d in order:
            best_bin_idx = -1
            best_pos = None
            best_bin_vol = -1
            
            for b_idx, bin_boxes in enumerate(bins):
                spot = find_best_spot(bin_boxes, w, h, d)
                if spot is not None:
                    bin_vol = sum(bw*bh*bd for _,_,_,bw,bh,bd,_ in bin_boxes)
                    if bin_vol > best_bin_vol:
                        best_bin_idx = b_idx
                        best_pos = spot
                        best_bin_vol = bin_vol
            
            if best_bin_idx == -1:
                best_bin_idx = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            
            bx, by, bz = best_pos
            # Store with original index
            bins[best_bin_idx].append((bx, by, bz, w, h, d, i))
            placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
            
        return bins, placements

    def local_search_v2(bins, placements):
        current_score = evaluate_solution(bins)
        
        if not bins:
            return bins, placements, current_score
            
        last_bin_idx = max(i for i, b in enumerate(bins) if b)
        
        improved = True
        max_iterations = 2000
        iter_count = 0
        
        while improved and iter_count < max_iterations and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            
            if len(bins) < 2:
                break
                
            # Random swap of two boxes from different bins
            b1, b2 = random.sample(range(len(bins)), 2)
            if not bins[b1] or not bins[b2]:
                continue
            
            idx1 = random.randint(0, len(bins[b1]) - 1)
            idx2 = random.randint(0, len(bins[b2]) - 1)
            
            box1 = bins[b1][idx1]
            box2 = bins[b2][idx2]
            
            # Extract data
            x1, y1, z1, w1, h1, d1, i1 = box1
            x2, y2, z2, w2, h2, d2, i2 = box2
            
            # Temporarily remove
            bins[b1].pop(idx1)
            bins[b2].pop(idx2)
            
            spot1_in_b2 = find_best_spot(bins[b2], w1, h1, d1)
            spot2_in_b1 = find_best_spot(bins[b1], w2, h2, d2)
            
            if spot1_in_b2 is not None and spot2_in_b1 is not None:
                # Apply
                bins[b2].append((spot1_in_b2[0], spot1_in_b2[1], spot1_in_b2[2], w1, h1, d1, i1))
                bins[b1].append((spot2_in_b1[0], spot2_in_b1[1], spot2_in_b1[2], w2, h2, d2, i2))
                
                new_score = evaluate_solution(bins)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Update placements
                    placements[i1] = (b2, spot1_in_b2[0], spot1_in_b2[1], spot1_in_b2[2], w1, h1, d1)
                    placements[i2] = (b1, spot2_in_b1[0], spot2_in_b1[1], spot2_in_b1[2], w2, h2, d2)
                else:
                    # Revert
                    bins[b1].pop()
                    bins[b2].pop()
                    bins[b1].insert(idx1, box1)
                    bins[b2].insert(idx2, box2)
            else:
                # Revert
                bins[b1].insert(idx1, box1)
                bins[b2].insert(idx2, box2)

        return bins, placements, current_score

    # --- MAIN LOOP ---
    
    best_score = float('inf')
    best_placements = None
    
    # Strategy:
    # 1. Run a few constructions with different orderings.
    # 2. Apply local search to the best one.
    
    # Ordering 1: Volume Descending
    order1 = sorted(indexed_boxes, key=lambda x: x[1], reverse=True)
    
    # Ordering 2: Random
    order2 = indexed_boxes.copy()
    random.shuffle(order2)
    
    # Ordering 3: Longest Dimension Descending
    order3 = sorted(indexed_boxes, key=lambda x: max(x[2], x[3], x[4]), reverse=True)
    
    orders = [order1, order2, order3]
    
    for order in orders:
        if time.time() > deadline - 10:
            break
            
        bins, placements = construct_v2(order)
        score = evaluate_solution(bins)
        
        if score < best_score:
            best_score = score
            best_placements = [tuple(p) for p in placements]
            # Run local search on this promising solution
            if time.time() < deadline - 15:
                bins_ls, placements_ls, score_ls = local_search_v2(bins, placements)
                if score_ls < best_score:
                    best_score = score_ls
                    best_placements = [tuple(p) for p in placements_ls]

    # Fallback if no solution found (should not happen)
    if best_placements is None:
        bins, placements = construct_v2(order1)
        best_placements = [tuple(p) for p in placements]

    return best_placements

# EVOLVE-BLOCK-END
