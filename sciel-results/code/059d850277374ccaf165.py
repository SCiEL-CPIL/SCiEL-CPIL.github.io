"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe within 120s
    
    # Pre-compute volumes and sort indices
    volumes = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D
    
    # Sort boxes by volume descending for initial construction
    # We will work with a list of indices sorted by volume
    sorted_indices = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    def get_score(bins, last_bin_idx):
        """
        Calculate the score: mean B + 0.1 * (1 - last-bin occupied volume fraction)
        B is the number of occupied bins.
        The 'last bin' is defined by the largest label.
        """
        if not bins:
            return 0.0
        
        # Find the largest label
        max_label = max(bins.keys())
        B = len(bins)
        
        # Calculate occupied volume in the last bin
        last_bin_boxes = bins[max_label]
        last_vol = sum(w * h * d for _, _, _, w, h, d in last_bin_boxes)
        
        # Fraction of last bin occupied
        if container_vol == 0:
            frac = 0.0
        else:
            frac = last_vol / container_vol
            
        return B + 0.1 * (1 - frac)

    def can_place(bin_boxes, x, y, z, w, h, d):
        # Check bounds
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes in this bin
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return False
        return True

    def generate_extreme_points(bin_boxes, w, h, d):
        """
        Generate candidate positions for a box of size (w,h,d) in a bin.
        Uses the 'Extreme Point' heuristic: positions are defined by the
        intersection of planes defined by the faces of existing boxes.
        """
        # Initialize with origin
        points = set([(0, 0, 0)])
        
        # For each existing box, generate points on its surfaces
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Planes: 
            # x: 0, bx, bx+bw
            # y: 0, by, by+bh
            # z: 0, bz, bz+bd
            
            # We only need to check combinations that are "supported" by existing boxes
            # A common efficient subset is:
            # 1. On top of box: (bx, by, bz+bd)
            # 2. Beside in X: (bx+bw, by, bz)
            # 3. Beside in Y: (bx, by+bh, bz)
            # 4. Corner combinations often help fill gaps
            
            # To keep it fast but effective, we add points derived from the "corners" of existing boxes
            # that are valid anchors.
            
            # Simple set of candidates:
            candidates = [
                (bx, by, bz + bd),      # On top
                (bx + bw, by, bz),      # Right
                (bx, by + bh, bz),      # Back
                (bx + bw, by + bh, bz), # Top-Right-Back corner
                (bx + bw, by, bz + bd), # Right-Top
                (bx, by + bh, bz + bd), # Back-Top
                (bx + bw, by + bh, bz + bd) # Top-Right-Back-Top
            ]
            for p in candidates:
                points.add(p)
                
        # Also, it's often beneficial to try shifting a box as close to the origin as possible
        # in dimensions where it doesn't conflict, but the extreme points usually cover the
        # "tightest" fits. However, simple FFD often works by just trying to slide into gaps.
        
        # To improve "Best-Fit" quality, we should also consider positions that minimize
        # the "height" (z) primarily, then y, then x.
        # The extreme points generated above are the standard anchors.
        
        return list(points)

    def find_best_spot(bin_boxes, w, h, d):
        """
        Find the best position in a bin for a box.
        Returns (x, y, z) or None.
        'Best' is defined by minimizing the 'wasted space' or simply the position (z, y, x)
        to ensure compactness.
        """
        candidates = generate_extreme_points(bin_boxes, w, h, d)
        
        best_pos = None
        best_metric = float('inf')
        
        for cx, cy, cz in candidates:
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                # Metric: Prefer lower Z, then lower Y, then lower X.
                # This creates a compact, bottom-left-front packed bin.
                metric = (cz, cy, cx)
                if metric < best_metric:
                    best_metric = metric
                    best_pos = (cx, cy, cz)
                    
        return best_pos

    def construct_solution(indices_to_place, bins, bin_counter):
        """
        Place a list of box indices into the existing bins structure.
        Returns True if all placed, False if some could not be placed (should not happen if we open new bins).
        """
        for idx in indices_to_place:
            w, h, d = boxes[idx]
            placed = False
            
            # Try existing bins first, sorted by "fullness" or just iterate?
            # For Best-Fit, we might want to try all bins and pick the one with the "best" spot.
            # However, to save time, let's try bins in order of current volume (descending)
            # so we try to fill nearly-full bins first.
            
            # Get list of bin labels
            bin_labels = list(bins.keys())
            # Sort by volume descending
            bin_labels.sort(key=lambda l: sum(bw*bh*bd for _,_,_,bw,bh,bd in bins[l]), reverse=True)
            
            best_bin_label = None
            best_pos = None
            best_spot_metric = float('inf')
            
            for label in bin_labels:
                pos = find_best_spot(bins[label], w, h, d)
                if pos is not None:
                    # Compare spots across bins? 
                    # Simple heuristic: Prefer the bin that results in the lowest (z,y,x) placement.
                    # This tends to keep the "active" packing area compact.
                    metric = (pos[2], pos[1], pos[0])
                    if metric < best_spot_metric:
                        best_spot_metric = metric
                        best_bin_label = label
                        best_pos = pos
            
            if best_bin_label is None:
                # Open a new bin
                best_bin_label = bin_counter
                bins[best_bin_label] = []
                best_pos = (0, 0, 0)
                bin_counter += 1
                
            # Place the box
            bins[best_bin_label].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
        return bin_counter

    # --- Phase 1: Initial Construction ---
    bins = {}
    bin_counter = 0
    # Place all boxes in sorted order (largest first)
    current_indices = sorted_indices[:]
    bin_counter = construct_solution(current_indices, bins, bin_counter)
    
    current_score = get_score(bins, max(bins.keys()) if bins else 0)
    
    # --- Phase 2: Local Search (LNS) ---
    # Strategy:
    # 1. Pick a random subset of boxes to "remove" (perturbation).
    # 2. Re-pack the entire set of boxes (or just the removed ones + potentially others?)
    #    Re-packing ALL boxes is expensive but N=40 is small.
    #    Let's try re-packing the removed boxes into the current bins, and if that fails,
    #    we can try re-packing a larger set.
    #    
    #    Actually, a simpler and effective LNS for Bin Packing is:
    #    1. Remove K random boxes.
    #    2. Try to re-insert them one by one using Best-Fit.
    #    3. If successful, check if score improved.
    #    4. If not, revert.
    #    
    #    Even better: "Shift" heuristic.
    #    1. Pick a box from a non-last bin.
    #    2. Try to move it to the last bin or another bin to reduce B or improve last-bin fill.
    
    # Let's implement a "Re-insertion" LNS.
    # We maintain the current best solution.
    
    best_bins = {k: v[:] for k, v in bins.items()}
    best_score = current_score
    
    # Random seed for reproducibility in local search if needed, but random is fine
    random.seed(42)
    
    # Time budget for LNS: use up to 100s
    lns_start = time.time()
    lns_deadline = lns_start + 100
    
    # Parameters
    K_MIN = 1
    K_MAX = 5
    
    while time.time() < lns_deadline:
        # 1. Perturbation: Select K random boxes to remove
        K = random.randint(K_MIN, K_MAX)
        # Ensure we don't remove more boxes than exist
        K = min(K, len(current_indices))
        
        # Select indices to remove. 
        # We should prioritize removing boxes from bins that are not the last bin, 
        # or boxes that are "large" to see if they fit better elsewhere.
        # For simplicity, pick random indices from the current placement.
        
        # We need to know which bin each box is in.
        # Let's create a map: box_idx -> (bin_label, pos)
        # Actually, we just need to remove them from the bin lists.
        
        removed_boxes = []
        removed_indices = set()
        
        # Pick K random boxes from the current set
        # We pick from the list of all indices
        candidates_for_removal = current_indices[:]
        random.shuffle(candidates_for_removal)
        to_remove = candidates_for_removal[:K]
        
        for idx in to_remove:
            # Find which bin this box is in and remove it
            # This is O(B * N) which is fine for small N
            found = False
            for label in list(bins.keys()):
                # Check if box idx is in this bin
                # We need to identify the box. 
                # Wait, the bins store (x,y,z,w,h,d). We don't store the original index!
                # This is a problem for LNS if we don't track indices.
                
                # Let's fix the data structure: 
                # bins[label] = list of (idx, x, y, z, w, h, d)
                
                # I need to refactor the bins structure to include indices.
                pass
        
        # Refactoring bins to include indices is crucial for LNS.
        # Let's restart the construction with indices tracked.
        
        break # Break out to implement the indexed version properly below.
        
    # --- RE-IMPLEMENTATION WITH INDEXED BINS ---
    
    # Reset
    bins = {}
    bin_counter = 0
    
    def construct_solution_indexed(indices_to_place, bins, bin_counter):
        for idx in indices_to_place:
            w, h, d = boxes[idx]
            placed = False
            
            bin_labels = list(bins.keys())
            bin_labels.sort(key=lambda l: sum(bw*bh*bd for _,_,_,bw,bh,bd in bins[l]), reverse=True)
            
            best_bin_label = None
            best_pos = None
            best_spot_metric = float('inf')
            
            for label in bin_labels:
                pos = find_best_spot(bins[label], w, h, d)
                if pos is not None:
                    metric = (pos[2], pos[1], pos[0])
                    if metric < best_spot_metric:
                        best_spot_metric = metric
                        best_bin_label = label
                        best_pos = pos
            
            if best_bin_label is None:
                best_bin_label = bin_counter
                bins[best_bin_label] = []
                best_pos = (0, 0, 0)
                bin_counter += 1
                
            # Store (idx, x, y, z, w, h, d)
            bins[best_bin_label].append((idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
        return bin_counter

    # Initial Construction
    bin_counter = construct_solution_indexed(sorted_indices, bins, bin_counter)
    current_score = get_score(bins, max(bins.keys()) if bins else 0)
    
    best_bins = {k: v[:] for k, v in bins.items()}
    best_score = current_score
    
    # LNS Loop
    lns_start = time.time()
    lns_deadline = lns_start + 100
    
    while time.time() < lns_deadline:
        # 1. Perturbation
        K = random.randint(1, 4)
        # Pick K random boxes to remove
        all_indices = list(range(n))
        random.shuffle(all_indices)
        to_remove = all_indices[:K]
        
        # Remove them from current bins
        removed_items = []
        for idx in to_remove:
            for label in list(bins.keys()):
                # Find item in bin
                for j, item in enumerate(bins[label]):
                    if item[0] == idx:
                        removed_items.append(item)
                        bins[label].pop(j)
                        break
                if bins[label]:
                    break
                else:
                    # If bin is empty, remove it? 
                    # The score depends on B (number of occupied bins).
                    # An empty bin shouldn't count as occupied? 
                    # "B is the number of occupied bins". 
                    # So we should remove empty bins to keep B accurate.
                    del bins[label]
                    
        # 2. Re-insertion
        # Sort removed items by volume descending
        removed_items.sort(key=lambda item: volumes[item[0]], reverse=True)
        
        # Try to re-insert them
        success = True
        for item in removed_items:
            idx, x, y, z, w, h, d = item
            placed = False
            
            # Try all existing bins
            bin_labels = list(bins.keys())
            # Sort by volume descending
            bin_labels.sort(key=lambda l: sum(bw*bh*bd for _,_,_,bw,bh,bd in bins[l]), reverse=True)
            
            best_bin_label = None
            best_pos = None
            best_spot_metric = float('inf')
            
            for label in bin_labels:
                pos = find_best_spot(bins[label], w, h, d)
                if pos is not None:
                    metric = (pos[2], pos[1], pos[0])
                    if metric < best_spot_metric:
                        best_spot_metric = metric
                        best_bin_label = label
                        best_pos = pos
            
            if best_bin_label is None:
                # Open new bin
                best_bin_label = bin_counter
                bins[best_bin_label] = []
                best_pos = (0, 0, 0)
                bin_counter += 1
                
            bins[best_bin_label].append((idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
        # 3. Evaluation
        if bins:
            new_score = get_score(bins, max(bins.keys()))
            if new_score < best_score:
                best_score = new_score
                best_bins = {k: v[:] for k, v in bins.items()}
                # Optional: Accept and continue from here (Greedy LNS)
                # Or revert and try again? 
                # Accepting is usually better for progress.
        else:
            # Should not happen unless n=0
            break

    # If we didn't improve, keep the initial solution
    if best_score < current_score:
        # We already updated best_bins in the loop
        pass
    else:
        # Use the initial construction if LNS didn't improve
        # But wait, the LNS loop modifies 'bins'. 
        # If we never updated best_bins, best_bins is the initial solution.
        # If we did update, best_bins is the improved one.
        # We should return the placement from best_bins.
        
        pass

    # Convert best_bins to final placements format
    # placements[i] = (bin_label, x, y, z, w, h, d)
    final_placements = [None] * n
    
    # If best_bins is empty (should not happen), fall back to simple packing
    if not best_bins:
        # Fallback: simple shelf packing
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            final_placements[i] = (b, x, y, z, w, h, d)
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return final_placements

    for label, items in best_bins.items():
        for idx, x, y, z, w, h, d in items:
            final_placements[idx] = (label, x, y, z, w, h, d)
            
    # Safety check: ensure all placements are not None
    # If any are None, it means an index was missed. 
    # This can happen if we removed a box and failed to re-insert it properly? 
    # No, the re-insertion logic always places it (opens new bin if needed).
    # So all should be placed.
    
    return final_placements

# EVOLVE-BLOCK-END
