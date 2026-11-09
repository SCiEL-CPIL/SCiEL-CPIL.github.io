"""3D Bin Packing: multi-strategy greedy with local improvement."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    container_vol = W * H * D
    if container_vol == 0:
        return [(0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin before 120s limit

    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]

    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    def can_place_in_bin(bin_items, x, y, z, w, h, d):
        """Check if box at (x,y,z) with dims (w,h,d) fits in bin without overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (_, bx, by, bz, bw, bh, bd) in bin_items:
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True

    def get_candidate_positions(bin_items, w, h, d):
        """Generate valid candidate positions for a box in a bin."""
        cands = set([(0, 0, 0)])
        for (_, bx, by, bz, bw, bh, bd) in bin_items:
            for dx in (0, bw):
                for dy in (0, bh):
                    for dz in (0, bd):
                        cands.add((bx + dx, by + dy, bz + dz))
        
        valid = []
        for (cx, cy, cz) in cands:
            if can_place_in_bin(bin_items, cx, cy, cz, w, h, d):
                valid.append((cx, cy, cz))
        return valid

    def compute_score(assignments):
        """Compute score: B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not assignments:
            return float('inf')
        bins_used = set(assignments.values())
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(box_vols[idx] for idx, b in assignments.items() if b == last_bin)
        frac = last_vol / container_vol
        return B + 0.1 * (1 - frac)

    def greedy_pack(order, bins, assignments):
        """Place boxes in given order. bins is list of lists of (idx, x, y, z, w, h, d)."""
        for idx, (w, h, d) in order:
            if time.time() > deadline:
                break
                
            placed = False
            if bins:
                best_bin_id = -1
                best_pos = None
                best_metric = float('inf')
                
                for b_id in range(len(bins)):
                    cands = get_candidate_positions(bins[b_id], w, h, d)
                    if cands:
                        cands.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))
                        cx, cy, cz = cands[0]
                        metric = cx + cy + cz
                        if metric < best_metric:
                            best_metric = metric
                            best_bin_id = b_id
                            best_pos = (cx, cy, cz)
                
                if best_bin_id >= 0:
                    cx, cy, cz = best_pos
                    bins[best_bin_id].append((idx, cx, cy, cz, w, h, d))
                    assignments[idx] = best_bin_id
                    placed = True
            
            if not placed:
                b_id = len(bins)
                bins.append([(idx, 0, 0, 0, w, h, d)])
                assignments[idx] = b_id

    def try_move_box(bins, assignments, idx, from_bin_id, to_bin_id, w, h, d):
        """Try to move box idx. Returns True if moved."""
        cands = get_candidate_positions(bins[to_bin_id], w, h, d)
        if cands:
            cands.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))
            cx, cy, cz = cands[0]
            if can_place_in_bin(bins[to_bin_id], cx, cy, cz, w, h, d):
                # Remove from source
                bins[from_bin_id] = [b for b in bins[from_bin_id] if b[0] != idx]
                # Add to target
                bins[to_bin_id].append((idx, cx, cy, cz, w, h, d))
                assignments[idx] = to_bin_id
                return True
        return False

    def local_search(bins, assignments):
        """Local search to minimize B, then maximize last-bin fill."""
        if time.time() > deadline:
            return

        # Phase 1: Minimize number of bins by moving boxes out of higher-numbered bins
        for _pass in range(30):
            if time.time() > deadline:
                break
            
            occupied_bins = sorted([b for b in range(len(bins)) if bins[b]], reverse=True)
            made_progress = False
            
            for b_id in occupied_bins:
                if time.time() > deadline:
                    break
                if not bins[b_id]:
                    continue
                
                boxes_in_bin = list(bins[b_id])
                
                for (idx, x, y, z, w, h, d) in boxes_in_bin:
                    if time.time() > deadline:
                        break
                    if not any(b[0] == idx for b in bins[b_id]):
                        continue
                    
                    moved = False
                    for target_b in range(b_id):
                        if not bins[target_b]:
                            continue
                        if try_move_box(bins, assignments, idx, b_id, target_b, w, h, d):
                            moved = True
                            made_progress = True
                            break
                    
                    if not moved:
                        for target_b in range(len(bins)):
                            if target_b == b_id or not bins[target_b]:
                                continue
                            if try_move_box(bins, assignments, idx, b_id, target_b, w, h, d):
                                moved = True
                                made_progress = True
                                break
            
            if not made_progress:
                break

        # Phase 2: Try to fill the last bin more to reduce the 0.1 penalty
        if time.time() < deadline:
            occupied_bins = [b for b in range(len(bins)) if bins[b]]
            if occupied_bins:
                last_bin_id = max(occupied_bins)
                
                for b_id in sorted(occupied_bins, reverse=True):
                    if time.time() > deadline:
                        break
                    if b_id == last_bin_id or not bins[b_id]:
                        continue
                    
                    boxes_in_bin = list(bins[b_id])
                    for (idx, x, y, z, w, h, d) in boxes_in_bin:
                        if time.time() > deadline:
                            break
                        if not any(b[0] == idx for b in bins[b_id]):
                            continue
                        
                        if try_move_box(bins, assignments, idx, b_id, last_bin_id, w, h, d):
                            pass

    def build_result(bins, assignments):
        """Construct result list from bins and assignments."""
        result = [None] * n
        for idx in range(n):
            if idx in assignments:
                b_id = assignments[idx]
                # Find placement in bin
                pos = None
                for b in bins[b_id]:
                    if b[0] == idx:
                        pos = b
                        break
                if pos:
                    result[idx] = (b_id, pos[1], pos[2], pos[3], pos[4], pos[5], pos[6])
                else:
                    # Fallback: place at origin in a new bin
                    w, h, d = boxes[idx]
                    new_b = len(bins)
                    bins.append([(idx, 0, 0, 0, w, h, d)])
                    assignments[idx] = new_b
                    result[idx] = (new_b, 0, 0, 0, w, h, d)
            else:
                # Box wasn't placed (time out during initial packing)
                w, h, d = boxes[idx]
                new_b = len(bins)
                bins.append([(idx, 0, 0, 0, w, h, d)])
                assignments[idx] = new_b
                result[idx] = (new_b, 0, 0, 0, w, h, d)
        
        # Renumber bins to be consecutive starting from 0
        if result:
            bin_labels = sorted(set(r[0] for r in result if r is not None))
            if bin_labels:
                label_map = {old: new for new, old in enumerate(bin_labels)}
                for i in range(n):
                    if result[i] is not None:
                        old_bin = result[i][0]
                        new_bin = label_map[old_bin]
                        result[i] = (new_bin, result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6])
        
        return result

    def validate_result(result):
        """Validate that result is feasible."""
        bins_boxes = {}
        for idx in range(n):
            b, x, y, z, w, h, d = result[idx]
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Verify dimensions match original
            ow, oh, od = boxes[idx]
            if w != ow or h != oh or d != od:
                return False
            if b not in bins_boxes:
                bins_boxes[b] = []
            bins_boxes[b].append((x, y, z, w, h, d))
        
        for b, bl in bins_boxes.items():
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    if is_overlap(bl[i][0], bl[i][1], bl[i][2], bl[i][3], bl[i][4], bl[i][5],
                                  bl[j][0], bl[j][1], bl[j][2], bl[j][3], bl[j][4], bl[j][5]):
                        return False
        return True

    # --- Multi-strategy greedy packing with randomized restarts ---
    
    strategies = [
        lambda t: -(t[1][0] * t[1][1] * t[1][2]),  # Volume desc
        lambda t: t[1][0] * t[1][1] * t[1][2],      # Volume asc
        lambda t: -(max(t[1])),                      # Max dim desc
        lambda t: -(t[1][0] + t[1][1] + t[1][2]),   # Sum dim desc
        lambda t: -(min(t[1])),                      # Min dim desc
        lambda t: -(t[1][0] * t[1][1]),              # Area desc
        lambda t: -(t[1][1] * t[1][2]),              # Side area desc
        lambda t: -(t[1][0] * t[1][2]),              # Front area desc
    ]

    best_score = float('inf')
    best_result = None
    
    # Allocate time: first 60% for diverse strategies, remaining for best strategy restarts
    total_time_budget = deadline - start_time
    phase1_budget = total_time_budget * 0.6
    
    # Phase 1: Try each strategy once
    for strat_func in strategies:
        if time.time() > start_time + phase1_budget:
            break
        
        order = list(enumerate(boxes))
        order.sort(key=strat_func)
        
        bins = []
        assignments = {}
        
        greedy_pack(order, bins, assignments)
        local_search(bins, assignments)
        result = build_result(bins, assignments)
        
        if not validate_result(result):
            continue
        
        score = compute_score(assignments)
        
        if score < best_score:
            best_score = score
            best_result = result

    # Phase 2: Focus restarts on the best-performing strategy
    if best_result is not None and time.time() < deadline:
        # Find which strategy gave the best result by trying a few more with the top strategies
        top_strategies = strategies[:4]  # top 4 strategies
        
        for strat_func in top_strategies:
            if time.time() > deadline:
                break
            
            for restart in range(15):
                if time.time() > deadline:
                    break
                
                order = list(enumerate(boxes))
                # Add random noise for diversity
                order.sort(key=lambda t: strat_func(t) + random.uniform(-0.05, 0.05) * (t[1][0] * t[1][1] * t[1][2] + 1))
                
                bins = []
                assignments = {}
                
                greedy_pack(order, bins, assignments)
                local_search(bins, assignments)
                result = build_result(bins, assignments)
                
                if not validate_result(result):
                    continue
                
                score = compute_score(assignments)
                
                if score < best_score:
                    best_score = score
                    best_result = result

    # Fallback if no valid result
    if best_result is None:
        bins = []
        assignments = {}
        order = list(enumerate(boxes))
        order.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))
        greedy_pack(order, bins, assignments)
        best_result = build_result(bins, assignments)
    
    return best_result
# EVOLVE-BLOCK-END
