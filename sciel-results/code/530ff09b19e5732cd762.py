"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Precompute orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    orientations_cache = [get_orientations(w, h, d) for w, h, d in boxes]
    
    # Quick overlap check using list of placed boxes
    def fits(x, y, z, w, h, d, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    # Generate candidate positions: origin + all face-adjacent and corner positions
    def get_candidates(placed):
        cands = [(0, 0, 0)]
        seen = set(cands)
        for (px, py, pz, pw, ph, pd) in placed:
            for (nx, ny, nz) in [
                (px + pw, py, pz),
                (px, py + ph, pz),
                (px, py, pz + pd),
                (px + pw, py + ph, pz),
                (px + pw, py, pz + pd),
                (px, py + ph, pz + pd),
                (px + pw, py + ph, pz + pd),
            ]:
                if (nx, ny, nz) not in seen and nx <= W and ny <= H and nz <= D:
                    seen.add((nx, ny, nz))
                    cands.append((nx, ny, nz))
        return cands
    
    # Score a position: prefer lower z, then lower y, then lower x (pack tight to origin)
    def position_score(x, y, z):
        return z * W * H + y * W + x
    
    # Try to place a single box given current placed list; return (x,y,z,ow,oh,od) or None
    def place_one(box_idx, placed):
        w, h, d = boxes[box_idx]
        best = None
        best_score = float('inf')
        for (ow, oh, od) in orientations_cache[box_idx]:
            if ow > W or oh > H or od > D:
                continue
            for (ax, ay, az) in get_candidates(placed):
                if fits(ax, ay, az, ow, oh, od, placed):
                    s = position_score(ax, ay, az)
                    if s < best_score:
                        best_score = s
                        best = (ax, ay, az, ow, oh, od)
        return best
    
    def evaluate(result):
        vol = 0
        for p in result:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        return vol / container_vol
    
    # Greedy packing with a given order
    def greedy(order):
        placed = []
        result = [None] * n
        for idx in order:
            pos = place_one(idx, placed)
            if pos is not None:
                x, y, z, ow, oh, od = pos
                placed.append((x, y, z, ow, oh, od))
                result[idx] = (0, x, y, z, ow, oh, od)
            else:
                w, h, d = boxes[idx]
                result[idx] = (-1, 0, 0, 0, w, h, d)
        return result
    
    # Build various orders
    indices = list(range(n))
    
    def order_by_volume():
        return sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def order_by_maxdim():
        return sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    
    def order_by_mindim():
        return sorted(indices, key=lambda i: min(boxes[i]), reverse=True)
    
    def order_by_surface():
        return sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1]+boxes[i][1]*boxes[i][2]+boxes[i][2]*boxes[i][0]), reverse=True)
    
    def order_by_aspect():
        def asp(i):
            w,h,d = boxes[i]
            return max(w,h,d)/min(w,h,d) if min(w,h,d)>0 else 0
        return sorted(indices, key=lambda i: asp(i))
    
    def order_random():
        r = list(indices)
        random.shuffle(r)
        return r
    
    # Local search: try to swap a placed box with an unplaced one, or re-insert unplaced boxes
    def local_improve(result, placed, budget):
        """Try to improve by re-inserting left-out boxes or swapping."""
        improved = True
        while improved and time.time() < budget:
            improved = False
            left_out = [i for i in range(n) if result[i][0] == -1]
            for idx in left_out:
                pos = place_one(idx, placed)
                if pos is not None:
                    x, y, z, ow, oh, od = pos
                    placed.append((x, y, z, ow, oh, od))
                    result[idx] = (0, x, y, z, ow, oh, od)
                    improved = True
        return result, placed
    
    best_result = None
    best_fill = 0
    
    # Try deterministic strategies
    strategies = [order_by_volume, order_by_maxdim, order_by_mindim, order_by_surface, order_by_aspect]
    
    for strat in strategies:
        if time.time() > deadline:
            break
        order = strat()
        res = greedy(order)
        fill = evaluate(res)
        if fill > best_fill:
            best_fill = fill
            best_result = res
            # Build placed list for local improvement
            placed = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in res if p[0]==0]
            # Local improvement
            if time.time() < deadline - 5:
                res, placed = local_improve(res, placed, deadline - 5)
                fill2 = evaluate(res)
                if fill2 > best_fill:
                    best_fill = fill2
                    best_result = res
    
    # Try random orders with local improvement
    n_random = 30
    for r in range(n_random):
        if time.time() > deadline - 2:
            break
        order = order_random()
        res = greedy(order)
        fill = evaluate(res)
        if fill > best_fill:
            best_fill = fill
            best_result = res
            placed = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in res if p[0]==0]
            if time.time() < deadline - 5:
                res, placed = local_improve(res, placed, deadline - 5)
                fill2 = evaluate(res)
                if fill2 > best_fill:
                    best_fill = fill2
                    best_result = res
    
    # Final: try to re-insert any remaining left-out boxes into the best result
    if best_result is not None and time.time() < deadline - 1:
        placed = [(p[1],p[2],p[3],p[4],p[5],p[6]) for p in best_result if p[0]==0]
        for idx in range(n):
            if best_result[idx][0] == -1:
                pos = place_one(idx, placed)
                if pos is not None:
                    x,y,z,ow,oh,od = pos
                    placed.append((x,y,z,ow,oh,od))
                    best_result[idx] = (0,x,y,z,ow,oh,od)
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result
# EVOLVE-BLOCK-END
