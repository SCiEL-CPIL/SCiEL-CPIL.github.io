"""Strong 3D Container Loading: skyline-based placement with deadline guard."""
import time

# EVOLVE-BLOCK-START
def load(boxes, container):
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin vs 120s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Precompute orientations for each box, sorted by priority
    box_data = []
    for idx, (w, h, d) in enumerate(boxes):
        # Generate all unique orientations
        seen = set()
        orients = []
        for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        
        # Filter to only orientations that fit in container
        valid_orients = [(bw, bh, bd) for bw, bh, bd in orients if bw <= W and bh <= H and bd <= D]
        if not valid_orients:
            valid_orients = [(w, h, d)]  # will be marked as not placed
        
        # Sort orientations: prefer those with smaller max dimension (more "flat")
        valid_orients.sort(key=lambda o: (max(o), o[0], o[1], o[2]))
        
        volume = w * h * d
        box_data.append((idx, valid_orients, volume, w, h, d))
    
    # Sort boxes by volume descending (largest first)
    box_data.sort(key=lambda x: x[2], reverse=True)
    
    placed = []  # list of (x, y, z, bw, bh, bd)
    results = [None] * len(boxes)
    
    def get_candidates():
        """Generate candidate positions from placed boxes and origin."""
        cx_set = {0}
        cy_set = {0}
        cz_set = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            cx_set.add(px)
            cx_set.add(px + pw)
            cy_set.add(py)
            cy_set.add(py + ph)
            cz_set.add(pz)
            cz_set.add(pz + pd)
        cx_list = sorted(cx_set)
        cy_list = sorted(cy_set)
        cz_list = sorted(cz_set)
        return cx_list, cy_list, cz_list
    
    def check_no_overlap(cx, cy, cz, bw, bh, bd):
        """Check if a box at (cx,cy,cz) with dims (bw,bh,bd) doesn't overlap any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            # No overlap if separated in any dimension
            if not ((cx + bw <= px) or (px + pw <= cx) or
                    (cy + bh <= py) or (py + ph <= cy) or
                    (cz + bd <= pz) or (pz + pd <= cz)):
                return False
        return True
    
    def try_place_box(idx, orients, volume, ow, oh, od):
        """Try to place a box using best-fit decreasing with early termination."""
        cx_list, cy_list, cz_list = get_candidates()
        
        # Generate all candidate positions sorted by (z, y, x) to prefer lowest positions
        candidates = []
        for cz in cz_list:
            for cy in cy_list:
                for cx in cx_list:
                    candidates.append((cx, cy, cz))
        candidates.sort(key=lambda c: (c[2], c[1], c[0]))
        
        # Sort orientations by max dimension (ascending) to prefer "flatter" boxes for the same position
        # This allows us to stop at the first valid orientation for a given candidate
        sorted_orients = sorted(orients, key=lambda o: (max(o), o[0], o[1], o[2]))
        
        for (cx, cy, cz) in candidates:
            for (bw, bh, bd) in sorted_orients:
                # Check bounds
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                # Check overlap
                if not check_no_overlap(cx, cy, cz, bw, bh, bd):
                    continue
                
                # If we reach here, this is the best possible placement
                # because candidates are sorted by (z,y,x) and orients by max_dim
                placed.append((cx, cy, cz, bw, bh, bd))
                results[idx] = (0, cx, cy, cz, bw, bh, bd)
                return True
        
        results[idx] = (-1, 0, 0, 0, ow, oh, od)
        return False
    
    def run_heuristic(box_order):
        """Run the placement algorithm with a specific box ordering."""
        local_placed = []
        local_results = [None] * len(boxes)
        
        # We need to redefine the nested functions or pass them in.
        # Since they depend on `placed` and `results` which are local to this closure,
        # it's easier to just duplicate the logic or refactor.
        # To keep it simple and avoid deep refactoring, we will just re-implement the loop
        # using the existing `try_place_box` but we need to reset `placed` and `results`.
        
        # The current `try_place_box` uses `placed` and `results` from the outer scope.
        # We need to make `try_place_box` accept `placed` and `results` as arguments or use a class.
        # Given the constraints, I will refactor `try_place_box` to take `placed` and `results`.
        
        pass # Placeholder, will be replaced by full refactor below

    # Refactor: Make try_place_box and get_candidates work with local state
    # This requires changing the definitions above.
    
    # Let's do a cleaner refactor of the whole load function body to support multiple runs.
    
    def execute_strategy(sort_key_func):
        """Execute a placement strategy and return the results."""
        nonlocal placed, results
        
        # Reset state
        placed.clear()
        for i in range(len(boxes)):
            results[i] = None
            
        # Create a local copy of box_data sorted by the strategy
        current_box_data = sorted(box_data, key=sort_key_func)
        
        # Phase 1
        for idx, orients, volume, ow, oh, od in current_box_data:
            if time.time() > deadline:
                break
            try_place_box(idx, orients, volume, ow, oh, od)
            
        # Phase 2
        unplaced = []
        for idx, orients, volume, ow, oh, od in current_box_data:
            if results[idx] is None or results[idx][0] == -1:
                unplaced.append((idx, orients, volume, ow, oh, od))
        
        unplaced.sort(key=lambda x: x[2])
        
        for idx, orients, volume, ow, oh, od in unplaced:
            if time.time() > deadline:
                break
            try_place_box(idx, orients, volume, ow, oh, od)
            
        # Fill remaining
        for i in range(len(boxes)):
            if results[i] is None:
                w_i, h_i, d_i = boxes[i]
                results[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
                
        # Calculate score
        total_vol = sum(w*h*d for w,h,d in boxes)
        filled_vol = 0
        for res in results:
            if res[0] == 0:
                filled_vol += res[4]*res[5]*res[6]
        
        return filled_vol, list(results)

    # Strategy 1: Volume Descending
    best_score = -1
    best_res = None
    
    vol1, res1 = execute_strategy(lambda x: -x[2])
    if vol1 > best_score:
        best_score = vol1
        best_res = res1
        
    # Strategy 2: Max Dimension Descending
    if time.time() < deadline - 20: # Reserve 20s
        vol2, res2 = execute_strategy(lambda x: -max(x[3], x[4], x[5]))
        if vol2 > best_score:
            best_score = vol2
            best_res = res2
            
    # Strategy 3: Surface Area Descending
    if time.time() < deadline - 10:
        vol3, res3 = execute_strategy(lambda x: -(2*(x[3]*x[4] + x[4]*x[5] + x[5]*x[3])))
        if vol3 > best_score:
            best_score = vol3
            best_res = res3
            
    if best_res:
        return best_res
    else:
        # Fallback if something went wrong
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
# EVOLVE-BLOCK-END
