import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    total = q ** n  # 729
    
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    neighbors_list = [None] * total
    neighbors_set = [None] * total
    neighbors_arr = [None] * total
    
    for idx in range(total):
        w = idx_to_tuple(idx)
        nbs = [idx]
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                nbs.append(tuple_to_idx(tuple(new_w)))
        neighbors_list[idx] = nbs
        neighbors_set[idx] = set(nbs)
        neighbors_arr[idx] = np.array(nbs, dtype=np.int32)

    # Tetracode structure: 9x9 grid where each cell (a,b) has 9 possible codewords
    # codeword = (a, b, (a+b)%3, (a+2b)%3, x, y) for x,y in {0,1,2}
    # We'll use SA over the 81 cells, each choosing from 9 options (or "empty")
    
    # Precompute for each cell (a,b) and option (x,y): the 19 covered points
    cell_options = {}  # (a,b) -> list of (codeword_idx, covered_set, covered_arr)
    for a in range(3):
        for b in range(3):
            opts = []
            for x in range(3):
                for y in range(3):
                    cw = (a, b, (a+b)%3, (a+2*b)%3, x, y)
                    cw_idx = tuple_to_idx(cw)
                    covered = neighbors_set[cw_idx]
                    covered_arr = neighbors_arr[cw_idx]
                    opts.append((cw_idx, covered, covered_arr))
            cell_options[(a,b)] = opts
    
    # Also precompute: for each point, which (cell, option) covers it
    point_to_coverers = [set() for _ in range(total)]
    for a in range(3):
        for b in range(3):
            for opt_idx, (cw_idx, covered, covered_arr) in enumerate(cell_options[(a,b)]):
                for pt in covered:
                    point_to_coverers[pt].add(((a,b), opt_idx))
    
    def greedy_cover():
        uncovered = set(range(total))
        code = []
        while uncovered:
            best_cand = -1
            best_count = -1
            candidates = list(range(total))
            random.shuffle(candidates)
            for cand in candidates:
                if not uncovered:
                    break
                covered_here = len(neighbors_set[cand] & uncovered)
                if covered_here > best_count:
                    best_count = covered_here
                    best_cand = cand
            if best_cand < 0:
                break
            code.append(best_cand)
            uncovered -= neighbors_set[best_cand]
        return code

    def tetracode_full():
        """Full tetracode: all 81 cells filled."""
        result = []
        for a in range(3):
            for b in range(3):
                for x in range(3):
                    for y in range(3):
                        cw = (a, b, (a+b)%3, (a+2*b)%3, x, y)
                        result.append(tuple_to_idx(cw))
        return result

    def tetracode_partial(k_cells):
        """Choose k_cells out of 81, fill them completely."""
        all_cells = [(a,b) for a in range(3) for b in range(3)]
        random.shuffle(all_cells)
        chosen = all_cells[:k_cells]
        result = []
        for (a,b) in chosen:
            for x in range(3):
                for y in range(3):
                    cw = (a, b, (a+b)%3, (a+2*b)%3, x, y)
                    result.append(tuple_to_idx(cw))
        return result

    def tetracode_random_options():
        """Each cell gets a random subset of its 9 options."""
        result = []
        for a in range(3):
            for b in range(3):
                num_opts = random.randint(1, 9)
                opts = random.sample(range(9), num_opts)
                for opt_idx in opts:
                    x, y = divmod(opt_idx, 3)
                    cw = (a, b, (a+b)%3, (a+2*b)%3, x, y)
                    result.append(tuple_to_idx(cw))
        return result

    start_time = time.time()
    deadline = 130.0  # Safety margin

    best_code = None
    best_size = float('inf')

    def validate_and_record(code):
        nonlocal best_code, best_size
        # Remove duplicates
        code = list(dict.fromkeys(code))
        if len(code) >= best_size:
            return
        # Validate coverage
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        if np.any(cover_count == 0):
            return
        best_code = list(code)
        best_size = len(code)

    def local_search(code, time_limit):
        """Greedy local search: try removals and swaps."""
        code = list(code)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(code)
            
            # Phase 1: removals
            for i in range(len(code)):
                if time.time() > time_limit:
                    break
                nb_arr = neighbors_arr[code[i]]
                old_counts = cover_count[nb_arr].copy()
                cover_count[nb_arr] -= 1
                uncovered = [nb for nb in nb_arr if cover_count[nb] == 0]
                if not uncovered:
                    code.pop(i)
                    improved = True
                    break
                else:
                    cover_count[nb_arr] = old_counts
            
            if improved or time.time() > time_limit:
                continue
            
            # Phase 2: single swaps
            for i in range(len(code)):
                if time.time() > time_limit:
                    break
                c = code[i]
                nb_arr = neighbors_arr[c]
                old_counts = cover_count[nb_arr].copy()
                cover_count[nb_arr] -= 1
                uncovered = [nb for nb in nb_arr if cover_count[nb] == 0]
                if not uncovered:
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Find replacement covering all uncovered
                    u0 = uncovered[0]
                    candidates = list(neighbors_list[u0])
                    random.shuffle(candidates)
                    found = False
                    for new_c in candidates:
                        if all(u in neighbors_set[new_c] for u in uncovered):
                            if new_c not in code:
                                cover_count[neighbors_arr[new_c]] += 1
                                code[i] = new_c
                                improved = True
                                found = True
                                break
                            else:
                                # Already in code, just restore
                                pass
                    if found:
                        break
                    cover_count[nb_arr] = old_counts
        
        return code

    def simulated_annealing_cell_based(time_limit):
        """SA over tetracode cell structure.
        State: for each of 81 cells, a bitmask of which of 9 options are active.
        Objective: minimize num_codewords + penalty * num_uncovered_points
        """
        num_cells = 81
        cells = [(a,b) for a in range(3) for b in range(3)]
        
        # Initialize: random subset of options per cell
        state = [0] * num_cells  # bitmask, bit i = option i active
        for i in range(num_cells):
            num_opts = random.randint(1, 5)
            opts = random.sample(range(9), num_opts)
            mask = 0
            for o in opts:
                mask |= (1 << o)
            state[i] = mask
        
        # Compute initial coverage
        def compute_coverage(state):
            cover_count = np.zeros(total, dtype=np.int32)
            num_cw = 0
            for i in range(num_cells):
                mask = state[i]
                if mask == 0:
                    continue
                for o in range(9):
                    if mask & (1 << o):
                        num_cw += 1
                        cw_idx, covered, covered_arr = cell_options[cells[i]][o]
                        cover_count[covered_arr] += 1
            uncovered = int(np.sum(cover_count == 0))
            return num_cw, uncovered, cover_count
        
        num_cw, uncovered, cover_count = compute_coverage(state)
        objective = num_cw + 100 * uncovered
        
        best_state = list(state)
        best_obj = objective
        best_num_cw = num_cw
        best_uncovered = uncovered
        
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        max_iter = 50000
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            # Pick a random cell and random move
            ci = random.randint(0, num_cells - 1)
            move_type = random.random()
            
            old_mask = state[ci]
            
            if move_type < 0.4:
                # Toggle a random option
                o = random.randint(0, 8)
                new_mask = old_mask ^ (1 << o)
            elif move_type < 0.7:
                # Add a random option
                o = random.randint(0, 8)
                new_mask = old_mask | (1 << o)
            else:
                # Remove a random option
                if old_mask == 0:
                    continue
                # Pick a set bit
                bits = [b for b in range(9) if old_mask & (1 << b)]
                o = random.choice(bits)
                new_mask = old_mask & ~(1 << o)
            
            if new_mask == old_mask:
                continue
            
            # Compute delta
            # Only need to update coverage for points in this cell's options
            delta_cw = 0
            for o in range(9):
                old_bit = 1 if (old_mask & (1 << o)) else 0
                new_bit = 1 if (new_mask & (1 << o)) else 0
                delta_cw += new_bit - old_bit
            
            # Update cover_count for affected points
            # Points affected: union of covered sets for toggled options
            affected_points = set()
            for o in range(9):
                old_bit = 1 if (old_mask & (1 << o)) else 0
                new_bit = 1 if (new_mask & (1 << o)) else 0
                if old_bit != new_bit:
                    affected_points.update(cell_options[cells[ci]][o][1])
            
            # Compute new uncovered count
            old_uncovered = uncovered
            new_uncovered = uncovered
            for pt in affected_points:
                if cover_count[pt] == 0:
                    new_uncovered -= 1  # was uncovered
                # After change, check if still covered
                # We need to compute new cover_count[pt]
            
            # More efficient: just recompute for affected points
            # Apply the change temporarily
            state[ci] = new_mask
            new_num_cw = num_cw + delta_cw
            
            # Recompute uncovered for affected points only
            # Actually, let's just recompute fully for simplicity since 81 cells is small
            # But that's 81*9*19 ops per step... let's be smarter
            
            # For now, do full recompute (81 cells * up to 9 options * 19 points = ~13833 ops)
            new_cover_count = np.zeros(total, dtype=np.int32)
            for i in range(num_cells):
                mask = state[i]
                if mask == 0:
                    continue
                for o in range(9):
                    if mask & (1 << o):
                        new_cover_count[cell_options[cells[i]][o][2]] += 1
            
            new_uncovered = int(np.sum(new_cover_count == 0))
            new_obj = new_num_cw + 100 * new_uncovered
            
            delta_obj = new_obj - objective
            
            if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                # Accept
                objective = new_obj
                num_cw = new_num_cw
                uncovered = new_uncovered
                cover_count = new_cover_count
                state[ci] = new_mask
                
                if new_uncovered == 0 and new_num_cw < best_num_cw:
                    best_num_cw = new_num_cw
                    best_uncovered = new_uncovered
                    best_state = list(state)
                    best_obj = new_obj
            else:
                # Reject
                state[ci] = old_mask
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Extract best code
        if best_uncovered == 0:
            code = []
            for i in range(num_cells):
                mask = best_state[i]
                for o in range(9):
                    if mask & (1 << o):
                        cw_idx, _, _ = cell_options[cells[i]][o]
                        code.append(cw_idx)
            return code
        return None

    # Try different strategies
    strategies = []
    
    # Strategy 1: Full tetracode + local search
    strategies.append(('tetracode_full', tetracode_full()))
    
    # Strategy 2: Greedy cover + local search
    strategies.append(('greedy', greedy_cover()))
    
    # Strategy 3: Partial tetracode
    for k in [75, 78, 80]:
        strategies.append((f'tetracode_partial_{k}', tetracode_partial(k)))
    
    # Strategy 4: Random options
    for _ in range(3):
        strategies.append(('random_opts', tetracode_random_options()))
    
    random.shuffle(strategies)
    
    for name, init_code in strategies:
        if time.time() > deadline:
            break
        
        # Ensure validity
        cover_count = np.zeros(total, dtype=np.int32)
        for c in init_code:
            cover_count[neighbors_arr[c]] += 1
        if np.any(cover_count == 0):
            uncovered = set(np.where(cover_count == 0)[0].tolist())
            while uncovered:
                best_cand = -1
                best_count = -1
                for cand in list(uncovered):
                    covered_here = len(neighbors_set[cand] & uncovered)
                    if covered_here > best_count:
                        best_count = covered_here
                        best_cand = cand
                if best_cand < 0:
                    break
                init_code.append(best_cand)
                cover_count[neighbors_arr[best_cand]] += 1
                uncovered -= neighbors_set[best_cand]
        
        init_code = list(dict.fromkeys(init_code))
        validate_and_record(init_code)
        
        # Local search with time budget
        remaining = deadline - time.time()
        if remaining > 5:
            ls_time = min(remaining - 5, 30)
            code = local_search(init_code, time.time() + ls_time)
            validate_and_record(code)
        
        if best_size <= 73:
            break
    
    # Strategy 5: Simulated Annealing on cell structure
    if time.time() < deadline - 10:
        sa_time = deadline - time.time() - 5
        sa_code = simulated_annealing_cell_based(time.time() + sa_time)
        if sa_code is not None:
            validate_and_record(sa_code)
    
    # Final: try more SA runs if time permits
    while time.time() < deadline - 5:
        sa_code = simulated_annealing_cell_based(time.time() + min(20, deadline - time.time() - 3))
        if sa_code is not None:
            validate_and_record(sa_code)
        if best_size <= 73:
            break
    
    if best_code is None:
        best_code = tetracode_full()
    
    # Final verification
    cover_count = np.zeros(total, dtype=np.int32)
    for c in best_code:
        cover_count[neighbors_arr[c]] += 1
    if np.any(cover_count == 0):
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    return [idx_to_tuple(c) for c in best_code]
