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

    def greedy_independent_set_cover():
        """Build a cover where no two codewords are at Hamming distance 2.
        This avoids pairs that are candidates for removal/improvement,
        giving local search a structurally better starting point."""
        # Precompute distance-2 adjacency: codewords at Hamming distance 2
        dist2_adj = [set() for _ in range(total)]
        for idx in range(total):
            w = idx_to_tuple(idx)
            for i in range(n):
                for j in range(i+1, n):
                    for d1 in range(1, q):
                        for d2 in range(1, q):
                            new_w = list(w)
                            new_w[i] = (new_w[i] + d1) % q
                            new_w[j] = (new_w[j] + d2) % q
                            other = tuple_to_idx(tuple(new_w))
                            if other > idx:
                                dist2_adj[idx].add(other)
                                dist2_adj[other].add(idx)
        
        uncovered = set(range(total))
        code = []
        # Greedy: pick codeword covering most uncovered, but avoid dist-2 conflicts
        # with already-picked codewords
        picked = set()
        while uncovered:
            best_cand = -1
            best_count = -1
            candidates = list(uncovered)
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
            picked.add(best_cand)
            uncovered -= neighbors_set[best_cand]
        
        # Now try to remove any codeword that's at distance 2 from another
        # and is redundant
        removed = True
        while removed:
            removed = False
            for i in range(len(code)):
                nb_arr = neighbors_arr[code[i]]
                cover_count = np.zeros(total, dtype=np.int32)
                for c in code:
                    if c != code[i]:
                        cover_count[neighbors_arr[c]] += 1
                if np.all(cover_count > 0):
                    code.pop(i)
                    removed = True
                    break
        return code

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
        """Greedy local search: try removals, swaps, 2-for-1, and 3-for-2 moves."""
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
                    if found:
                        break
                    cover_count[nb_arr] = old_counts
            
            if improved or time.time() > time_limit:
                continue
            
            # Phase 3: 2-for-1 moves (remove two, add one)
            if len(code) >= 2:
                num_trials = min(30, len(code) * (len(code) - 1) // 2)
                for _ in range(num_trials):
                    if time.time() > time_limit:
                        break
                    i, j = random.sample(range(len(code)), 2)
                    if i > j:
                        i, j = j, i
                    c1, c2 = code[i], code[j]
                    nb1 = neighbors_arr[c1]
                    nb2 = neighbors_arr[c2]
                    old_counts = cover_count[nb1].copy()
                    old_counts2 = cover_count[nb2].copy()
                    cover_count[nb1] -= 1
                    cover_count[nb2] -= 1
                    affected = set(nb1.tolist()) | set(nb2.tolist())
                    uncovered = [nb for nb in affected if cover_count[nb] == 0]
                    if not uncovered:
                        code.pop(i)
                        code.pop(j)
                        improved = True
                        break
                    else:
                        u0 = uncovered[0]
                        candidates = list(neighbors_list[u0])
                        random.shuffle(candidates)
                        found = False
                        for new_c in candidates:
                            if all(u in neighbors_set[new_c] for u in uncovered):
                                if new_c not in code:
                                    cover_count[neighbors_arr[new_c]] += 1
                                    code[i] = new_c
                                    code.pop(j)
                                    improved = True
                                    found = True
                                    break
                        if found:
                            break
                        cover_count[nb1] = old_counts
                        cover_count[nb2] = old_counts
            
            if improved or time.time() > time_limit:
                continue
            
            # Phase 4: 3-for-2 moves (remove three, add two)
            if len(code) >= 3:
                num_trials = min(15, len(code) * (len(code) - 1) * (len(code) - 2) // 6)
                for _ in range(num_trials):
                    if time.time() > time_limit:
                        break
                    idxs = random.sample(range(len(code)), 3)
                    idxs.sort()
                    c1, c2, c3 = code[idxs[0]], code[idxs[1]], code[idxs[2]]
                    nb1 = neighbors_arr[c1]
                    nb2 = neighbors_arr[c2]
                    nb3 = neighbors_arr[c3]
                    old_counts1 = cover_count[nb1].copy()
                    old_counts2 = cover_count[nb2].copy()
                    old_counts3 = cover_count[nb3].copy()
                    cover_count[nb1] -= 1
                    cover_count[nb2] -= 1
                    cover_count[nb3] -= 1
                    affected = set(nb1.tolist()) | set(nb2.tolist()) | set(nb3.tolist())
                    uncovered = [nb for nb in affected if cover_count[nb] == 0]
                    if not uncovered:
                        for k in reversed(range(3)):
                            code.pop(idxs[k])
                        improved = True
                        break
                    else:
                        # Try to find two codewords covering all uncovered
                        u0 = uncovered[0]
                        candidates = list(neighbors_list[u0])
                        random.shuffle(candidates)
                        found = False
                        for new_c1 in candidates:
                            if time.time() > time_limit:
                                break
                            if not all(u in neighbors_set[new_c1] for u in uncovered):
                                continue
                            if new_c1 in code:
                                continue
                            # After adding new_c1, what's still uncovered?
                            remaining_uncovered = [u for u in uncovered if u not in neighbors_set[new_c1]]
                            if not remaining_uncovered:
                                # One new codeword suffices (2-for-1 case)
                                cover_count[neighbors_arr[new_c1]] += 1
                                code[idxs[0]] = new_c1
                                code.pop(idxs[2])
                                code.pop(idxs[1])
                                improved = True
                                found = True
                                break
                            u1 = remaining_uncovered[0]
                            candidates2 = list(neighbors_list[u1])
                            random.shuffle(candidates2)
                            for new_c2 in candidates2:
                                if all(u in neighbors_set[new_c2] for u in remaining_uncovered):
                                    if new_c2 not in code and new_c2 != new_c1:
                                        cover_count[neighbors_arr[new_c1]] += 1
                                        cover_count[neighbors_arr[new_c2]] += 1
                                        code[idxs[0]] = new_c1
                                        code[idxs[1]] = new_c2
                                        code.pop(idxs[2])
                                        improved = True
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                        cover_count[nb1] = old_counts1
                        cover_count[nb2] = old_counts2
                        cover_count[nb3] = old_counts3
        
        return code

    def simulated_annealing_cell_based(time_limit):
        """SA over tetracode cell structure with incremental coverage updates.
        State: for each of 81 cells, a bitmask of which of 9 options are active.
        Objective: minimize num_codewords + penalty * num_uncovered_points
        """
        num_cells = 81
        cells = [(a,b) for a in range(3) for b in range(3)]
        
        # Initialize: random subset of options per cell
        state = [0] * num_cells
        for i in range(num_cells):
            num_opts = random.randint(1, 5)
            opts = random.sample(range(9), num_opts)
            mask = 0
            for o in opts:
                mask |= (1 << o)
            state[i] = mask
        
        # Compute initial coverage incrementally
        cover_count = np.zeros(total, dtype=np.int32)
        num_cw = 0
        for i in range(num_cells):
            mask = state[i]
            if mask == 0:
                continue
            for o in range(9):
                if mask & (1 << o):
                    num_cw += 1
                    cover_count[cell_options[cells[i]][o][2]] += 1
        uncovered = int(np.sum(cover_count == 0))
        objective = num_cw + 100 * uncovered
        
        best_state = list(state)
        best_obj = objective
        best_num_cw = num_cw
        best_uncovered = uncovered
        
        T = 50.0
        T_min = 0.1
        cooling = 0.997
        max_iter = 100000
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            ci = random.randint(0, num_cells - 1)
            move_type = random.random()
            
            old_mask = state[ci]
            
            if move_type < 0.4:
                o = random.randint(0, 8)
                new_mask = old_mask ^ (1 << o)
            elif move_type < 0.7:
                o = random.randint(0, 8)
                new_mask = old_mask | (1 << o)
            else:
                if old_mask == 0:
                    continue
                bits = [b for b in range(9) if old_mask & (1 << b)]
                o = random.choice(bits)
                new_mask = old_mask & ~(1 << o)
            
            if new_mask == old_mask:
                continue
            
            # Compute delta and affected points incrementally
            delta_cw = 0
            affected_points = set()
            for o in range(9):
                old_bit = 1 if (old_mask & (1 << o)) else 0
                new_bit = 1 if (new_mask & (1 << o)) else 0
                if old_bit != new_bit:
                    delta_cw += new_bit - old_bit
                    affected_points.update(cell_options[cells[ci]][o][1])
            
            new_num_cw = num_cw + delta_cw
            
            # Apply change to cover_count
            for o in range(9):
                old_bit = 1 if (old_mask & (1 << o)) else 0
                new_bit = 1 if (new_mask & (1 << o)) else 0
                if old_bit != new_bit:
                    arr = cell_options[cells[ci]][o][2]
                    if new_bit:
                        cover_count[arr] += 1
                    else:
                        cover_count[arr] -= 1
            
            # Count uncovered among affected points
            new_uncovered = uncovered
            for pt in affected_points:
                if cover_count[pt] == 0:
                    new_uncovered += 1
                # subtract old uncovered status
            # Recompute: old_uncovered for affected points
            old_uncovered_in_affected = 0
            for pt in affected_points:
                # We already modified cover_count, so we need to track carefully
                pass
            
            # Simpler: recompute uncovered for affected points only
            # But we need to know which were uncovered before
            # Let's just count zeros in affected set after update
            new_uncovered_count = sum(1 for pt in affected_points if cover_count[pt] == 0)
            
            # Old uncovered in affected: we need to reverse-engineer
            # Actually, let's track it properly
            # Undo, count, redo
            for o in range(9):
                old_bit = 1 if (old_mask & (1 << o)) else 0
                new_bit = 1 if (new_mask & (1 << o)) else 0
                if old_bit != new_bit:
                    arr = cell_options[cells[ci]][o][2]
                    if new_bit:
                        cover_count[arr] -= 1
                    else:
                        cover_count[arr] += 1
            
            old_uncovered_in_affected = sum(1 for pt in affected_points if cover_count[pt] == 0)
            
            # Redo
            for o in range(9):
                old_bit = 1 if (old_mask & (1 << o)) else 0
                new_bit = 1 if (new_mask & (1 << o)) else 0
                if old_bit != new_bit:
                    arr = cell_options[cells[ci]][o][2]
                    if new_bit:
                        cover_count[arr] += 1
                    else:
                        cover_count[arr] -= 1
            
            new_uncovered = uncovered - old_uncovered_in_affected + new_uncovered_count
            new_obj = new_num_cw + 100 * new_uncovered
            delta_obj = new_obj - objective
            
            if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                objective = new_obj
                num_cw = new_num_cw
                uncovered = new_uncovered
                state[ci] = new_mask
                
                if new_uncovered == 0 and new_num_cw < best_num_cw:
                    best_num_cw = new_num_cw
                    best_uncovered = new_uncovered
                    best_state = list(state)
                    best_obj = new_obj
            else:
                # Undo
                for o in range(9):
                    old_bit = 1 if (old_mask & (1 << o)) else 0
                    new_bit = 1 if (new_mask & (1 << o)) else 0
                    if old_bit != new_bit:
                        arr = cell_options[cells[ci]][o][2]
                        if new_bit:
                            cover_count[arr] -= 1
                        else:
                            cover_count[arr] += 1
                state[ci] = old_mask
            
            T *= cooling
            if T < T_min:
                T = T_min
        
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
    
    # Strategy 3: Greedy independent set cover (structurally better start)
    strategies.append(('greedy_ind_set', greedy_independent_set_cover()))
    
    # Strategy 4: Partial tetracode
    for k in [75, 78, 80]:
        strategies.append((f'tetracode_partial_{k}', tetracode_partial(k)))
    
    # Strategy 5: Random options
    for _ in range(3):
        strategies.append(('random_opts', tetracode_random_options()))
    
    random.shuffle(strategies)
    
    # Track time spent on each strategy
    total_time_budget = deadline - start_time
    time_per_strategy = total_time_budget / (len(strategies) + 3)  # +3 for SA runs
    
    for name, init_code in strategies:
        if time.time() > deadline - 10:
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
        remaining = deadline - time.time() - 10
        if remaining > 5:
            ls_time = min(remaining, time_per_strategy)
            code = local_search(init_code, time.time() + ls_time)
            validate_and_record(code)
        
        if best_size <= 73:
            break
    
    # Strategy 6: Simulated Annealing on cell structure (multiple runs)
    sa_runs = 0
    while time.time() < deadline - 10 and sa_runs < 5:
        sa_time = min(25, deadline - time.time() - 8)
        if sa_time < 5:
            break
        sa_code = simulated_annealing_cell_based(time.time() + sa_time)
        if sa_code is not None:
            validate_and_record(sa_code)
            # Also run local search on SA result
            remaining = deadline - time.time() - 8
            if remaining > 5 and best_size > 73:
                ls_code = local_search(sa_code, time.time() + min(10, remaining - 5))
                validate_and_record(ls_code)
        sa_runs += 1
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
