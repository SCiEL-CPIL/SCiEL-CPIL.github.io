"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """Compute HP energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _fast_grow(sequence, rng, gain_weight=2.0, free_weight=0.1, noise=0.5):
    """Randomized greedy chain growth biased toward H-H contact formation."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            options.append((gain * gain_weight + free_weight * free + rng.random() * noise, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy_delta_single(sequence, chain, i, old_pos, new_pos, occupied):
    """Compute energy delta for single-residue move of residue i from old_pos to new_pos."""
    if new_pos in occupied:
        return 0, False
    
    for q in _neighbours2d(new_pos):
        j = occupied.get(q)
        if j is not None and abs(j - i) > 1:
            return 0, False
    
    delta = 0
    is_h_i = sequence[i] == "H"
    
    if is_h_i:
        for q in _neighbours2d(old_pos):
            j = occupied.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                delta += 1
    
    if is_h_i:
        for q in _neighbours2d(new_pos):
            j = occupied.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                delta -= 1
    
    return delta, True


def _kink_flip_positions(chain, i):
    """Return the two possible flip positions for a kink at residue i.
    A kink is when chain[i-1] -> chain[i] -> chain[i+1] makes a 90-degree turn.
    The flip rotates chain[i+1] to the other 90-degree position around chain[i]."""
    if i < 1 or i > len(chain) - 2:
        return []
    
    p_prev = chain[i - 1]
    p_curr = chain[i]
    p_next = chain[i + 1]
    
    # Vector from prev to curr
    dx1 = p_curr[0] - p_prev[0]
    dy1 = p_curr[1] - p_prev[1]
    # Vector from curr to next
    dx2 = p_next[0] - p_curr[0]
    dy2 = p_next[1] - p_curr[1]
    
    # Check if it's a kink (perpendicular vectors)
    if dx1 * dx2 + dy1 * dy2 != 0:
        return []
    
    # The flip position: reflect p_next across the line through p_prev and p_curr
    # Actually, for a 90-degree turn, the flip is the other 90-degree turn
    # p_new = p_curr + (p_curr - p_prev) rotated 90 degrees? No.
    # If p_prev=(0,0), p_curr=(1,0), p_next=(1,1), flip is (1,-1)
    # p_new = p_curr + (p_next - p_curr) reflected across direction (p_curr - p_prev)
    # Direction d = (dx1, dy1). Reflect v=(dx2,dy2) across d.
    # Reflection of v across d: 2*(v.d)/(d.d)*d - v
    # But simpler: if d is horizontal, flip the vertical component of v
    # if d is vertical, flip the horizontal component of v
    
    if dx1 != 0:  # horizontal direction
        # p_next is offset vertically from p_curr
        p_new = (p_curr[0], p_curr[1] - dy2)
    else:  # vertical direction
        # p_next is offset horizontally from p_curr
        p_new = (p_curr[0] - dx2, p_curr[1])
    
    return [p_new]


def _local_search_sa(sequence, chain, max_iter=2000, use_kinks=True, use_sa=True):
    """Local search with single-residue moves and optional kink flips, with optional SA."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    # SA parameters
    if use_sa:
        T = 2.0
        T_min = 0.01
        cooling = (T_min / T) ** (1.0 / max(1, max_iter))
    
    no_improve = 0
    max_no_improve = 50
    
    for it in range(max_iter):
        if no_improve > max_no_improve:
            break
        
        improved = False
        rng_local = random.Random(hash((id(chain), it)) & 0xFFFFFFFF)
        
        # Choose a move type
        do_kink = use_kinks and n >= 3 and rng_local.random() < 0.4
        do_single = not do_kink
        
        if do_single:
            # Single residue move
            i = rng_local.randint(1, n - 2)
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                delta, _ = _energy_delta_single(sequence, best_chain, i, current_pos, new_pos, occupied)
                
                accept = False
                if delta < 0:
                    accept = True
                elif use_sa and T > T_min:
                    import math
                    accept = rng_local.random() < math.exp(-delta / T)
                
                if accept:
                    best_chain[i] = new_pos
                    occupied.pop(current_pos)
                    occupied[new_pos] = i
                    best_energy += delta
                    if delta < 0:
                        improved = True
                    no_improve = 0 if delta < 0 else no_improve + 1
                    break
        else:
            # Kink flip
            i = rng_local.randint(1, n - 2)
            new_pos_list = _kink_flip_positions(best_chain, i)
            if new_pos_list:
                new_pos = new_pos_list[0]
                if new_pos not in occupied:
                    # Check validity
                    valid = True
                    for q in _neighbours2d(new_pos):
                        j = occupied.get(q)
                        if j is not None and abs(j - (i + 1)) > 1:
                            valid = False
                            break
                    if valid:
                        # Compute energy delta for moving residue i+1
                        old_pos_next = best_chain[i + 1]
                        # Remove old position
                        occupied.pop(old_pos_next)
                        occupied[new_pos] = i + 1
                        new_energy = _compute_energy(sequence, best_chain[:i+1] + [new_pos] + best_chain[i+2:])
                        delta = new_energy - best_energy
                        best_chain[i + 1] = new_pos
                        
                        accept = False
                        if delta < 0:
                            accept = True
                        elif use_sa and T > T_min:
                            import math
                            accept = rng_local.random() < math.exp(-delta / T)
                        
                        if not accept:
                            # Revert
                            best_chain[i + 1] = old_pos_next
                            occupied.pop(new_pos)
                            occupied[old_pos_next] = i + 1
                        else:
                            best_energy = new_energy
                            if delta < 0:
                                improved = True
                            no_improve = 0 if delta < 0 else no_improve + 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        if not improved and not do_kink:
            pass
        
        if use_sa:
            T *= cooling
    
    # Final greedy descent
    for _ in range(200):
        improved = False
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                delta, _ = _energy_delta_single(sequence, best_chain, i, current_pos, new_pos, occupied)
                if delta < 0:
                    best_chain[i] = new_pos
                    occupied.pop(current_pos)
                    occupied[new_pos] = i
                    best_energy += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_chain, best_energy


def _min_conflicts_fold(sequence, max_iter=5000):
    """Min-conflicts style local search."""
    n = len(sequence)
    if n < 4:
        chain = [(i, 0) for i in range(n)]
        return chain, _compute_energy(sequence, chain)
    
    rng = random.Random(42 + n)
    chain = _fast_grow(sequence, rng)
    if chain is None:
        chain = [(i, 0) for i in range(n)]
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        best_i, best_delta, best_new_pos = -1, 0, None
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if new_pos in occupied:
                    continue
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                delta, _ = _energy_delta_single(sequence, best_chain, i, current_pos, new_pos, occupied)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_pos = new_pos
        
        if best_i == -1 or best_delta >= 0:
            break
        
        old_pos = best_chain[best_i]
        best_chain[best_i] = best_new_pos
        occupied.pop(old_pos)
        occupied[best_new_pos] = best_i
        best_energy += best_delta
    
    return best_chain, best_energy


def _regrow_sa_fast(sequence, start_chain, rng, max_iter=1000):
    """Simulated annealing on chain regrowth."""
    n = len(sequence)
    if n < 6:
        return _fast_grow(sequence, rng)
    
    best_chain = list(start_chain)
    best_energy = _compute_energy(sequence, best_chain)
    
    for _ in range(max_iter):
        split = rng.randint(3, n - 3)
        
        prefix = best_chain[:split]
        occupied = {p: i for i, p in enumerate(prefix)}
        chain = list(prefix)
        
        success = True
        for i in range(split, n):
            options = []
            for site in _neighbours2d(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours2d(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours2d(site) if q not in occupied)
                options.append((gain * 2.0 + 0.1 * free + rng.random(), site))
            if not options:
                success = False
                break
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        
        if not success:
            continue
        
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using diverse growth, SA local search with kinks, and min-conflicts."""
    start_time = time.time()
    deadline = start_time + 175.0
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Phase 1: Diverse fast growth + SA local search with kinks
    num_phase1 = max(30, 2000 // n)
    for attempt in range(num_phase1):
        if time.time() > deadline:
            break
        
        # Vary growth parameters for diversity
        gw = 1.5 + rng.random() * 2.0
        fw = 0.05 + rng.random() * 0.15
        noise = 0.3 + rng.random() * 0.7
        
        chain = _fast_grow(sequence, rng, gain_weight=gw, free_weight=fw, noise=noise)
        if chain is None:
            continue
        
        use_kinks = n >= 5
        use_sa = n >= 8
        max_iter_ls = min(1500, max(200, 4000 // n))
        chain, energy = _local_search_sa(sequence, chain, max_iter=max_iter_ls, use_kinks=use_kinks, use_sa=use_sa)
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    # Phase 2: Min-conflicts on the best found so far
    if best is not None and time.time() < deadline - 15:
        mc_chain, mc_energy = _min_conflicts_fold(sequence, max_iter=min(3000, max(500, 8000 // n)))
        if mc_energy < best_energy:
            best, best_energy = mc_chain, mc_energy
    
    # Phase 3: Regrowth with SA on the best found
    if best is not None and time.time() < deadline - 10:
        regrown = _regrow_sa_fast(sequence, best, rng, max_iter=min(400, max(50, 1500 // n)))
        if regrown is not None:
            regrown, regrown_energy = _local_search_sa(sequence, regrown, max_iter=min(1000, max(100, 2000 // n)), use_kinks=True, use_sa=True)
            if regrown_energy < best_energy:
                best, best_energy = regrown, regrown_energy
    
    # Phase 4: If time remains, do another round of SA local search on best
    if best is not None and time.time() < deadline - 5:
        remaining_iter = min(1000, max(100, int((deadline - time.time()) * 200)))
        best, best_energy = _local_search_sa(sequence, best, max_iter=remaining_iter, use_kinks=True, use_sa=True)
    
    if best is None:
        best = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
