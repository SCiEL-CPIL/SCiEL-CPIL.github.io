"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation.
    Energy = - (number of H-H contacts)
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _contact_map(sequence):
    """
    Identify high-priority H-H pairs for the contact map heuristic.
    Returns a list of (i, j) pairs sorted by priority (distance in sequence).
    Only pairs that are not adjacent in the chain are considered.
    """
    n = len(sequence)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    pairs = []
    for idx_i, i in enumerate(h_indices):
        for j in h_indices[idx_i+1:]:
            if abs(i - j) > 1:  # Not adjacent in chain
                pairs.append((i, j))
    # Sort by sequence distance (longer range = higher priority for constraint)
    pairs.sort(key=lambda p: abs(p[0] - p[1]), reverse=True)
    return pairs


def _apply_pivot(chain, pivot_idx, angle):
    """
    Apply a pivot move at pivot_idx with given angle (90, -90, 180).
    Returns None if invalid (collision), else new chain list.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    pivot = chain[pivot_idx]
    fixed = chain[:pivot_idx + 1]
    moving_start = pivot_idx + 1
    segment = chain[moving_start:]
    
    if not segment:
        return chain
    
    fixed_set = set(fixed)
    
    rotated_segment = []
    px, py = pivot
    
    valid = True
    for p in segment:
        dx, dy = p[0] - px, p[1] - py
        
        if angle == 90:
            ndx, ndy = -dy, dx
        elif angle == -90:
            ndx, ndy = dy, -dx
        else:  # 180
            ndx, ndy = -dx, -dy
            
        new_p = (px + ndx, py + ndy)
        
        if new_p in fixed_set:
            valid = False
            break
            
        rotated_segment.append(new_p)
        
    if not valid:
        return None
        
    return fixed + rotated_segment


def _apply_slither(chain, rng):
    """
    Slither move: Move the last residue to a different valid neighbor of the second-to-last.
    """
    n = len(chain)
    if n < 2:
        return None
    
    prev = chain[-2]
    last = chain[-1]
    occupied = set(chain[:-1])
    
    candidates = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        site = (prev[0] + dx, prev[1] + dy)
        if site != last and site not in occupied:
            candidates.append(site)
            
    if not candidates:
        return None
        
    new_last = rng.choice(candidates)
    return chain[:-1] + [new_last]


def _apply_crankshaft(chain, start_idx, length, angle, rng):
    """
    Crankshaft move: Rotate a segment of the chain around two pivots.
    This is more complex and less likely to cause collisions than long pivots.
    """
    n = len(chain)
    if start_idx < 1 or start_idx + length >= n:
        return None
    
    # Pivots are at start_idx-1 and start_idx+length
    pivot1 = chain[start_idx - 1]
    pivot2 = chain[start_idx + length]
    
    fixed_left = chain[:start_idx]
    fixed_right = chain[start_idx + length:]
    segment = chain[start_idx:start_idx + length]
    
    if not segment:
        return chain
    
    fixed_set = set(fixed_left) | set(fixed_right)
    
    # Rotation around pivot1
    px, py = pivot1
    rotated_segment = []
    
    # For crankshaft, we rotate the segment relative to pivot1,
    # but we must ensure the end connects to pivot2.
    # Actually, standard crankshaft rotates the segment around the axis defined by the two pivots?
    # No, in 2D, crankshaft usually means rotating the segment around one end,
    # but keeping the other end fixed is impossible unless the length matches.
    # Let's stick to simpler pivot moves for reliability, or implement a "segment move"
    # where we move the whole segment to a new position.
    # Given time constraints, let's skip complex crankshaft and rely on pivots + slithers.
    # We can add a "segment shift" later if needed.
    return None


def _grow_greedy_contact_guided(sequence, bias, rng, contact_pairs):
    """
    Greedy growth guided by contact map priorities.
    """
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Pre-compute which residues are involved in high-priority contacts
    # and their target partners
    # For each residue i, we want to be close to its contact partners if possible.
    
    for i in range(1, n):
        curr = chain[-1]
        options = []
        
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            site = (curr[0] + dx, curr[1] + dy)
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                sx, sy = site
                for ndx, ndy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    q = (sx + ndx, sy + ndy)
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            
            free = 0
            for ndx, ndy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                q = (site[0] + ndx, site[1] + ndy)
                if q not in occupied:
                    free += 1
            
            # Contact map bonus: if this residue i is part of a high-priority pair
            # with a previously placed residue j, and we are close to j, bonus.
            # This is a heuristic to enforce spatial proximity.
            contact_bonus = 0
            if sequence[i] == 'H':
                # Check if i is in contact_pairs with any previously placed H
                for (pi, pj) in contact_pairs[:10]:  # Top 10 priority
                    if pi == i:
                        # Partner is pj, which must be < i to be already placed
                        if pj < i and sequence[pj] == 'H':
                            # Distance to partner
                            partner_pos = chain[pj]
                            d = abs(site[0] - partner_pos[0]) + abs(site[1] - partner_pos[1])
                            if d == 2:  # Close enough to potentially form a contact later
                                contact_bonus += 1
                    elif pj == i:
                        # Partner is pi, which must be < i
                        if pi < i and sequence[pi] == 'H':
                            partner_pos = chain[pi]
                            d = abs(site[0] - partner_pos[0]) + abs(site[1] - partner_pos[1])
                            if d == 2:
                                contact_bonus += 1
            
            score = bias * gain + 0.5 * contact_bonus - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))
            
        if not options:
            return None
            
        best_site = max(options, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def _local_search(sequence, chain, time_limit, rng):
    """
    Simulated Annealing local search.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(current_chain)
    
    T = 5.0
    T_min = 0.01
    
    if n > 80:
        cooling = 0.9999
    elif n > 50:
        cooling = 0.9997
    else:
        cooling = 0.9995
        
    max_iters = 10000
    
    start_time = time.time()
    
    for iter in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break
            
        r = rng.random()
        new_chain = None
        
        if r < 0.8:
            pivot_idx = rng.randint(1, n - 2)
            angle = rng.choice([90, -90, 180])
            new_chain = _apply_pivot(current_chain, pivot_idx, angle)
        else:
            new_chain = _apply_slither(current_chain, rng)
            
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            if T > T_min:
                exp_val = math.exp(-delta / T)
                if rng.random() < exp_val:
                    current_chain = new_chain
                    current_energy = new_energy
        
        T *= cooling
        
    return best_chain


def _polish(sequence, chain, time_limit, rng):
    """
    Deterministic hill-climbing polish: accept only improving moves.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    n = len(current_chain)
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit and _time_left() > 15:
        improved = False
        for pivot_idx in range(1, n - 1):
            for angle in (90, -90, 180):
                new_chain = _apply_pivot(current_chain, pivot_idx, angle)
                if new_chain:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        improved = True
                        break
            if improved:
                break
        # Also try slither
        if not improved:
            new_chain = _apply_slither(current_chain, rng)
            if new_chain:
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    
    return current_chain


def fold(sequence, dimension):
    """
    Main entry point.
    """
    seq_len = len(sequence)
    remaining = _time_left()
    
    if remaining < 5:
        return [[i, 0] for i in range(seq_len)]
        
    # Time allocation
    if remaining < 20:
        time_for_init = 1.0
        time_for_search = 5.0
    else:
        time_for_init = 2.0
        time_for_search = 8.0
        
    rng = random.Random(12345 + seq_len)
    
    best_chain = None
    best_energy = 0
    
    # Pre-compute contact map for heuristic guidance
    contact_pairs = _contact_map(sequence)
    
    # Phase 1: Initialization
    init_start = time.time()
    candidates = []
    
    num_greedy = 20
    for i in range(num_greedy):
        if time.time() - init_start > time_for_init:
            break
        bias = 0.5 + rng.random() * 2.0
        chain = _grow_greedy_contact_guided(sequence, bias, rng, contact_pairs)
        if chain:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
            
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:3]
    
    if not top_candidates:
        best_chain = [[i, 0] for i in range(seq_len)]
        best_energy = _energy(sequence, best_chain)
        return best_chain
        
    # Phase 2: Local Search on top candidates
    search_start = time.time()
    
    for e, chain in top_candidates:
        if time.time() - search_start > time_for_search:
            break
        if _time_left() < 15:
            break
            
        remaining_search = time_for_search - (time.time() - search_start)
        if remaining_search < 0.5:
            break
            
        improved = _local_search(sequence, chain, remaining_search * 0.9, rng)
        
        if improved:
            e_imp = _energy(sequence, improved)
            if e_imp < best_energy or best_chain is None:
                best_energy = e_imp
                best_chain = improved

    # Phase 3: Final Polish
    if best_chain and _time_left() > 15:
        polish_time = min(2.0, _time_left() - 15)
        polished = _polish(sequence, best_chain, polish_time, rng)
        if polished:
            e_pol = _energy(sequence, polished)
            if e_pol < best_energy:
                best_energy = e_pol
                best_chain = polished

    if best_chain is None:
        best_chain = [[i, 0] for i in range(seq_len)]
        
    return best_chain
# EVOLVE-BLOCK-END
