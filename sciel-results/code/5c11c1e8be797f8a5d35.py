"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 190.0  # seconds, leaving margin before 210s kill


def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT


def _time_up():
    return time.time() >= _DEADLINE


def _neighbours2d(x, y):
    """Return the 4 unit-step neighbours of (x,y)."""
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """Compute energy = -number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, (x, y) in enumerate(chain):
        pos_to_idx[(x, y)] = i

    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for nx, ny in _neighbours2d(x, y):
            j = pos_to_idx.get((nx, ny))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _fast_energy_delta(sequence, chain, pos_to_idx, i, new_pos, old_pos):
    """
    Compute the change in energy when moving residue i from old_pos to new_pos.
    Only contacts involving residue i or its neighbours can change.
    Returns delta_energy = new_energy - old_energy.
    """
    # Contacts that are removed: those involving old_pos
    removed = 0
    # Check contacts between residue i (at old_pos) and others
    if sequence[i] == 'H':
        for nx, ny in _neighbours2d(old_pos[0], old_pos[1]):
            j = pos_to_idx.get((nx, ny))
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                if j > i:
                    removed += 1
    
    # Also check if any other H residue at old_pos's neighbours had a contact with i
    # (already counted above since we check all neighbours of old_pos)
    
    # Contacts that are added: those involving new_pos
    added = 0
    if sequence[i] == 'H':
        for nx, ny in _neighbours2d(new_pos[0], new_pos[1]):
            j = pos_to_idx.get((nx, ny))
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                if j > i:
                    added += 1
    
    return -(added - removed)


def _greedy_grow(sequence, rng, temperature=0.0):
    """
    Greedy chain growth with optional stochastic element.
    Returns chain (list of (x,y)) or None if stuck.
    """
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}

    for i in range(1, n):
        x, y = chain[-1]
        options = []
        for nx, ny in _neighbours2d(x, y):
            if (nx, ny) in occupied:
                continue
            # Score: number of new H-H contacts this placement would create
            gain = 0
            if sequence[i] == 'H':
                for qx, qy in _neighbours2d(nx, ny):
                    j = occupied.get((qx, qy))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Free space bonus to avoid trapping
            free = 0
            for qx, qy in _neighbours2d(nx, ny):
                if (qx, qy) not in occupied:
                    free += 1
            # Stochastic component
            noise = rng.random() * temperature
            score = gain * 10.0 + free * 0.1 + noise
            options.append((score, (nx, ny)))

        if not options:
            return None

        # Pick best (or stochastically if temperature > 0)
        if temperature > 0 and len(options) > 1:
            # Boltzmann-like selection
            best_score = max(o[0] for o in options)
            weights = []
            for s, _ in options:
                weights.append(1.0 / (1.0 + abs(best_score - s)))
            total = sum(weights)
            r = rng.random() * total
            cum = 0
            chosen = options[-1][1]
            for (s, site), w in zip(options, weights):
                cum += w
                if r <= cum:
                    chosen = site
                    break
        else:
            chosen = max(options)[1]

        chain.append(chosen)
        occupied[chosen] = i

    return chain


def _local_search_fast(sequence, chain, rng, max_steps=2000):
    """
    Local search with incremental energy updates for speed.
    Tries single-residue moves that maintain connectivity.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    current_energy = _compute_energy(sequence, chain)
    chain = list(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    best_energy = current_energy
    best_chain = chain[:]

    for step in range(max_steps):
        if _time_up():
            break
        
        # Pick a random internal residue to try moving
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        for new_pos in _neighbours2d(old_pos[0], old_pos[1]):
            if new_pos in pos_to_idx:
                continue
            
            # Check connectivity: new_pos must be adjacent to both chain[i-1] and chain[i+1]
            px, py = chain[i - 1]
            qx, qy = chain[i + 1]
            if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                continue
            if abs(new_pos[0] - qx) + abs(new_pos[1] - qy) != 1:
                continue
            
            # Compute energy delta using fast method
            delta = _fast_energy_delta(sequence, chain, pos_to_idx, i, new_pos, old_pos)
            new_energy = current_energy + delta
            
            if new_energy <= current_energy:
                # Accept the move
                del pos_to_idx[old_pos]
                chain[i] = new_pos
                pos_to_idx[new_pos] = i
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                break
            else:
                # With small probability, accept worse moves (simulated annealing lite)
                if rng.random() < 0.05:
                    del pos_to_idx[old_pos]
                    chain[i] = new_pos
                    pos_to_idx[new_pos] = i
                    current_energy = new_energy
                    break

    return best_chain, best_energy


def _kink_flip(sequence, chain, rng, max_steps=1000):
    """
    Kink flip: reverse a segment of the chain if it fits in the same bounding box.
    This is a more powerful move than single-residue shifts.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    current_energy = _compute_energy(sequence, chain)
    chain = list(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    best_energy = current_energy
    best_chain = chain[:]

    for step in range(max_steps):
        if _time_up():
            break
        
        # Choose a random segment [i, j] to flip
        # The segment must be short enough to potentially fit
        seg_len = rng.randint(2, min(8, n - 2))
        i = rng.randint(1, n - 1 - seg_len)
        j = i + seg_len - 1
        
        # Get the bounding box of the segment
        seg = chain[i:j+1]
        min_x = min(p[0] for p in seg)
        max_x = max(p[0] for p in seg)
        min_y = min(p[1] for p in seg)
        max_y = max(p[1] for p in seg)
        
        # Try to reverse the segment
        reversed_seg = seg[::-1]
        
        # Check if all positions in reversed_seg are either:
        # 1. Already occupied by the segment itself
        # 2. Free (not occupied by other parts of the chain)
        seg_positions = set(seg)
        valid = True
        for p in reversed_seg:
            if p not in seg_positions:
                if p in pos_to_idx:
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Check connectivity at the boundaries
        # chain[i-1] must be adjacent to reversed_seg[0]
        if i > 0:
            px, py = chain[i-1]
            rx, ry = reversed_seg[0]
            if abs(px - rx) + abs(py - ry) != 1:
                continue
        
        # chain[j+1] must be adjacent to reversed_seg[-1]
        if j < n - 1:
            px, py = chain[j+1]
            rx, ry = reversed_seg[-1]
            if abs(px - rx) + abs(py - ry) != 1:
                continue
        
        # Perform the flip
        # Remove old positions
        for p in seg:
            if p in pos_to_idx and pos_to_idx[p] >= i and pos_to_idx[p] <= j:
                del pos_to_idx[p]
        
        # Add new positions
        for k, p in enumerate(reversed_seg):
            chain[i+k] = p
            pos_to_idx[p] = i+k
        
        new_energy = _compute_energy(sequence, chain)
        
        if new_energy <= current_energy:
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        else:
            # Revert
            for k, p in enumerate(seg):
                chain[i+k] = p
                pos_to_idx[p] = i+k
            # With small probability, keep the flip even if worse
            if rng.random() < 0.02:
                # Keep it
                pass
            else:
                # Already reverted above
                pass

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find a low-energy self-avoiding conformation for the given HP sequence.
    Returns list of len(sequence) integer lattice points (lists of length dimension).
    """
    _init_deadline()
    n = len(sequence)
    if n == 0:
        return []

    rng = random.Random(42 + len(sequence) * 137)

    best_chain = None
    best_energy = 0  # 0 means no valid chain yet (energy is negative or zero)

    # Phase 1: Multiple greedy growths with varying temperatures
    temps = [0.0, 0.1, 0.3, 0.5, 1.0]
    num_greedy = 40

    for t_idx, temp in enumerate(temps):
        for trial in range(num_greedy // len(temps) + 1):
            if _time_up():
                break
            chain = _greedy_grow(sequence, rng, temperature=temp)
            if chain is None:
                continue
            energy = _compute_energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy

    # Phase 2: Local search with fast energy updates on best chains
    if best_chain is not None and not _time_up():
        improved_chain, improved_energy = _local_search_fast(sequence, best_chain, rng, max_steps=3000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Phase 3: Kink flips for more powerful moves
    if best_chain is not None and not _time_up():
        improved_chain, improved_energy = _kink_flip(sequence, best_chain, rng, max_steps=500)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Phase 4: More restarts with local search
    for trial in range(10):
        if _time_up():
            break
        chain = _greedy_grow(sequence, rng, temperature=0.2)
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_chain = chain
            best_energy = energy
        improved_chain, improved_energy = _local_search_fast(sequence, chain, rng, max_steps=1000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 5: Final kink flips on the best chain
    if best_chain is not None and not _time_up():
        improved_chain, improved_energy = _kink_flip(sequence, best_chain, rng, max_steps=300)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    # Ensure output format: list of lists of integers
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
