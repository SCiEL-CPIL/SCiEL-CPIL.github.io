"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 180.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _greedy_grow_from(suffix_sequence, start_pos, prefix_chain, rng, temperature=0.0):
    """
    Greedy growth starting from start_pos (end of prefix), avoiding prefix sites.
    Returns the new suffix as a list of tuples (excluding start_pos).
    """
    if not suffix_sequence:
        return []
    
    # Build occupied set from prefix
    occupied = {p: i for i, p in enumerate(prefix_chain)}
    # Mark start_pos as occupied (it's the last prefix position)
    # We'll track the suffix chain separately
    chain = [start_pos]
    # The suffix indices in the full chain start at len(prefix_chain)
    offset = len(prefix_chain) - 1  # start_pos is at this index in full chain
    
    for i, res in enumerate(suffix_sequence):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and sequence_at_index(prefix_chain, suffix_sequence, j, offset) == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = offset + 1 + i
        chain.append(best_site)

    return chain[1:]


def sequence_at_index(prefix_chain, suffix_sequence, idx, offset):
    """Get the residue at global index idx given prefix and suffix."""
    prefix_len = len(prefix_chain)
    if idx < prefix_len:
        return prefix_chain[idx] if isinstance(prefix_chain[idx], str) else None
    # This won't work because prefix_chain contains tuples, not residues
    # We need to pass the full sequence instead
    return None


def _regrow_sa(sequence, chain, rng, max_iters=2000, start_temp=3.0, cool_rate=0.997):
    """
    Simulated annealing using regrow moves: pick a split point, regrow the suffix.
    This can make larger changes than single-residue moves.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0

    for it in range(max_iters):
        if time.time() > _DEADLINE - 15:
            break
        if temp < 0.05:
            break

        # Pick a random split point (keep at least 3 residues in prefix, 2 in suffix)
        min_split = 3
        max_split = max(min_split, n - 3)
        if max_split <= min_split:
            break
        split = rng.randint(min_split, max_split)
        
        prefix = chain[:split]
        suffix_seq = sequence[split:]
        
        # Regrow suffix from the end of prefix
        new_suffix = _greedy_grow_from_fixed(suffix_seq, prefix[-1], prefix, sequence, rng, temperature=temp)
        if new_suffix is None:
            temp *= cool_rate
            no_improve += 1
            continue
        
        new_chain = prefix + new_suffix
        new_energy = _compute_energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        temp *= cool_rate
        
        if no_improve > 300:
            break

    return best_chain, best_energy


def _greedy_grow_from_fixed(suffix_seq, start_pos, prefix, full_sequence, rng, temperature=0.0):
    """
    Greedy growth for suffix, with proper access to full sequence for H checks.
    """
    if not suffix_seq:
        return []
    
    # Build occupied set from prefix with their global indices
    occupied = {p: i for i, p in enumerate(prefix)}
    chain = [start_pos]
    
    for i, res in enumerate(suffix_seq):
        global_idx = len(prefix) + i
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < global_idx - 1 and full_sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = global_idx
        chain.append(best_site)

    return chain[1:]


def _local_improve(sequence, chain, rng, max_moves=5000):
    """
    Improve conformation via single-residue moves that preserve connectivity.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 10:
            break
        
        i = rng.randint(1, n - 2)
        p = chain[i]
        tried = False

        for np in _neighbours(p):
            if np == p:
                continue
            if np in occupied and occupied[np] != i:
                continue
            
            # Check if moving to np preserves connectivity
            prev, nxt = chain[i-1], chain[i+1]
            dist_prev = abs(np[0]-prev[0]) + abs(np[1]-prev[1])
            dist_next = abs(np[0]-nxt[0]) + abs(np[1]-nxt[1])
            
            if dist_prev != 1 or dist_next != 1:
                continue
            
            # Perform move
            del occupied[p]
            chain[i] = np
            occupied[np] = i
            new_energy = _compute_energy(sequence, chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                no_improve_count = 0
            else:
                # Revert
                del occupied[np]
                chain[i] = p
                occupied[p] = i
                no_improve_count += 1
            
            tried = True
            break
        
        if not tried:
            no_improve_count += 1
        
        if no_improve_count > 500:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using a portfolio of greedy growths with different random seeds
    and temperatures, plus regrow-SA and local improvement. Keep the best within the time budget.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)

    n = len(sequence)
    
    # Time allocation: ~55% greedy restarts, ~25% regrow-SA, ~20% local improvement
    # Total budget: 180s, leave 15s safety margin => 165s usable
    # Phase 1: 0 to ~90s, Phase 2: ~90s to ~130s, Phase 3: ~130s to ~165s
    
    # Phase 1: Multiple greedy restarts with varying temperatures
    phase1_end = time.time() + 90.0
    for attempt in range(5000):
        if time.time() > phase1_end:
            break

        temp = rng.uniform(0.0, 3.0)
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue

        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Regrow-based simulated annealing on the best found
    if best_chain is not None:
        phase2_end = time.time() + 40.0
        sa_chain, sa_energy = _regrow_sa(
            sequence, best_chain, rng, max_iters=5000, start_temp=3.0, cool_rate=0.997)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy

    # Phase 3: Local improvement on the best found
    if best_chain is not None:
        phase3_end = time.time() + 30.0
        improved_chain, improved_energy = _local_improve(
            sequence, best_chain, rng, max_moves=10000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
