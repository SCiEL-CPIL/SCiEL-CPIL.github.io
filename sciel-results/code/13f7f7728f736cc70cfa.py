"""HP lattice protein folding - incremental energy deltas + simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((p[0] + dx, p[1] + dy))
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta_reverse(sequence, chain, i, j):
    """
    Return (delta_energy, is_valid) for reversing chain[i..j].
    delta_energy = new_energy - old_energy (negative = improvement).
    is_valid = False if boundary connections are broken.
    """
    n = len(chain)
    
    # Check boundary connections
    if i > 0:
        a, b = chain[i-1], chain[j]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return 0, False
    if j < n - 1:
        a, b = chain[i], chain[j+1]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return 0, False
    
    # Build position to index map
    pos_to_idx = {p: idx for idx, p in enumerate(chain)}
    
    # Old contacts involving at least one position in segment [i, j]
    old_c = 0
    for k in range(i, j+1):
        p = chain[k]
        if sequence[k] != "H":
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + dx, p[1] + dy)
            j_idx = pos_to_idx.get(q)
            if j_idx is not None and j_idx != k and abs(j_idx - k) > 1 and sequence[j_idx] == "H":
                old_c += 1
    
    # New contacts: position chain[k] moves to index (i+j-k) for k in [i, j]
    new_c = 0
    for k in range(i, j+1):
        p = chain[k]
        if sequence[k] != "H":
            continue
        new_k = i + j - k  # new index of position p
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + dx, p[1] + dy)
            j_idx = pos_to_idx.get(q)
            if j_idx is None:
                continue
            # New index of q
            if i <= j_idx <= j:
                new_j = i + j - j_idx
            else:
                new_j = j_idx
            if new_j != new_k and abs(new_j - new_k) > 1 and sequence[new_j] == "H":
                new_c += 1
    
    # Each contact counted twice (once from each endpoint)
    delta = (new_c - old_c) // 2
    return delta, True


def _energy_delta_rotate(sequence, chain, i, j, angle):
    """
    Return (delta_energy, new_chain, is_valid) for rotating chain[i..j] by angle around chain[i].
    """
    n = len(chain)
    center = chain[i]
    
    # Compute new positions
    new_positions = []
    for k in range(i, j+1):
        dx = chain[k][0] - center[0]
        dy = chain[k][1] - center[1]
        if angle == 1:  # 90 deg CCW
            nx, ny = -dy, dx
        elif angle == 2:  # 180 deg
            nx, ny = -dx, -dy
        else:  # 270 deg CCW
            nx, ny = dy, -dx
        new_positions.append((center[0] + nx, center[1] + ny))
    
    # Check self-avoidance
    outside_positions = set(chain[:i]) | set(chain[j+1:])
    seen = set()
    for p in new_positions:
        if p in seen or p in outside_positions:
            return 0, None, False
        seen.add(p)
    
    # Check boundary connections
    if i > 0:
        a, b = chain[i-1], new_positions[-1]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return 0, None, False
    if j < n - 1:
        a, b = new_positions[0], chain[j+1]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return 0, None, False
    
    new_chain = chain[:i] + new_positions + chain[j+1:]
    
    # Compute energy delta
    pos_to_idx_old = {p: idx for idx, p in enumerate(chain)}
    pos_to_idx_new = {p: idx for idx, p in enumerate(new_chain)}
    
    old_c = 0
    for k in range(i, j+1):
        p = chain[k]
        if sequence[k] != "H":
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + dx, p[1] + dy)
            j_idx = pos_to_idx_old.get(q)
            if j_idx is not None and j_idx != k and abs(j_idx - k) > 1 and sequence[j_idx] == "H":
                old_c += 1
    
    new_c = 0
    for k in range(i, j+1):
        p = new_positions[k - i]
        if sequence[k] != "H":
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + dx, p[1] + dy)
            j_idx = pos_to_idx_new.get(q)
            if j_idx is not None and j_idx != k and abs(j_idx - k) > 1 and sequence[j_idx] == "H":
                new_c += 1
    
    delta = (new_c - old_c) // 2
    return delta, new_chain, True


def _grow(sequence, dimension, rng, bias=0.5):
    """One randomized greedy chain growth with adjustable bias."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (chain[-1][0] + dx, chain[-1][1] + dy)
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                for ndx, ndy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (site[0] + ndx, site[1] + ndy)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            free = sum(1 for ndx, ndy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                       if (site[0] + ndx, site[1] + ndy) not in occupied)
            
            contact_w = 1.0 + bias * 2.0
            free_w = 1.0 - bias * 0.5
            score = contact_w * gain + free_w * 0.1 * free + rng.random() * 0.3
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: -x[0])
        if len(options) > 1 and rng.random() < 0.3:
            site = options[1][1]
        else:
            site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _reverse_move_fast(chain, i, j):
    """Perform reverse move, return new chain or None if invalid."""
    n = len(chain)
    if i > 0:
        a, b = chain[i-1], chain[j]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return None
    if j < n - 1:
        a, b = chain[i], chain[j+1]
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1:
            return None
    return chain[:i] + chain[i:j+1][::-1] + chain[j+1:]


def _anneal(sequence, chain, rng, time_budget, initial_temp=5.0):
    """Simulated annealing with incremental energy deltas."""
    start_time = time.time()
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    temperature = initial_temp
    cooling = 0.997
    min_temp = 0.1
    
    while time.time() - start_time < time_budget and temperature > min_temp:
        move_type = rng.random()
        
        if move_type < 0.8:
            max_len = min(n - 1, max(3, int(n * 0.25)))
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, min(i + max_len, n - 1))
            
            new_chain = _reverse_move_fast(chain, i, j)
            if new_chain is None:
                temperature *= cooling
                continue
            
            delta, valid = _energy_delta_reverse(sequence, chain, i, j)
            if not valid:
                temperature *= cooling
                continue
        else:
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, min(i + 8, n - 1))
            angle = rng.choice([1, 2, 3])
            
            delta, new_chain, valid = _energy_delta_rotate(sequence, chain, i, j, angle)
            if not valid:
                temperature *= cooling
                continue
        
        if delta < 0 or (temperature > 0.15 and rng.random() < exp(-delta / temperature)):
            chain = new_chain
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        temperature *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using diverse greedy growth followed by simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    total_budget = min(_remaining_time() - 5.0, 180.0)
    
    if total_budget < 2.0:
        best, best_energy = None, 100
        for _ in range(20):
            chain = _grow(sequence, dimension, rng, bias=0.7)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best, best_energy = chain, value
        if best is None:
            best = [(i, 0) for i in range(n)]
        return [list(p) for p in best]
    
    # Generate diverse initial chains
    num_growths = min(40, max(10, n // 2))
    biases = [0.3, 0.5, 0.7, 0.9]
    candidates = []
    for _ in range(num_growths):
        bias = biases[rng.randint(0, len(biases) - 1)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
    
    if not candidates:
        best = [(i, 0) for i in range(n)]
        return [list(p) for p in best]
    
    candidates.sort(key=lambda x: x[0])
    
    # Anneal top candidates
    num_to_anneal = min(len(candidates), max(2, int(total_budget / 15.0)))
    time_per_anneal = total_budget / max(1, num_to_anneal)
    
    best_chain = candidates[0][1]
    best_energy = candidates[0][0]
    
    for idx in range(num_to_anneal):
        if _remaining_time() < 8.0:
            break
        e, chain = candidates[idx]
        if e > best_energy + 3 and idx > 0:
            break
        
        anneal_budget = min(time_per_anneal, _remaining_time() - 8.0)
        if anneal_budget < 1.0:
            break
        
        improved_chain, improved_energy = _anneal(
            sequence, chain, rng, anneal_budget,
            initial_temp=3.0 + idx * 0.5
        )
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    # Final polish
    if _remaining_time() > 5.0:
        polish_budget = min(_remaining_time() - 5.0, 3.0)
        best_chain, best_energy = _anneal(
            sequence, best_chain, rng, polish_budget, initial_temp=0.5
        )
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
