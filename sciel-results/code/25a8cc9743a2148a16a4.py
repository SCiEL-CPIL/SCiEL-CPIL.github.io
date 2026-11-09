# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # Safety margin before 330s kill


def _neighbours_3d(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _compute_energy(sequence, chain):
    """Compute total energy: -1 * (number of H-H non-consecutive contacts)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours_3d(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1:
                    # Only count each pair once
                    if j > i:
                        if sequence[j] == 'H':
                            contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy chain growth preferring sites with new H-H contacts."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours_3d(last):
            if site in occupied:
                continue
            # Count new H-H contacts if we place residue i at site
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours_3d(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Small random perturbation for diversity
            score = gain + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        best_score, best_site = max(options, key=lambda x: x[0])
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _local_search(sequence, chain, max_iter=5000, rng=None):
    """Local search using pivot and slide moves with simulated annealing."""
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n <= 2:
        return [list(p) for p in chain]
    
    # Build occupied set and position lookup
    occupied = set(chain)
    pos_map = {p: i for i, p in enumerate(chain)}
    
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    T = 1.0
    T_end = 0.01
    cooling = 0.999
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Pivot move: pick a position, rotate a sub-chain
            if n < 3:
                continue
            pos = rng.randint(0, n - 2)
            # Pick a segment to pivot
            seg_len = rng.randint(1, min(4, n - pos - 1))
            if seg_len < 1:
                continue
            
            # The pivot point is at pos, and we try to move residue at pos+seg_len
            # Actually, let's do a simpler pivot: pick two non-adjacent positions
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if abs(i - j) <= 1:
                continue
            
            # Try swapping is not valid for chain connectivity
            # Instead, do a "pull" move: pick a position and try to move it to an adjacent free site
            # that maintains connectivity
            pos = i
            if pos == 0 or pos == n - 1:
                continue
            
            current_pos = chain[pos]
            prev_pos = chain[pos - 1]
            next_pos = chain[pos + 1]
            
            # Valid moves: the new position must be adjacent to both prev and next
            # OR we can do a "slide" where we move pos to a site adjacent to prev or next
            # Simplest: try moving pos to each neighbor of prev or next that is free
            
            candidates = set()
            for nb in _neighbours_3d(prev_pos):
                if nb != current_pos and nb != next_pos and nb not in occupied:
                    # Check if nb is adjacent to next_pos
                    dist = abs(nb[0] - next_pos[0]) + abs(nb[1] - next_pos[1]) + abs(nb[2] - next_pos[2])
                    if dist == 1:
                        candidates.add(nb)
            for nb in _neighbours_3d(next_pos):
                if nb != current_pos and nb != prev_pos and nb not in occupied:
                    dist = abs(nb[0] - prev_pos[0]) + abs(nb[1] - prev_pos[1]) + abs(nb[2] - prev_pos[2])
                    if dist == 1:
                        candidates.add(nb)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(list(candidates))
            
            # Remove old, add new
            occupied.remove(current_pos)
            occupied.add(new_pos)
            old_chain = list(chain)
            chain[pos] = new_pos
            
            new_energy = _compute_energy(sequence, chain)
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain = old_chain
                occupied.remove(new_pos)
                occupied.add(current_pos)
        
        else:
            # Slide move: move a single interior residue to a free adjacent site
            if n < 3:
                continue
            pos = rng.randint(1, n - 2)
            current_pos = chain[pos]
            prev_pos = chain[pos - 1]
            next_pos = chain[pos + 1]
            
            candidates = []
            for nb in _neighbours_3d(current_pos):
                if nb in occupied and nb != prev_pos and nb != next_pos:
                    continue
                if nb in occupied:
                    continue
                # Check connectivity: new pos must be adjacent to prev or next
                # Actually for a valid chain, new_pos must be adjacent to BOTH prev and next
                # That's too restrictive. Let's allow moving to any free neighbor and check connectivity
                # After move, chain[pos] = nb, so prev->nb and nb->next must both be unit steps
                d1 = abs(nb[0] - prev_pos[0]) + abs(nb[1] - prev_pos[1]) + abs(nb[2] - prev_pos[2])
                d2 = abs(nb[0] - next_pos[0]) + abs(nb[1] - next_pos[1]) + abs(nb[2] - next_pos[2])
                if d1 == 1 and d2 == 1:
                    candidates.append(nb)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            occupied.remove(current_pos)
            occupied.add(new_pos)
            old_chain = list(chain)
            chain[pos] = new_pos
            
            new_energy = _compute_energy(sequence, chain)
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                chain = old_chain
                occupied.remove(new_pos)
                occupied.add(current_pos)
        
        T *= cooling
        if T < T_end:
            T = T_end
    
    return [list(p) for p in best_chain]


def fold(sequence, dimension):
    """Fold the HP sequence on the cubic lattice. Returns list of lattice points."""
    if dimension != 3:
        # Fallback for non-3D (shouldn't happen per spec)
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 1: Multiple greedy growths
    num_growths = min(200, max(50, 20000 // n))
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 30:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Straight line fallback
        return [[i, 0, 0] for i in range(n)]
    
    # Phase 2: Local search refinement on best chain
    remaining_time = _DEADLINE - time.time() - 10
    if remaining_time > 5:
        # Allocate time for local search
        ls_iters = int(min(10000, max(1000, remaining_time * 2000)))
        refined = _local_search(sequence, best_chain, max_iter=ls_iters, rng=rng)
        ref_energy = _compute_energy(sequence, refined)
        if ref_energy < best_energy:
            best_energy = ref_energy
            best_chain = refined
    
    return [[p[0], p[1], p[2]] for p in best_chain]
# EVOLVE-BLOCK-END
