"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds to keep before hard limit


def _setup_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 235.0  # 235s budget, leaving 5s margin


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield 6 neighbors on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, rng, beam_width=50, max_time=10.0):
    """Beam search with dynamic width and improved heuristic."""
    start = (0,) * dimension
    # Beam entry: (chain_list, occupied_dict, current_energy)
    beams = [([start], {start: 0}, 0)]
    n = len(sequence)
    
    t0 = time.time()
    for i in range(1, n):
        if time.time() - t0 > max_time:
            break
        
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            idx = occupied[q]
                            if idx < i - 1 and sequence[idx] == "H":
                                gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Look-ahead: count H residues in next few positions that could contact
                lookahead = 0
                if sequence[i] == "H":
                    for j in range(i + 2, min(i + 5, n)):
                        if sequence[j] == "H":
                            lookahead += 1
                # Heuristic: prefer high gain, then free space, then lookahead potential
                score = gain * 100 + free * 10 + lookahead * 5 + rng.random()
                options.append((score, site, gain))
            
            if not options:
                continue
            
            options.sort(key=lambda x: -x[0])
            # Keep top candidates per beam
            keep = min(beam_width, max(1, beam_width // 2))
            for score, site, gain in options[:keep]:
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_energy = energy - gain
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    best = min(beams, key=lambda x: x[2])
    return best[0]


def _local_search(sequence, chain, max_iters=2000, seed=42):
    """Improve a conformation via 180-degree turns, 3-residue shifts, and pivot moves with incremental energy."""
    n = len(chain)
    if n < 4:
        return chain
    
    rng = random.Random(seed)
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
    
    def _contact_count_at(pos, idx):
        """Count H-H contacts involving residue idx at position pos, excluding chain-adjacent."""
        if sequence[idx] != "H":
            return 0
        cnt = 0
        for q in _neighbours(pos):
            j = occupied.get(q)
            if j is not None and abs(j - idx) > 1:
                if sequence[j] == "H":
                    cnt += 1
        return cnt
    
    def _delta_180(i):
        """Compute energy delta for 180-degree turn at position i."""
        a, b, c = best_chain[i-1], best_chain[i], best_chain[i+1]
        new_a = (2*b[0]-c[0], 2*b[1]-c[1], 2*b[2]-c[2])
        new_c = (2*b[0]-a[0], 2*b[1]-a[1], 2*b[2]-a[2])
        
        if new_a == new_c or new_a == b or new_c == b:
            return None, None, None
        if new_a in occupied and occupied[new_a] != i-1:
            return None, None, None
        if new_c in occupied and occupied[new_c] != i+1:
            return None, None, None
        
        # Remove old contacts for i-1 and i+1
        old_cnt = 0
        for idx, pos in [(i-1, a), (i+1, c)]:
            if sequence[idx] == "H":
                for q in _neighbours(pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - idx) > 1 and j != i-1 and j != i+1:
                        if sequence[j] == "H":
                            old_cnt += 1
        
        # Add new contacts for i-1 and i+1 at new positions
        new_cnt = 0
        for idx, pos in [(i-1, new_a), (i+1, new_c)]:
            if sequence[idx] == "H":
                for q in _neighbours(pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - idx) > 1 and j != i-1 and j != i+1:
                        if sequence[j] == "H":
                            new_cnt += 1
        
        # Check if new_a and new_c are adjacent to each other (both H, non-adjacent in chain)
        if sequence[i-1] == "H" and sequence[i+1] == "H" and abs((i-1) - (i+1)) > 1:
            if manhattan(new_a, new_c) == 1:
                new_cnt += 1
        
        delta = -(new_cnt - old_cnt)
        return delta, new_a, new_c
    
    def _delta_3shift(i):
        """Compute energy delta for 3-residue shift at position i."""
        a, b, c = best_chain[i-1], best_chain[i], best_chain[i+1]
        for axis in range(3):
            for step in (-1, 1):
                na = (a[0]+(step if axis==0 else 0), a[1]+(step if axis==1 else 0), a[2]+(step if axis==2 else 0))
                nb = (b[0]+(step if axis==0 else 0), b[1]+(step if axis==1 else 0), b[2]+(step if axis==2 else 0))
                nc = (c[0]+(step if axis==0 else 0), c[1]+(step if axis==1 else 0), c[2]+(step if axis==2 else 0))
                
                if na in occupied and occupied[na] != i-1:
                    continue
                if nb in occupied and occupied[nb] != i:
                    continue
                if nc in occupied and occupied[nc] != i+1:
                    continue
                if manhattan(na, nb) != 1 or manhattan(nb, nc) != 1:
                    continue
                
                # Compute delta
                old_cnt = 0
                for idx, pos in [(i-1, a), (i, b), (i+1, c)]:
                    if sequence[idx] == "H":
                        for q in _neighbours(pos):
                            j = occupied.get(q)
                            if j is not None and abs(j - idx) > 1 and j not in (i-1, i, i+1):
                                if sequence[j] == "H":
                                    old_cnt += 1
                
                new_cnt = 0
                for idx, pos in [(i-1, na), (i, nb), (i+1, nc)]:
                    if sequence[idx] == "H":
                        for q in _neighbours(pos):
                            j = occupied.get(q)
                            if j is not None and abs(j - idx) > 1 and j not in (i-1, i, i+1):
                                if sequence[j] == "H":
                                    new_cnt += 1
                
                # Check internal contacts among moved residues
                pairs = [(i-1, i), (i, i+1), (i-1, i+1)]
                for ia, ib in pairs:
                    if abs(ia - ib) <= 1:
                        continue
                    if sequence[ia] == "H" and sequence[ib] == "H":
                        pa = na if ia == i-1 else (nb if ia == i else nc)
                        pb = na if ib == i-1 else (nb if ib == i else nc)
                        if manhattan(pa, pb) == 1:
                            new_cnt += 1
                
                delta = -(new_cnt - old_cnt)
                return delta, na, nb, nc
        return None, None, None, None
    
    def _delta_pivot(i):
        """Compute energy delta for pivot move at bond i-i+1."""
        a, b = best_chain[i], best_chain[i+1]
        best_delta = None
        best_new_b = None
        
        # Try all 5 alternative positions for b (excluding current and position opposite to a)
        for axis in range(3):
            for step in (-1, 1):
                nb = (b[0]+(step if axis==0 else 0), b[1]+(step if axis==1 else 0), b[2]+(step if axis==2 else 0))
                if nb == b:
                    continue
                if nb in occupied and occupied[nb] != i+1:
                    continue
                if manhattan(a, nb) != 1:
                    continue
                if i + 2 < n:
                    if manhattan(nb, best_chain[i+2]) != 1:
                        continue
                
                # Compute delta
                old_cnt = 0
                if sequence[i+1] == "H":
                    for q in _neighbours(b):
                        j = occupied.get(q)
                        if j is not None and abs(j - (i+1)) > 1:
                            if sequence[j] == "H":
                                old_cnt += 1
                
                new_cnt = 0
                if sequence[i+1] == "H":
                    for q in _neighbours(nb):
                        j = occupied.get(q)
                        if j is not None and abs(j - (i+1)) > 1:
                            if sequence[j] == "H":
                                new_cnt += 1
                
                delta = -(new_cnt - old_cnt)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_new_b = nb
        
        return best_delta, best_new_b
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        improved = False
        
        for _attempt in range(40):
            move_type = rng.random()
            i = rng.randint(1, n - 2)
            
            if move_type < 0.45:
                # 180-degree turn
                delta, new_a, new_c = _delta_180(i)
                if delta is None:
                    continue
                if delta < 0:
                    old_a, old_c = best_chain[i-1], best_chain[i+1]
                    best_chain[i-1] = new_a
                    best_chain[i+1] = new_c
                    occupied.pop(old_a)
                    occupied.pop(old_c)
                    occupied[new_a] = i-1
                    occupied[new_c] = i+1
                    best_energy += delta
                    improved = True
            elif move_type < 0.75:
                # 3-residue shift
                delta, na, nb, nc = _delta_3shift(i)
                if delta is None:
                    continue
                if delta < 0:
                    old_a, old_b, old_c = best_chain[i-1], best_chain[i], best_chain[i+1]
                    best_chain[i-1] = na
                    best_chain[i] = nb
                    best_chain[i+1] = nc
                    occupied.pop(old_a)
                    occupied.pop(old_b)
                    occupied.pop(old_c)
                    occupied[na] = i-1
                    occupied[nb] = i
                    occupied[nc] = i+1
                    best_energy += delta
                    improved = True
            else:
                # Pivot move
                if i < 1 or i > n - 3:
                    continue
                delta, new_b = _delta_pivot(i)
                if delta is None or new_b is None:
                    continue
                if delta < 0:
                    old_b = best_chain[i+1]
                    best_chain[i+1] = new_b
                    occupied.pop(old_b)
                    occupied[new_b] = i+1
                    best_energy += delta
                    improved = True
        
        if not improved:
            if it > 10:
                break
    
    return best_chain


def _spiral_fold(sequence, dimension, seed=0):
    """Create a spiral-like initial conformation."""
    n = len(sequence)
    rng = random.Random(seed)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    dir_idx = rng.randint(0, 5)
    
    for i in range(1, n):
        placed = False
        # Try preferred direction first, then others
        order = [dir_idx % 6] + [d for d in range(6) if d != dir_idx % 6]
        rng.shuffle(order[1:])
        for d_idx in order:
            d = directions[d_idx % 6]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                # Occasionally turn
                if rng.random() < 0.3:
                    dir_idx = (d_idx + 1) % 6
                break
        
        if not placed:
            # Stuck, try all 6 directions
            for d in directions:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    break
            else:
                break
    
    return chain


def _simulated_annealing(sequence, chain, max_iters=5000, seed=42):
    """Simulated annealing with pivot moves and 180-degree turns."""
    n = len(chain)
    if n < 4:
        return chain
    
    rng = random.Random(seed)
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t0 = time.time()
    t_max = _time_left() * 0.8
    if t_max < 1.0:
        return best_chain
    
    t = 2.0
    cooling = 0.9995
    
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
    
    def _delta_180(i):
        a, b, c = current_chain[i-1], current_chain[i], current_chain[i+1]
        new_a = (2*b[0]-c[0], 2*b[1]-c[1], 2*b[2]-c[2])
        new_c = (2*b[0]-a[0], 2*b[1]-a[1], 2*b[2]-a[2])
        
        if new_a == new_c or new_a == b or new_c == b:
            return None, None, None
        if new_a in occupied and occupied[new_a] != i-1:
            return None, None, None
        if new_c in occupied and occupied[new_c] != i+1:
            return None, None, None
        
        old_cnt = 0
        for idx, pos in [(i-1, a), (i+1, c)]:
            if sequence[idx] == "H":
                for q in _neighbours(pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - idx) > 1 and j != i-1 and j != i+1:
                        if sequence[j] == "H":
                            old_cnt += 1
        
        new_cnt = 0
        for idx, pos in [(i-1, new_a), (i+1, new_c)]:
            if sequence[idx] == "H":
                for q in _neighbours(pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - idx) > 1 and j != i-1 and j != i+1:
                        if sequence[j] == "H":
                            new_cnt += 1
        
        if sequence[i-1] == "H" and sequence[i+1] == "H":
            if manhattan(new_a, new_c) == 1:
                new_cnt += 1
        
        delta = -(new_cnt - old_cnt)
        return delta, new_a, new_c
    
    def _delta_pivot(i):
        a, b = current_chain[i], current_chain[i+1]
        best_delta = None
        best_new_b = None
        
        for axis in range(3):
            for step in (-1, 1):
                nb = (b[0]+(step if axis==0 else 0), b[1]+(step if axis==1 else 0), b[2]+(step if axis==2 else 0))
                if nb == b:
                    continue
                if nb in occupied and occupied[nb] != i+1:
                    continue
                if manhattan(a, nb) != 1:
                    continue
                if i + 2 < n:
                    if manhattan(nb, current_chain[i+2]) != 1:
                        continue
                
                old_cnt = 0
                if sequence[i+1] == "H":
                    for q in _neighbours(b):
                        j = occupied.get(q)
                        if j is not None and abs(j - (i+1)) > 1:
                            if sequence[j] == "H":
                                old_cnt += 1
                
                new_cnt = 0
                if sequence[i+1] == "H":
                    for q in _neighbours(nb):
                        j = occupied.get(q)
                        if j is not None and abs(j - (i+1)) > 1:
                            if sequence[j] == "H":
                                new_cnt += 1
                
                delta = -(new_cnt - old_cnt)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_new_b = nb
        
        return best_delta, best_new_b
    
    for it in range(max_iters):
        if time.time() - t0 > t_max:
            break
        if _time_left() < 1.0:
            break
        
        i = rng.randint(1, n - 2)
        move_type = rng.random()
        
        if move_type < 0.5:
            delta, new_a, new_c = _delta_180(i)
            if delta is None:
                continue
            if delta < 0 or rng.random() < _math.exp(-delta / max(t, 0.01)):
                old_a, old_c = current_chain[i-1], current_chain[i+1]
                current_chain[i-1] = new_a
                current_chain[i+1] = new_c
                occupied.pop(old_a)
                occupied.pop(old_c)
                occupied[new_a] = i-1
                occupied[new_c] = i+1
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        else:
            if i < 1 or i > n - 3:
                continue
            delta, new_b = _delta_pivot(i)
            if delta is None or new_b is None:
                continue
            if delta < 0 or rng.random() < _math.exp(-delta / max(t, 0.01)):
                old_b = current_chain[i+1]
                current_chain[i+1] = new_b
                occupied.pop(old_b)
                occupied[new_b] = i+1
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        t *= cooling
    
    return best_chain


import math as _math


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice using multiple strategies."""
    _setup_deadline()
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Multiple spiral initializations
    for seed in range(5):
        if _time_left() < 3.0:
            break
        chain1 = _spiral_fold(sequence, dimension, seed=seed)
        if len(chain1) == n:
            e1 = _compute_energy(sequence, chain1)
            if best_chain is None or e1 < best_energy:
                best_chain = chain1
                best_energy = e1
    
    # Strategy 2: Beam search with different widths
    if _time_left() > 5.0:
        for bw in [30, 80, 150]:
            if _time_left() < 3.0:
                break
            max_t = min(8.0, _time_left() * 0.25)
            chain2 = _beam_search(sequence, dimension, rng, beam_width=bw, max_time=max_t)
            if chain2 and len(chain2) == n:
                e2 = _compute_energy(sequence, chain2)
                if best_chain is None or e2 < best_energy:
                    best_chain = chain2
                    best_energy = e2
    
    # Strategy 3: Local search refinement on best chain
    if best_chain is not None and _time_left() > 2.0:
        for ls_seed in range(8):
            if _time_left() < 2.0:
                break
            max_iters = min(6000, int(_time_left() * 150))
            refined = _local_search(sequence, best_chain, max_iters=max_iters, seed=ls_seed * 7 + n)
            if refined and len(refined) == n:
                er = _compute_energy(sequence, refined)
                if er < best_energy:
                    best_chain = refined
                    best_energy = er
    
    # Strategy 4: Simulated annealing for further refinement
    if best_chain is not None and _time_left() > 3.0:
        for sa_seed in range(4):
            if _time_left() < 2.0:
                break
            max_iters = min(8000, int(_time_left() * 200))
            sa_refined = _simulated_annealing(sequence, best_chain, max_iters=max_iters, seed=sa_seed * 13 + n)
            if sa_refined and len(sa_refined) == n:
                esa = _compute_energy(sequence, sa_refined)
                if esa < best_energy:
                    best_chain = sa_refined
                    best_energy = esa
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
