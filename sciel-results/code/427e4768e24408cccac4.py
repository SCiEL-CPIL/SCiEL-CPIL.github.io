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


def _greedy_grow(sequence, dimension, rng, width=1):
    """Greedy growth with optional beam width."""
    start = (0,) * dimension
    # Each beam entry: (chain, occupied_set, energy_so_far)
    beams = [([start], {start: 0}, 0)]
    
    for i in range(1, len(sequence)):
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
                # Count free neighbors for potential
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 10 + free + rng.random() * 0.1
                options.append((score, site, gain))
            
            if not options:
                continue
            
            options.sort(key=lambda x: -x[0])
            # Keep best width options per beam
            for score, site, gain in options[:width]:
                new_chain = chain + [site]
                new_occupied = occupied | {site: i}
                new_energy = energy - gain  # energy decreases (more negative)
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        # Keep top width beams overall
        new_beams.sort(key=lambda x: x[2])  # most negative energy first
        beams = new_beams[:width]
    
    if not beams:
        return None
    
    # Return best beam
    best = min(beams, key=lambda x: x[2])
    return best[0]


def _beam_search(sequence, dimension, rng, beam_width=50, max_time=10.0):
    """Beam search with dynamic width."""
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
                # Heuristic: prefer high gain, then free space
                score = gain * 100 + free * 10 + rng.random()
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


def _local_search(sequence, chain, max_iters=2000):
    """Improve a conformation via local moves (180-degree turns, shifts)."""
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    def try_move(i, new_pos):
        """Try moving residue i to new_pos."""
        if i == 0 or i == n - 1:
            return None
        # Check self-avoidance
        if new_pos in occupied:
            # Check if it's the current position
            if occupied.get(new_pos) == i:
                return None
            return None
        # Check adjacency constraints
        prev = best_chain[i - 1]
        nxt = best_chain[i + 1]
        if new_pos == prev or new_pos == nxt:
            return None
        # Check unit step from prev
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2]) != 1:
            return None
        # Check unit step from next
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2]) != 1:
            return None
        return new_pos
    
    for _ in range(max_iters):
        improved = False
        # Try random local moves
        for _attempt in range(50):
            i = random.randint(1, n - 2)
            current = best_chain[i]
            # Try shifting to a nearby position
            candidates = []
            for axis in range(3):
                for step in (-1, 1):
                    new_pos = list(current)
                    new_pos[axis] += step
                    new_pos = tuple(new_pos)
                    if try_move(i, new_pos) is not None:
                        candidates.append(new_pos)
            
            if not candidates:
                continue
            
            new_pos = random.choice(candidates)
            # Apply move
            old_pos = best_chain[i]
            best_chain[i] = new_pos
            occupied.pop(old_pos)
            occupied[new_pos] = i
            
            new_energy = _compute_energy(sequence, best_chain)
            if new_energy < best_energy:
                best_energy = new_energy
                improved = True
            else:
                # Revert
                best_chain[i] = old_pos
                occupied.pop(new_pos)
                occupied[old_pos] = i
        
        if not improved:
            break
    
    return best_chain


def _spiral_fold(sequence, dimension):
    """Create a spiral-like initial conformation."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    dir_idx = 0
    
    for i in range(1, n):
        placed = False
        for _ in range(6):
            d = directions[dir_idx % 6]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                # Turn right every few steps
                if i % 3 == 0:
                    dir_idx = (dir_idx + 1) % 6
                break
            dir_idx = (dir_idx + 1) % 6
        
        if not placed:
            # Try all 6 directions
            for d in directions:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    break
            else:
                # Stuck, extend in any available direction
                for d in directions:
                    new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                    if new_pos not in occupied:
                        chain.append(new_pos)
                        occupied[new_pos] = i
                        break
                else:
                    # Complete failure
                    break
    
    return chain


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice using multiple strategies."""
    _setup_deadline()
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Spiral initialization
    chain1 = _spiral_fold(sequence, dimension)
    if len(chain1) == n:
        e1 = _compute_energy(sequence, chain1)
        if best_chain is None or e1 < best_energy:
            best_chain = chain1
            best_energy = e1
    
    # Strategy 2: Beam search with different widths
    if _time_left() > 5.0:
        for bw in [20, 50, 100]:
            if _time_left() < 2.0:
                break
            max_t = min(5.0, _time_left() * 0.3)
            chain2 = _beam_search(sequence, dimension, rng, beam_width=bw, max_time=max_t)
            if chain2 and len(chain2) == n:
                e2 = _compute_energy(sequence, chain2)
                if best_chain is None or e2 < best_energy:
                    best_chain = chain2
                    best_energy = e2
    
    # Strategy 3: Multiple greedy growths
    if _time_left() > 3.0 and best_chain is not None:
        for _ in range(20):
            if _time_left() < 1.0:
                break
            chain3 = _greedy_grow(sequence, dimension, rng, width=1)
            if chain3 and len(chain3) == n:
                e3 = _compute_energy(sequence, chain3)
                if best_chain is None or e3 < best_energy:
                    best_chain = chain3
                    best_energy = e3
    
    # Strategy 4: Local search refinement
    if best_chain is not None and _time_left() > 2.0:
        max_iters = min(5000, int(_time_left() * 200))
        refined = _local_search(sequence, best_chain, max_iters=max_iters)
        if refined and len(refined) == n:
            er = _compute_energy(sequence, refined)
            if er < best_energy:
                best_chain = refined
                best_energy = er
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
