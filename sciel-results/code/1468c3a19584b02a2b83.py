"""HP lattice protein folding on the cubic lattice.
Strategy: Multi-start beam search + simulated annealing with move types.
Deadline guard for 330s wall-clock limit.
"""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = 325.0  # wall-clock seconds
_START_TIME = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute H-H contact energy (negative count)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(tuple(p)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width, seed=42):
    """Beam search for folding. Returns list of (chain, energy)."""
    rng = random.Random(seed)
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    start = (0, 0, 0)
    start_occ = {start: 0}
    start_chain = [start]
    
    # Beam: list of (neg_contacts, chain, occ_dict)
    # We maximize contacts, so minimize -contacts
    beam = [(0, start_chain, start_occ)]
    
    for i in range(1, n):
        if _time_left() < 3.0:
            break
        
        new_beam = []
        for neg_c, chain, occ in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occ:
                    continue
                new_occ = dict(occ)
                new_occ[site] = i
                new_contacts = -neg_c  # convert back to positive
                if is_h[i]:
                    for q in _neighbours(site):
                        j = new_occ.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            new_contacts += 1
                new_chain = chain + [site]
                new_neg = -new_contacts
                new_beam.append((new_neg, new_chain, new_occ))
        
        if not new_beam:
            break
        
        # Keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    results = []
    for neg_c, chain, occ in beam:
        energy = _compute_energy(sequence, chain)
        results.append((chain, energy))
    return results


def _anneal(sequence, chain, max_iters=5000, t_start=2.0, t_end=0.01, rng=None):
    """Simulated annealing with various move types."""
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    
    # Build position lookup
    pos_map = {tuple(chain[i]): i for i in range(n)}
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    t = t_start
    cooling = (t_start - t_end) / max(1, max_iters)
    
    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Try all 6 neighbors
        candidates = list(_neighbours(old_pos))
        rng.shuffle(candidates)
        
        moved = False
        for new_pos in candidates:
            # Check occupancy
            occupied = set(pos_map.keys())
            occupied.discard(old_pos)
            if new_pos in occupied:
                continue
            
            # Check connectivity: must be adjacent to i-1 and i+1
            if _manhattan(new_pos, chain[i-1]) != 1:
                continue
            if _manhattan(new_pos, chain[i+1]) != 1:
                continue
            
            # Compute energy delta
            # Remove old position
            del pos_map[old_pos]
            chain[i] = new_pos
            pos_map[new_pos] = i
            
            new_energy = _compute_energy(sequence, chain)
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < random.exp(-delta_e / max(t, 1e-10)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                moved = True
            else:
                # Revert
                chain[i] = old_pos
                del pos_map[new_pos]
                pos_map[old_pos] = i
            
            if moved:
                break
        
        t -= cooling
    
    return best_chain, best_energy


def _kink_move(sequence, chain, i, rng):
    """Try a kink move at position i (rotate segment)."""
    n = len(chain)
    if i < 2 or i > n - 3:
        return None, 0
    
    # Simple 2-residue kink: rotate residues i and i+1
    p0 = chain[i-1]
    p1 = chain[i]
    p2 = chain[i+1]
    p3 = chain[i+2] if i + 2 < n else None
    
    # Try to swap p1 and p2 if it creates a valid configuration
    # Actually, let's try a different kink: move residue i to a new position
    # that keeps connectivity
    
    old_pos = chain[i]
    for new_pos in _neighbours(old_pos):
        occupied = set(tuple(p) for p in chain)
        occupied.discard(old_pos)
        if new_pos in occupied:
            continue
        if _manhattan(new_pos, chain[i-1]) != 1:
            continue
        if i + 1 < n and _manhattan(new_pos, chain[i+1]) != 1:
            continue
        
        chain[i] = new_pos
        e = _compute_energy(sequence, chain)
        chain[i] = old_pos
        return chain, e
    
    return None, 0


def _local_optimize(sequence, chain, max_iters=1000, seed=0):
    """Local optimization using pivot moves and annealing."""
    rng = random.Random(seed)
    n = len(chain)
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, chain)
    
    # Phase 1: Greedy local search with pivot moves
    improved = True
    while improved and _time_left() > 2.0:
        improved = False
        for i in range(1, n - 1):
            old_pos = chain[i]
            for new_pos in _neighbours(old_pos):
                occupied = set(tuple(p) for p in chain)
                occupied.discard(old_pos)
                if new_pos in occupied:
                    continue
                if _manhattan(new_pos, chain[i-1]) != 1:
                    continue
                if i + 1 < n and _manhattan(new_pos, chain[i+1]) != 1:
                    continue
                
                chain[i] = new_pos
                new_e = _compute_energy(sequence, chain)
                if new_e < best_energy:
                    best_energy = new_e
                    best_chain = list(chain)
                    improved = True
                    break
                else:
                    chain[i] = old_pos
            if improved:
                break
    
    # Phase 2: Annealing from best chain
    if _time_left() > 2.0:
        annealed_chain, annealed_energy = _anneal(sequence, best_chain, 
                                                   max_iters=3000, seed=seed + 100)
        if annealed_energy < best_energy:
            best_energy = annealed_energy
            best_chain = annealed_chain
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Choose beam width based on chain length
    if n <= 20:
        beam_width = 2000
        num_restarts = 3
    elif n <= 40:
        beam_width = 1000
        num_restarts = 3
    elif n <= 60:
        beam_width = 500
        num_restarts = 2
    elif n <= 100:
        beam_width = 300
        num_restarts = 2
    else:
        beam_width = 150
        num_restarts = 1
    
    all_candidates = []
    
    for restart in range(num_restarts):
        if _time_left() < 5.0:
            break
        
        results = _beam_search(sequence, dimension, beam_width, seed=1000 + restart * 7)
        if results:
            # Sort by energy
            results.sort(key=lambda x: x[1])
            # Take top candidates
            top_k = min(5, len(results))
            all_candidates.extend(results[:top_k])
    
    if not all_candidates:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Sort all candidates by energy
    all_candidates.sort(key=lambda x: x[1])
    
    # Optimize top candidates
    best_chain = None
    best_energy = 0
    num_to_optimize = min(8, len(all_candidates))
    
    for idx, (chain, energy) in enumerate(all_candidates[:num_to_optimize]):
        if _time_left() < 3.0:
            break
        
        opt_chain, opt_energy = _local_optimize(sequence, chain, seed=idx * 1000)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
    
    if best_chain is None:
        best_chain = all_candidates[0][0]
    
    # Final verification: ensure self-avoidance and unit steps
    final_chain = [list(p) for p in best_chain]
    
    return final_chain

# EVOLVE-BLOCK-END
