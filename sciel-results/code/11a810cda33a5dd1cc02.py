"""HP lattice protein folding on the cubic lattice.
Strategy: Beam search with incremental contact tracking + multi-stage local optimization
using kink moves, pivot moves, and reversal moves with simulated annealing.
"""
import time
import random
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = 325.0  # wall-clock seconds, safety margin
_START_TIME = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute H-H contact energy (negative count)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width, rng, time_budget):
    """Beam search for folding. Returns list of (chain, energy) for top candidates."""
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    start = (0,) * dimension
    start_occ = {start: 0}
    start_chain = [start]
    start_contacts = 0
    
    beam = [(0, start_chain, start_occ, start_contacts)]
    
    for i in range(1, n):
        if _time_left() < 5.0:
            break
        
        new_beam = []
        for neg_score, chain, occ, contacts in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occ:
                    continue
                new_occ = dict(occ)
                new_occ[site] = i
                new_contacts = contacts
                if is_h[i]:
                    for q in _neighbours(site):
                        j = new_occ.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            new_contacts += 1
                new_chain = chain + [site]
                new_neg = -new_contacts
                new_beam.append((new_neg, new_chain, new_occ, new_contacts))
        
        if not new_beam:
            break
        
        # Keep top beam_width by neg_score (most negative = best)
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    results = []
    for neg_score, chain, occ, contacts in beam:
        energy = -contacts
        results.append((chain, energy))
    return results


def _pivot_move_fast(chain, i, rng, dimension):
    """Fast pivot move with early collision abort."""
    if i <= 1 or i >= len(chain) - 1:
        return None

    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]

    axis = rng.randint(0, dimension - 1)
    sign = rng.choice([-1, 1])

    new_tail = []
    seen_in_tail = set()
    head_set = set(head)

    for pos in tail:
        vec = [pos[k] - pivot[k] for k in range(dimension)]
        if axis == 0:
            new_vec = [vec[0], -sign * vec[2], sign * vec[1]]
        elif axis == 1:
            new_vec = [sign * vec[2], vec[1], -sign * vec[0]]
        else:
            new_vec = [-sign * vec[1], sign * vec[0], vec[2]]
        new_pos = tuple(pivot[k] + new_vec[k] for k in range(dimension))

        if new_pos in seen_in_tail or new_pos in head_set:
            return None
        seen_in_tail.add(new_pos)
        new_tail.append(new_pos)

    return head + new_tail


def _reversal_move(chain, i, j, rng):
    """Reverse a segment of the chain from i to j (inclusive)."""
    if i < 1 or j >= len(chain) - 1 or j <= i:
        return None

    if i > 0:
        d1 = sum(abs(chain[i-1][k] - chain[j][k]) for k in range(len(chain[i])))
        if d1 != 1:
            return None
    if j < len(chain) - 1:
        d2 = sum(abs(chain[j+1][k] - chain[i][k]) for k in range(len(chain[i])))
        if d2 != 1:
            return None

    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]

    if len(set(new_chain)) != len(new_chain):
        return None

    return new_chain


def _kink_move(chain, i, rng, dimension):
    """Try to move residue i to an adjacent free site (kink move)."""
    if i <= 0 or i >= len(chain) - 1:
        return None

    occupied = set(chain)
    occupied.discard(chain[i])

    prev_pos = chain[i-1]
    next_pos = chain[i+1]

    for site in _neighbours(chain[i]):
        if site in occupied:
            continue

        dist_prev = sum(abs(site[k] - prev_pos[k]) for k in range(dimension))
        dist_next = sum(abs(site[k] - next_pos[k]) for k in range(dimension))

        if dist_prev == 1 and dist_next == 1:
            new_chain = list(chain)
            new_chain[i] = site
            return new_chain

    return None


def _local_optimize_sa(sequence, chain, max_iters=2000, rng=None):
    """Simulated annealing local optimization."""
    if rng is None:
        rng = random.Random(42)
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    current_energy = _compute_energy(sequence, chain)
    dimension = len(chain[0])
    
    best_chain = list(chain)
    best_energy = current_energy
    
    temperature = 2.0
    cooling = 0.999
    
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Randomly choose a move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Kink move
            i = rng.randint(1, n - 2)
            new_chain = _kink_move(chain, i, rng, dimension)
        elif move_type < 0.8:
            # Pivot move
            i = rng.randint(2, n - 3)
            new_chain = _pivot_move_fast(chain, i, rng, dimension)
        else:
            # Reversal move
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, min(i + 6, n - 2))
            new_chain = _reversal_move(chain, i, j, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / temperature):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    return best_chain, best_energy


def _local_optimize_greedy(sequence, chain, max_iters=1000, rng=None):
    """Greedy local optimization with multiple move types."""
    if rng is None:
        rng = random.Random(42)
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    current_energy = _compute_energy(sequence, chain)
    dimension = len(chain[0])
    
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        
        improved = False
        
        # Try kink moves
        for i in range(1, n - 1):
            new_chain = _kink_move(chain, i, rng, dimension)
            if new_chain is not None:
                new_energy = _compute_energy(sequence, new_chain)
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try pivot moves
        for i in range(2, n - 2):
            new_chain = _pivot_move_fast(chain, i, rng, dimension)
            if new_chain is not None:
                new_energy = _compute_energy(sequence, new_chain)
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try reversal moves
        for i in range(1, n - 2):
            for j in range(i + 1, min(i + 5, n - 1)):
                new_chain = _reversal_move(chain, i, j, rng)
                if new_chain is not None:
                    new_energy = _compute_energy(sequence, new_chain)
                    if new_energy < current_energy:
                        chain = new_chain
                        current_energy = new_energy
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return chain, current_energy


def _multi_start_fold(sequence, dimension, n_starts=3):
    """Run beam search multiple times with different seeds and return best result."""
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    for start in range(n_starts):
        if _time_left() < 10.0:
            break
        
        rng = random.Random(12345 + n + start * 1000)
        
        # Vary beam width slightly
        if n <= 30:
            beam_width = 800 + start * 100
        elif n <= 50:
            beam_width = 400 + start * 50
        elif n <= 80:
            beam_width = 250 + start * 25
        else:
            beam_width = 120 + start * 10
        
        results = _beam_search(sequence, dimension, beam_width, rng, _time_left())
        
        if not results:
            continue
        
        results.sort(key=lambda x: x[1])
        
        # Optimize top candidates
        for chain, energy in results[:5]:
            if _time_left() < 5.0:
                break
            
            # First do greedy, then SA
            opt_chain, opt_energy = _local_optimize_greedy(sequence, chain, max_iters=300)
            opt_chain, opt_energy = _local_optimize_sa(sequence, opt_chain, max_iters=500)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    
    # Determine number of restarts based on remaining time
    if n <= 40:
        n_starts = 4
    elif n <= 80:
        n_starts = 3
    else:
        n_starts = 2
    
    best_chain, best_energy = _multi_start_fold(sequence, dimension, n_starts)
    
    if best_chain is None:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
