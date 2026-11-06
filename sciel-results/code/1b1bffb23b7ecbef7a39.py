"""HP lattice protein folding on the cubic lattice.
Strategy: Beam search with incremental contact tracking + multi-move local optimization
with simulated annealing for escaping local minima.
Deadline guard for 330s wall-clock limit.
"""
import time
import random
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = 330.0  # wall-clock seconds
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
    
    # State: (neg_score, chain, occupied_dict, current_contacts)
    # occupied_dict maps point -> index
    
    start = (0,) * dimension
    start_occ = {start: 0}
    start_chain = [start]
    start_contacts = 0
    
    beam = [(0, start_chain, start_occ, start_contacts)]
    
    for i in range(1, n):
        if _time_left() < 5.0:
            break  # Return what we have
        
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
    
    # Convert to results
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

    # Check boundary connectivity after reversal
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


def _local_optimize_sa(sequence, chain, max_iters=300, rng_seed=42):
    """Local optimization with simulated annealing to escape local minima."""
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    current_energy = _compute_energy(sequence, chain)
    rng = random.Random(rng_seed)
    dimension = len(chain[0])
    
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.95
    
    for it in range(max_iters):
        if _time_left() < 3.0:
            break
        
        # Choose move type: 30% kink, 40% pivot, 30% reversal
        move_type = rng.random()
        
        new_chain = None
        if move_type < 0.3:
            # Kink move
            i = rng.randint(1, n - 2)
            new_chain = _kink_move(chain, i, rng, dimension)
        elif move_type < 0.7:
            # Pivot move
            i = rng.randint(2, n - 2)
            new_chain = _pivot_move_fast(chain, i, rng, dimension)
        else:
            # Reversal move
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, min(i + 5, n - 1))
            new_chain = _reversal_move(chain, i, j, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e/T) if worse
        if delta_e < 0 or (T > T_min and rng.random() < pow(2.718281828, -delta_e / T)):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling_rate
    
    return best_chain, best_energy


def _local_optimize_greedy(sequence, chain, max_iters=500):
    """Try single-residue moves and pivot moves to improve energy (greedy)."""
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    current_energy = _compute_energy(sequence, chain)
    rng = random.Random(42)
    dimension = len(chain[0])
    
    for _ in range(max_iters):
        if _time_left() < 3.0:
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


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Choose beam width based on chain length
    if n <= 30:
        beam_width = 1000
    elif n <= 50:
        beam_width = 500
    elif n <= 80:
        beam_width = 300
    else:
        beam_width = 150
    
    # Run beam search
    results = _beam_search(sequence, dimension, beam_width, rng, _time_left())
    
    if not results:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Take top candidates and locally optimize
    best_chain = None
    best_energy = 0
    
    # Sort by energy (most negative first)
    results.sort(key=lambda x: x[1])
    
    # Determine how many candidates to optimize based on time
    time_per_candidate = 15.0 if n <= 50 else 8.0
    max_candidates = max(3, int(_time_left() / time_per_candidate))
    max_candidates = min(max_candidates, 15)
    
    # Optimize top candidates with both greedy and SA
    for idx, (chain, energy) in enumerate(results[:max_candidates]):
        if _time_left() < 5.0:
            break
        
        # First try greedy optimization
        opt_chain, opt_energy = _local_optimize_greedy(sequence, chain)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
        
        # Then try SA on the greedy-optimized result
        if _time_left() > 10.0:
            sa_chain, sa_energy = _local_optimize_sa(sequence, opt_chain, 
                                                     max_iters=200 if n <= 50 else 100,
                                                     rng_seed=42 + idx)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_chain = sa_chain
    
    if best_chain is None:
        best_chain = results[0][0]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
