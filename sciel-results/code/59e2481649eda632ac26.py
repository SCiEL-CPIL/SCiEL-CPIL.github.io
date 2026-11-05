"""Rewritten program for HP lattice protein folding with multi-start simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Global deadline guard
_DEADLINE = 300.0  # seconds wall-clock limit with safety margin
_start_time = None


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP contact energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _pivot_move(chain, n, rng):
    """
    Perform a pivot move: rotate a random suffix of the chain.
    Returns a new chain if valid, None otherwise.
    """
    if n <= 2:
        return None
    
    # Choose pivot point (index from 1 to n-2)
    pivot = rng.randint(1, n - 2)
    prefix = chain[:pivot + 1]
    suffix = chain[pivot + 1:]
    
    # The vector from pivot to next
    v = tuple(suffix[0][d] - prefix[-1][d] for d in range(len(prefix[-1])))
    
    # Generate all possible rotations of this vector in 3D
    # For a unit vector, there are 5 other unit vectors at distance 1
    candidates = []
    for axis in range(3):
        for step in (-1, 1):
            new_v = list(v)
            new_v[axis] = step
            # Check it's a unit step and different from original
            if new_v != list(v):
                candidates.append(tuple(new_v))
    
    if not candidates:
        return None
    
    # Randomly pick a rotation
    new_first = tuple(prefix[-1][d] + candidates[rng.randint(0, len(candidates) - 1)][d] 
                      for d in range(len(prefix[-1])))
    
    # Apply the rotation to the entire suffix
    # The rotation is: new_pos = old_pos - old_first + new_first
    dx = new_first[0] - suffix[0][0]
    dy = new_first[1] - suffix[0][1]
    dz = new_first[2] - suffix[0][2]
    
    new_suffix = [(suffix[i][0] + dx, suffix[i][1] + dy, suffix[i][2] + dz) for i in range(len(suffix))]
    
    new_chain = prefix + new_suffix
    
    # Check self-avoidance
    occupied = set(new_chain)
    if len(occupied) != len(new_chain):
        return None
    
    # Also check that the bond from prefix to suffix is valid (unit step)
    # Already ensured by construction
    
    return new_chain


def _tail_regen_move(chain, sequence, n, rng, regen_len=10):
    """
    Tail regeneration: remove the last few residues and regrow them greedily.
    This acts like a quantum tunneling move to escape local minima.
    Returns a new chain if valid, None otherwise.
    """
    if n <= regen_len + 2:
        return None
    
    # Keep a random prefix length
    keep = n - rng.randint(regen_len, min(20, n - 2))
    if keep < 2:
        return None
    
    prefix = chain[:keep]
    occupied = set(prefix)
    
    # Regrow the tail
    new_chain = list(prefix)
    for i in range(keep, n):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        # Check if it's an H residue in the prefix
                        idx_in_prefix = -1
                        for j, p in enumerate(prefix):
                            if p == q:
                                idx_in_prefix = j
                                break
                        if idx_in_prefix >= 0 and sequence[idx_in_prefix] == "H" and idx_in_prefix < i - 1:
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied.add(site)
        new_chain.append(site)
    
    return new_chain


def _simulated_annealing(sequence, dimension, initial_chain, max_iters, rng, deadline):
    """
    Run simulated annealing starting from initial_chain.
    Returns the best chain found.
    """
    n = len(sequence)
    current = list(initial_chain)
    current_energy = _energy(sequence, current)
    
    best = list(current)
    best_energy = current_energy
    
    # Initial and final temperatures
    T_start = 1.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    T = T_start
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Choose move type
        if rng.random() < 0.3 and n > 12:
            new_chain = _tail_regen_move(current, sequence, n, rng, regen_len=rng.randint(5, min(15, n - 3)))
        else:
            new_chain = _pivot_move(current, n, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis acceptance
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(T, 1e-10)):
            current = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best = list(current)
        
        T *= cooling_rate
    
    return best, best_energy


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Uses multi-start simulated annealing with tail regeneration moves.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    deadline = _start_time + _DEADLINE - 5.0  # 5 second safety margin
    
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 10 ** 9
    
    # Phase 1: Generate diverse starting points via greedy growth
    starts = []
    num_starts = min(50, max(10, 300 // n))  # Scale with chain length
    for _ in range(num_starts):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            starts.append(chain)
    
    if not starts:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Phase 2: Run simulated annealing from each start
    # Allocate time evenly
    time_per_sa = (deadline - time.time()) / len(starts) if len(starts) > 0 else 0
    
    for idx, start_chain in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Adjust iterations based on remaining time
        elapsed = time.time() - _start_time
        remaining = deadline - time.time()
        
        # Estimate: each SA iteration takes roughly 0.01-0.05 ms for small chains
        # Be conservative
        max_iters = int(remaining * 50000)  # ~50k iters/sec estimate
        max_iters = max(100, min(max_iters, 50000))
        
        sa_chain, sa_energy = _simulated_annealing(
            sequence, dimension, start_chain, max_iters, rng, deadline
        )
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    if best_chain is None:
        best_chain = starts[0]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
