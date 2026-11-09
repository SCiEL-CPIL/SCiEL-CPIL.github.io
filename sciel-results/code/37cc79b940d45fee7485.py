"""HP lattice protein folding with constraint satisfaction and local search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _greedy_growth(sequence, dimension, rng, bias=0.0):
    """Greedy chain growth with randomization bias."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites with H-H contacts
            score = rng.random() * bias
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H':
                        score += 1.0
            # Small preference for open space to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best with some randomness
        options.sort(key=lambda x: -x[0])
        if bias > 0 and len(options) > 1:
            # Random pick from top few
            k = min(len(options), max(1, int(bias * 3)))
            chosen = rng.choice(options[:k])
        else:
            chosen = options[0]
        
        site = chosen[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, rng, max_iters=200):
    """Simple local search with pivot moves."""
    if chain is None or len(chain) < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    n = len(best_chain)
    
    for _ in range(max_iters):
        if _remaining_time() < 0.1:
            break
        
        # Try a random pivot move
        if n < 3:
            break
        i = rng.randint(1, n - 2)
        
        # Get the segment from i to i+1
        a = best_chain[i - 1]
        b = best_chain[i]
        c = best_chain[i + 1]
        
        # Find all valid positions for b that keep a-b and b-c as unit steps
        # b must be adjacent to a and c
        valid_sites = []
        for site in _neighbours(a):
            if site == c:
                continue
            # Check if site is adjacent to c
            if abs(site[0] - c[0]) + abs(site[1] - c[1]) == 1:
                # Check if site is not occupied by any other residue
                occupied_set = set(best_chain)
                occupied_set.discard(b)
                if site not in occupied_set:
                    valid_sites.append(site)
        
        if not valid_sites:
            continue
        
        # Try each valid site
        for new_b in valid_sites:
            if new_b == b:
                continue
            
            # Check if new_b is already in the chain (excluding position i)
            for j, p in enumerate(best_chain):
                if j != i and p == new_b:
                    break
            else:
                # Apply move
                old_b = best_chain[i]
                best_chain[i] = new_b
                
                # Check self-avoidance
                valid = True
                for j, p in enumerate(best_chain):
                    if j != i and p == new_b:
                        valid = False
                        break
                
                if valid:
                    new_energy = _energy(sequence, best_chain)
                    if new_energy <= best_energy:
                        best_energy = new_energy
                    else:
                        # Revert
                        best_chain[i] = old_b
                else:
                    best_chain[i] = old_b
    
    return best_chain


def _backtracking_fold(sequence, dimension, max_depth=None, time_limit=None):
    """Backtracking search with heuristic placement of H residues."""
    if max_depth is None:
        max_depth = len(sequence)
    if time_limit is None:
        time_limit = 5.0
    
    start_time = time.time()
    best_chain = None
    best_energy = 0
    
    # Use iterative deepening or simple backtracking
    # For simplicity, use a randomized greedy approach with multiple restarts
    rng = random.Random(42)
    
    attempts = 0
    while time.time() - start_time < time_limit and _remaining_time() > 1.0:
        chain = _greedy_growth(sequence, dimension, rng, bias=0.5 + rng.random() * 0.5)
        if chain is not None:
            # Apply local search
            chain = _local_search(sequence, chain, rng, max_iters=50)
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
        attempts += 1
        if attempts > 1000:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using hybrid search strategy."""
    if not sequence or dimension != 2:
        return []
    
    n = len(sequence)
    if n == 0:
        return []
    
    # For very short sequences, use exhaustive or simple methods
    if n <= 10:
        # Try many greedy growths
        rng = random.Random(12345 + n)
        best_chain = None
        best_energy = 0
        for _ in range(500):
            chain = _greedy_growth(sequence, dimension, rng, bias=rng.random())
            if chain is not None:
                energy = _energy(sequence, chain)
                if best_chain is None or energy < best_energy:
                    best_chain = chain
                    best_energy = energy
        if best_chain is None:
            best_chain = [(i, 0) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # For longer sequences, use hybrid approach
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths with local search
    rng = random.Random(12345 + n)
    for i in range(100):
        if _remaining_time() < 5.0:
            break
        chain = _greedy_growth(sequence, dimension, rng, bias=0.3 + rng.random() * 0.7)
        if chain is not None:
            chain = _local_search(sequence, chain, rng, max_iters=30)
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    # Phase 2: If we have time, try more sophisticated search
    if _remaining_time() > 10.0 and best_chain is not None:
        # Try to improve with more local search iterations
        improved = _local_search(sequence, best_chain, rng, max_iters=200)
        if improved is not None:
            energy = _energy(sequence, improved)
            if energy < best_energy:
                best_chain = improved
                best_energy = energy
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
