"""HP lattice protein folding using greedy growth with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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


def _local_search(sequence, chain, max_iters=5000, rng=None):
    """Local search: try moving each residue to a neighbouring empty site, keep if energy improves."""
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for iteration in range(max_iters):
        # Pick a random residue to move (not endpoints)
        i = rng.randint(1, n - 2)
        orig = chain[i]
        
        # Try all neighbouring sites
        best_move = None
        best_move_energy = current_energy
        
        for site in _neighbours(orig):
            if site in occupied:
                continue
            # Check if moving residue i to site is valid
            # The site must not be occupied by any other residue
            # And the new position must not create a self-intersection
            
            # Tentatively move
            del occupied[orig]
            occupied[site] = i
            chain[i] = site
            
            # Check self-avoidance: the new position should not conflict
            # Since we already checked it's not occupied, it should be fine
            # But we need to verify the chain is still a valid walk
            # (unit steps between consecutive residues)
            
            # Compute new energy
            new_energy = _energy(sequence, chain)
            
            if new_energy < best_move_energy:
                best_move_energy = new_energy
                best_move = (i, site)
            
            # Undo the move
            del occupied[site]
            occupied[orig] = i
            chain[i] = orig
        
        if best_move is not None:
            # Apply the best move
            i, site = best_move
            orig = chain[i]
            del occupied[orig]
            occupied[site] = i
            chain[i] = site
            current_energy = best_move_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # No improving move found; restart with a small perturbation
            # Try a random non-improving move to escape local minima
            if iteration % 10 == 0:
                i = rng.randint(1, n - 2)
                orig = chain[i]
                neighbours = [s for s in _neighbours(orig) if s not in occupied]
                if neighbours:
                    site = rng.choice(neighbours)
                    del occupied[orig]
                    occupied[site] = i
                    chain[i] = site
                    current_energy = _energy(sequence, chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by local search refinement."""
    start_time = time.time()
    deadline = start_time + 150  # Leave 60s margin for 210s wall-clock limit
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    
    # Phase 1: Greedy growth with multiple restarts
    num_grow = 50 if n <= 30 else (30 if n <= 60 else 20)
    
    for trial in range(num_grow):
        if time.time() > deadline:
            break
        
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        # Phase 2: Local search refinement
        refined_chain, refined_energy = _local_search(
            sequence, chain, 
            max_iters=min(2000, max(500, 50000 // n)),
            rng=rng
        )
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
        
        # If we found a very good solution, do additional refinement
        if best_energy <= -5 and time.time() < deadline - 30:
            further_chain, further_energy = _local_search(
                sequence, best_chain,
                max_iters=min(3000, max(1000, 80000 // n)),
                rng=rng
            )
            if further_energy < best_energy:
                best_energy = further_energy
                best_chain = further_chain
    
    if best_chain is None:
        # Fallback: a straight line is always self-avoiding
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
