"""HP lattice protein folding: simulated annealing with local search moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 195.0  # wall-clock limit with safety margin (210s kill, leave 15s buffer)


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    """Yield 4-neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy (negative count of H-H non-adjacent contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk of length len(sequence)."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Start at origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        
        if not options:
            return None  # Stuck
        
        # Pick randomly
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
    
    return chain


def _local_search(sequence, chain, rng, time_budget):
    """Improve a conformation using simulated annealing with local moves."""
    n = len(sequence)
    if n <= 1:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    start_temp = 2.0
    end_temp = 0.01
    steps = min(5000, max(500, n * 50))
    
    start_time = time.time()
    
    for step in range(steps):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Generate a random move: pick a residue and try to move it
        i = rng.randint(1, n - 1)  # Don't move the first residue
        old_pos = current_chain[i]
        
        # Try each neighbour
        for new_pos in _neighbours(old_pos):
            # Check if new position is free
            if new_pos in set(current_chain):
                continue
            
            # Check if moving i to new_pos keeps the chain connected
            # The chain from 0 to i-1 is unchanged, so we just need to check
            # that new_pos is adjacent to old_pos (which it is by construction)
            # and that the rest of the chain from i+1 onward doesn't need to change
            # Actually, we need to check that the segment from i to n-1 remains valid
            # This is complex; let's use a simpler move: swap two non-adjacent residues' positions
            # or use a pivot move. Let's use a "pivot" move: pick i, rotate the tail.
            
            # Actually, let's use a simpler approach: try to move a single residue
            # to an adjacent empty site, but we need to ensure the chain remains connected.
            # A single residue move only works if the new position is adjacent to the old
            # AND the old position is adjacent to the new position's neighbours in the chain.
            # This is tricky. Let's use a different move: "reversal" or "insertion".
            
            # For simplicity, let's use a "kick" move: pick a random residue i (not 0 or n-1),
            # and try to move it to an adjacent empty site. But this breaks connectivity
            # unless the new site is adjacent to both i-1 and i+1 in the chain.
            
            # Let's use a different approach: try to insert a "kink" by moving a short segment.
            # Actually, the simplest valid move is to pick i and try to move residue i to a
            # neighbouring site that is empty and such that the chain remains a valid SAW.
            # This requires that new_pos is adjacent to current_chain[i-1] (if i>0) and
            # that current_chain[i+1] (if i<n-1) is adjacent to new_pos.
            
            # This is getting complicated. Let's use a well-known move: "local move" where
            # we pick i and try to move it to a site adjacent to its current position,
            # but we need to verify the whole chain remains self-avoiding and connected.
            
            # A simpler and valid move: pick two non-adjacent residues i and j (|i-j|>1),
            # and swap their positions if the resulting chain is self-avoiding and connected.
            # But this is also complex to verify.
            
            # Let's use a "pivot" move: pick i, and rotate the tail from i to n-1 by 90, 180, or 270
            # degrees around current_chain[i-1]. This is a standard move in SAW simulations.
            
            # For now, let's use a simpler heuristic: try to improve the energy by moving
            # a hydrophobic residue closer to other hydrophobic residues.
            
            break  # Skip this move type for now
        
        # Use a "kick" move: pick i, try to move it to an adjacent empty site.
        # Verify that the chain remains valid.
        i = rng.randint(1, n - 1)
        old_pos = current_chain[i]
        moved = False
        
        for new_pos in _neighbours(old_pos):
            # Check if new_pos is already occupied
            if new_pos in set(current_chain):
                continue
            
            # Check connectivity: new_pos must be adjacent to current_chain[i-1]
            # and current_chain[i+1] must be adjacent to new_pos
            if i > 0 and new_pos not in _neighbours(current_chain[i - 1]):
                continue
            if i < n - 1 and new_pos not in _neighbours(current_chain[i + 1]):
                continue
            
            # Try the move
            current_chain[i] = new_pos
            new_energy = _energy(sequence, current_chain)
            
            if new_energy <= current_energy or rng.random() < math.exp((new_energy - current_energy) / temp):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                moved = True
                break
            else:
                # Revert
                current_chain[i] = old_pos
        
        if not moved:
            # Try a different move: pick a random residue and try all its neighbours
            i = rng.randint(1, n - 1)
            old_pos = current_chain[i]
            for new_pos in _neighbours(old_pos):
                if new_pos in set(current_chain):
                    continue
                # Check connectivity
                if i > 0 and new_pos not in _neighbours(current_chain[i - 1]):
                    continue
                if i < n - 1 and new_pos not in _neighbours(current_chain[i + 1]):
                    continue
                
                current_chain[i] = new_pos
                new_energy = _energy(sequence, current_chain)
                
                if new_energy <= current_energy or rng.random() < math.exp((new_energy - current_energy) / temp):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                    break
                else:
                    current_chain[i] = old_pos
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple random walks and local search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so start at 0 (no contacts)
    
    # Time budget for this sequence
    total_budget = _remaining()
    if total_budget <= 0:
        # Out of time, return a straight line
        return [[i, 0] for i in range(n)]
    
    # Allocate time: 30% for random walks, 70% for local search
    walk_time = total_budget * 0.3
    search_time = total_budget * 0.7
    
    # Generate initial conformations
    initial_chains = []
    walk_start = time.time()
    
    while time.time() - walk_start < walk_time:
        chain = _random_walk(sequence, dimension, rng)
        if chain is not None:
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
            initial_chains.append(chain)
        
        # Limit number of walks
        if len(initial_chains) >= 50:
            break
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Local search on the best initial conformation and a few others
    search_start = time.time()
    chains_to_improve = [best_chain]
    if len(initial_chains) > 1:
        # Add a few diverse chains
        for chain in initial_chains[:5]:
            if chain != best_chain:
                chains_to_improve.append(chain)
    
    for chain in chains_to_improve:
        if time.time() - search_start > search_time:
            break
        
        time_left = search_time - (time.time() - search_start)
        if time_left <= 0:
            break
        
        improved = _local_search(sequence, chain, rng, time_left / len(chains_to_improve))
        energy = _energy(sequence, improved)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = improved
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
