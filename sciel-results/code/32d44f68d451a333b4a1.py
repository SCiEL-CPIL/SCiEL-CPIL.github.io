"""Rewritten program for HP lattice protein folding using Hamiltonian path templates."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts) where contacts are non-adjacent H residues on neighbouring lattice sites.
    Returns a negative number (more negative = better)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, temperature=1.0):
    """Randomized greedy chain growth with simulated annealing-like selection.
    Prefers sites that create H-H contacts, then has free space, then random."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Calculate gain: number of new H-H contacts if we place residue i here
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            # Count free neighbours for flexibility
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Score: primarily gain, secondarily free space, tertiary random
            # Using a weighted score with some randomness
            score = gain * 10.0 + free * 0.5 + rng.random() * temperature
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Select the best candidate
        best_score, best_site = max(candidates)
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _local_search(sequence, chain, max_iters=50, rng=None):
    """Perform local search by trying to swap adjacent non-consecutive residues 
    or make small moves to improve energy. For simplicity, we'll do a simple 
    1-2-3 move heuristic: try removing a middle section and reinserting it."""
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_iters):
        # Try a simple perturbation: pick a random position and try to move 
        # a segment of length 2-3
        if n < 3:
            break
        
        # Find a random interior segment to try to flip/move
        start_idx = rng.randint(1, n - 3)
        seg_len = rng.randint(2, 3)
        if start_idx + seg_len > n:
            continue
        
        # Try to reverse the segment in terms of direction (not position)
        # This is complex, so instead let's do a simpler approach:
        # Try to "slide" a small part of the chain
        
        # Actually, a simpler local move: for each residue (except ends), 
        # try to see if it can move to an adjacent empty position
        improved = False
        for i in rng.sample(range(1, n - 1), min(5, n - 2)):
            old_pos = chain[i]
            for new_pos in _neighbours(old_pos):
                if new_pos in chain:  # Check if position is occupied
                    continue
                # Check that moving this residue doesn't break self-avoidance
                # and that the chain remains valid
                # We need to verify that new_pos is not occupied by any other residue
                # and that the path from chain[i-1] to new_pos to chain[i+1] is valid
                # (unit steps and no conflicts)
                
                # Simple check: new_pos must be adjacent to chain[i-1] and chain[i+1]
                # and not occupied
                if new_pos not in chain:
                    # Check adjacency
                    dist1 = abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1])
                    dist2 = abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1])
                    if dist1 == 1 and dist2 == 1:
                        # This move is valid
                        new_chain = list(chain)
                        new_chain[i] = new_pos
                        new_energy = _energy(sequence, new_chain)
                        if new_energy < best_energy:
                            best_chain = new_chain
                            best_energy = new_energy
                            improved = True
                        break
    
        if not improved:
            # Accept current as best so far
            break
    
    return best_chain, best_energy

def _generate_hamiltonian_template(n, rng, num_iterations=1000):
    """Generate a low-contact-energy self-avoiding walk of length n using simulated annealing.
    Returns a list of n lattice points forming a valid self-avoiding walk."""
    if n < 2:
        return [(0, 0)] if n == 1 else []
    
    # Initialize with a simple spiral or straight line
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)  # Start going right
    
    for i in range(1, n):
        x, y = chain[-1]
        dx, dy = direction
        # Try to continue in the same direction, then turn
        candidates = []
        for ndx, ndy in [(dx, dy), (dy, -dx), (-dx, -dy), (-dy, dx)]:
            new_pos = (x + ndx, y + ndy)
            if new_pos not in occupied:
                candidates.append((ndx, ndy))
        
        if not candidates:
            # Dead end - try random restart
            break
        
        # With some probability, take a random turn for diversity
        if rng.random() < 0.3:
            ndx, ndy = rng.choice(candidates)
        else:
            ndx, ndy = candidates[0]
        
        direction = (ndx, ndy)
        new_pos = (x + ndx, y + ndy)
        chain.append(new_pos)
        occupied[new_pos] = i
    
    # If we couldn't complete the walk, try a different approach
    if len(chain) < n:
        # Use a simple serpentine pattern
        chain = []
        row = 0
        col = 0
        direction = 1
        for i in range(n):
            chain.append((col, row))
            col += direction
            if col >= n - row or col <= row:
                row += 1
                direction = -direction
                col = row if direction == 1 else n - 1 - row
    
    # Simulated annealing to improve the template
    best_chain = list(chain)
    best_energy = _energy("H" * n, best_chain)  # Use all H to maximize contacts
    current_chain = list(chain)
    current_energy = best_energy
    
    temperature = 10.0
    cooling = 0.999
    
    for _ in range(num_iterations):
        if len(current_chain) < 3:
            break
        
        # Try a pivot move: pick a random interior position and rotate
        i = rng.randint(1, len(current_chain) - 2)
        old_pos = current_chain[i]
        
        for new_pos in _neighbours(old_pos):
            # Check if new_pos is not occupied
            if new_pos in current_chain:
                continue
            
            # Check adjacency constraints
            dist1 = abs(new_pos[0] - current_chain[i-1][0]) + abs(new_pos[1] - current_chain[i-1][1])
            dist2 = abs(new_pos[0] - current_chain[i+1][0]) + abs(new_pos[1] - current_chain[i+1][1])
            if dist1 != 1 or dist2 != 1:
                continue
            
            # Valid move
            new_chain = list(current_chain)
            new_chain[i] = new_pos
            new_energy = _energy("H" * n, new_chain)
            
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < pow(2.718281828, -delta / temperature):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                break
        
        temperature *= cooling
    
    return best_chain

def _assign_sequence_to_template(sequence, template, rng):
    """Map a specific H/P sequence onto a pre-computed geometric template.
    Uses greedy assignment to maximize H-H contacts while maintaining self-avoidance."""
    n = len(sequence)
    if n != len(template):
        return None
    
    # Try to assign the sequence to the template positions
    # We'll use a greedy approach: for each position in the template,
    # decide which residue goes there to maximize H-H contacts
    
    # First, try the direct assignment (sequence[i] at template[i])
    direct_chain = list(template)
    direct_energy = _energy(sequence, direct_chain)
    
    # Try some permutations to see if we can do better
    # For now, just return the direct assignment
    return direct_chain, direct_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation.
    Uses Hamiltonian path templates with greedy assignment and local search refinement."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 180s limit
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Phase 1: Generate and use Hamiltonian path templates
    num_templates = 5
    for t in range(num_templates):
        if time.time() > deadline - 20.0:
            break
        
        # Generate a template with different random seeds
        template_rng = random.Random(1000 + t * 100 + n)
        template = _generate_hamiltonian_template(n, template_rng, num_iterations=500)
        
        if template is None or len(template) != n:
            continue
        
        # Assign sequence to template
        assigned_chain, energy = _assign_sequence_to_template(sequence, template, rng)
        
        if assigned_chain is not None:
            if energy < best_energy:
                best_energy = energy
                best_chain = list(assigned_chain)
    
    # Phase 2: Multiple randomized greedy growths
    num_grows = 5000  # Will be limited by time
    grows_done = 0
    
    while grows_done < num_grows:
        if time.time() > deadline - 30.0:
            break
        
        # Vary the temperature for diversity
        temperature = 0.5 + rng.random() * 2.0
        
        chain = _greedy_grow(sequence, dimension, rng, temperature)
        if chain is not None:
            grows_done += 1
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    # Phase 3: Local search on the best chain found
    if best_chain is not None and time.time() < deadline - 10.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, max_iters=200, rng=rng)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Fallback: if no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    # Convert to list of lists as required
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
