"""HP lattice protein folding using direct energy-driven search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts) where contacts are non-adjacent H residues on neighbouring lattice sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _build_chain_direct(sequence, rng):
    """Build a chain directly by growing from the center, placing H residues
    to maximize contacts with previously placed H residues."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    # Start with a small seed
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # We'll grow the chain by adding residues at either end
    # Track which positions in the sequence are placed
    placed = [False] * n
    placed[0] = True
    
    # Place residues in order of hydrophobicity first for the growing ends
    # Actually, let's just grow sequentially but with smart direction choice
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Score this position
            score = 0.0
            
            if sequence[i] == 'H':
                # Count potential H-H contacts
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10.0
            
            # Prefer positions that leave more free space
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free_count * 0.1
            
            # Add small random perturbation for diversity
            score += rng.random() * 0.5
            
            candidates.append((score, site))
        
        if not candidates:
            # Dead end - return None to indicate failure
            return None
        
        # Choose the best candidate
        candidates.sort(key=lambda x: -x[0])
        # With some probability, pick a non-best option for diversity
        if rng.random() < 0.1 and len(candidates) > 1:
            best_site = candidates[rng.randint(1, min(2, len(candidates) - 1))][1]
        else:
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _refine_chain(sequence, chain, rng, max_iter=200):
    """Refine a chain by trying local moves that improve energy."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
        
        # Try moving each interior residue to an adjacent empty position
        for i in range(1, n - 1):
            old_pos = best_chain[i]
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                # Check if the move maintains valid chain connectivity
                # new_pos must be adjacent to both chain[i-1] and chain[i+1]
                prev_pos = best_chain[i - 1]
                next_pos = best_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev != 1 or dist_next != 1:
                    continue
                
                # Check if new_pos is actually empty (not occupied by another residue)
                if new_pos in occupied:
                    continue
                
                # Perform the move temporarily
                old_occ = occupied[old_pos]
                del occupied[old_pos]
                occupied[new_pos] = old_occ
                best_chain[i] = new_pos
                
                new_energy = _energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                else:
                    # Revert
                    del occupied[new_pos]
                    occupied[old_pos] = old_occ
                    best_chain[i] = old_pos
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy

def _spiral_template(n, rng):
    """Generate a compact spiral-shaped self-avoiding walk of length n."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)  # East
    turns = 0
    
    for i in range(1, n):
        # Try to continue in current direction
        found = False
        for step in range(4):
            x, y = chain[-1]
            dx, dy = direction
            new_pos = (x + dx, y + dy)
            
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                found = True
                break
            
            # Turn right
            direction = (dy, -dx)
            turns += 1
        
        if not found:
            return None  # Dead end
    
    return chain

def _generate_spiral_templates(n, num_templates=20):
    """Pre-compute multiple spiral-like compact templates of length n.
    Returns a list of chains (each is a list of (x,y) tuples)."""
    templates = []
    seen = set()
    
    for t in range(num_templates):
        rng = random.Random(1000 + t)
        
        # Different starting directions and turn patterns
        start_dirs = [(1,0), (0,1), (-1,0), (0,-1)]
        start_dir = start_dirs[t % 4]
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        direction = start_dir
        
        for i in range(1, n):
            found = False
            for step in range(4):
                x, y = chain[-1]
                dx, dy = direction
                new_pos = (x + dx, y + dy)
                
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    found = True
                    break
                
                # Turn right (or left for some templates)
                if t % 3 == 0:
                    direction = (-dy, dx)  # left turn
                else:
                    direction = (dy, -dx)  # right turn
            
            if not found:
                break
        
        if len(chain) == n:
            # Normalize: shift so min x,y are 0
            min_x = min(p[0] for p in chain)
            min_y = min(p[1] for p in chain)
            normalized = tuple((p[0] - min_x, p[1] - min_y) for p in chain)
            if normalized not in seen:
                seen.add(normalized)
                templates.append(list(chain))
    
    return templates

def _assign_sequence_to_template(sequence, template_chain, rng):
    """Map a specific H/P sequence onto a geometric template using greedy
    assignment that maximizes H-H contacts. The template defines the spatial
    positions; we need to find a permutation of the sequence that fits the
    walk structure while maximizing contacts.
    
    Since the walk order is fixed by the template, we can't permute arbitrarily.
    Instead, we try different "orientations" of the sequence along the template:
    forward, reverse, and shifts.
    
    Actually, the chain order is fixed by the template geometry. What we can do
    is try multiple templates and pick the one where the natural assignment
    (sequence[i] at template[i]) gives the best energy. But that's just evaluating
    energy. 
    
    A better approach: generate templates where H positions are clustered.
    We'll score templates by how many H-H adjacencies they have in the spatial
    sense, then assign the sequence greedily.
    
    For now, this function evaluates the energy of placing the sequence directly
    on the template (position i gets sequence[i])."""
    n = len(sequence)
    if n != len(template_chain):
        return None, 0
    
    # Direct assignment: sequence[i] at template_chain[i]
    energy = _energy(sequence, template_chain)
    return template_chain, energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation.
    Uses a portfolio of direct construction methods plus local refinement."""
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s limit
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Strategy 1: Use pre-computed spiral templates with greedy assignment
    # Generate a diverse set of compact templates
    templates = _generate_spiral_templates(n, num_templates=30)
    
    for template in templates:
        if time.time() > deadline - 40.0:
            break
        
        # Try direct assignment
        chain, energy = _assign_sequence_to_template(sequence, template, rng)
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Also try reversed sequence assignment
        rev_sequence = sequence[::-1]
        chain_rev, energy_rev = _assign_sequence_to_template(rev_sequence, template, rng)
        if chain_rev is not None and energy_rev < best_energy:
            best_energy = energy_rev
            best_chain = list(chain_rev)
    
    # Strategy 2: Multiple randomized direct constructions
    num_attempts = 0
    max_attempts = 5000
    
    while num_attempts < max_attempts:
        if time.time() > deadline - 40.0:
            break
        
        # Alternate between different construction strategies
        strategy = rng.randint(0, 2)
        
        if strategy == 0:
            chain = _build_chain_direct(sequence, rng)
        elif strategy == 1:
            chain = _spiral_template(n, rng)
        else:
            # Another variant: grow with more randomness
            temp = 0.5 + rng.random() * 1.5
            chain = _build_chain_direct(sequence, rng)
        
        if chain is not None and len(chain) == n:
            num_attempts += 1
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    # Strategy 3: Local refinement on the best chain found
    if best_chain is not None and time.time() < deadline - 15.0:
        improved_chain, improved_energy = _refine_chain(sequence, best_chain, rng, max_iter=800)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Fallback: if no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    # Convert to list of lists as required
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
