"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_chain(sequence, dimension, rng):
    """Create an initial self-avoiding random walk."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    direction = (1, 0, 0) if dimension >= 1 else ()
    
    for i in range(1, n):
        # Try to continue in the same direction first, then random
        candidates = []
        # Same direction
        new_pos = tuple(chain[-1][d] + direction[d] for d in range(dimension))
        if new_pos not in occupied:
            candidates.append(new_pos)
        
        # Random neighbors
        for site in _neighbours(chain[-1]):
            if site not in occupied and site not in candidates:
                candidates.append(site)
        
        if not candidates:
            # Backtrack: try all directions
            for site in _neighbours(chain[-1]):
                if site not in occupied:
                    candidates.append(site)
        
        if not candidates:
            return None
        
        site = candidates[rng.randint(0, len(candidates) - 1)]
        chain.append(site)
        occupied[site] = i
        
        # Update direction
        direction = tuple(site[d] - chain[i-1][d] for d in range(dimension))
    
    return chain

def _contacts_at(sequence, chain, occupied, pos, idx):
    """Count H-H contacts involving position pos at index idx."""
    if sequence[idx] != "H":
        return 0
    count = 0
    for q in _neighbours(pos):
        j = occupied.get(q)
        if j is not None and j > idx + 1 and sequence[j] == "H":
            count += 1
    return count

def _simulate_annealing(sequence, dimension, rng, max_iters=50000, t0=1.0, t1=0.01):
    """Simulated annealing with pivot and single-residue moves."""
    n = len(sequence)
    
    # Initialize
    chain = _initial_chain(sequence, dimension, rng)
    if chain is None:
        # Fallback: straight line
        chain = [(i,) + (0,) * (dimension - 1) for i in range(n)]
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    # Temperature schedule
    cooling = (t1 / t0) ** (1.0 / max(1, max_iters))
    
    for it in range(max_iters):
        T = t0 * (cooling ** it)
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single residue move
            idx = rng.randint(0, n - 1)
            old_pos = chain[idx]
            candidates = [q for q in _neighbours(old_pos) if q not in occupied]
            if not candidates:
                continue
            new_pos = candidates[rng.randint(0, len(candidates) - 1)]
            
            # Check self-avoidance: new_pos not in occupied (already checked)
            # Compute energy delta
            # Remove contacts at old_pos, add contacts at new_pos
            # But need to be careful: contacts with adjacent residues in chain shouldn't count
            
            # Un-occupy old_pos
            del occupied[old_pos]
            
            # Count contacts lost at old_pos
            lost = 0
            if sequence[idx] == "H":
                for q in _neighbours(old_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - idx) > 1 and sequence[j] == "H":
                        lost += 1
            
            # Occupy new_pos
            occupied[new_pos] = idx
            
            # Count contacts gained at new_pos
            gained = 0
            if sequence[idx] == "H":
                for q in _neighbours(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - idx) > 1 and sequence[j] == "H":
                        gained += 1
            
            delta = -(gained - lost)
            
            # Accept or reject
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                chain[idx] = new_pos
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Revert
                del occupied[new_pos]
                occupied[old_pos] = idx
        
        elif move_type < 0.8:
            # Pivot move
            if n < 3:
                continue
            k = rng.randint(1, n - 1)
            pivot = chain[k - 1]
            suffix = chain[k:]
            
            # Try rotating suffix around pivot
            # The suffix must be re-placed starting from a neighbor of pivot
            # Simple approach: try to place suffix as a random walk from each neighbor of pivot
            success = False
            for _ in range(5):  # Try a few times
                # Choose a starting direction
                start_candidates = [q for q in _neighbours(pivot) if q != chain[k]]
                if not start_candidates:
                    continue
                new_start = start_candidates[rng.randint(0, len(start_candidates) - 1)]
                
                # Try to place the rest of the suffix as a random walk
                new_suffix = [new_start]
                temp_occupied = set(occupied.keys())
                temp_occupied.discard(chain[k])  # We're moving the whole suffix
                for p in suffix[1:]:
                    temp_occupied.discard(p)
                
                valid = True
                for i in range(1, len(suffix)):
                    last = new_suffix[-1]
                    candidates = [q for q in _neighbours(last) if q not in temp_occupied]
                    if not candidates:
                        valid = False
                        break
                    site = candidates[rng.randint(0, len(candidates) - 1)]
                    new_suffix.append(site)
                    temp_occupied.add(site)
                
                if not valid:
                    continue
                
                # Check that new_suffix doesn't collide with chain[0:k]
                prefix_set = set(chain[:k])
                if any(p in prefix_set for p in new_suffix):
                    continue
                
                # Compute energy delta
                # Remove old suffix contacts, add new suffix contacts
                old_energy_suffix = 0
                # This is complex; let's just recompute full energy for simplicity
                # But that's O(n^2). For now, let's accept and recompute.
                
                # Actually, let's compute delta properly
                # Contacts lost: all H-H contacts involving any residue in old suffix
                # Contacts gained: all H-H contacts involving any residue in new suffix
                # But this is tricky. Let's just recompute energy.
                
                new_chain = chain[:k] + new_suffix
                new_occupied = {p: i for i, p in enumerate(new_chain)}
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                
                if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                    chain = new_chain
                    occupied = new_occupied
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                success = True
                break
            
            # If no successful move, continue
        
        else:
            # Crankshaft move
            if n < 4:
                continue
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 1, n - 1)
            if j - i < 2:
                continue
            
            anchor1 = chain[i]
            anchor2 = chain[j]
            segment = chain[i + 1:j]
            
            # Try to re-place segment between anchor1 and anchor2
            success = False
            for _ in range(3):
                # Start from a neighbor of anchor1
                start_candidates = [q for q in _neighbours(anchor1) if q != chain[i + 1]]
                if not start_candidates:
                    continue
                new_start = start_candidates[rng.randint(0, len(start_candidates) - 1)]
                
                # Try to place segment as random walk ending near anchor2
                new_seg = [new_start]
                temp_occupied = set(occupied.keys())
                for p in segment:
                    temp_occupied.discard(p)
                
                valid = True
                for idx in range(1, len(segment)):
                    last = new_seg[-1]
                    candidates = [q for q in _neighbours(last) if q not in temp_occupied]
                    if not candidates:
                        valid = False
                        break
                    site = candidates[rng.randint(0, len(candidates) - 1)]
                    new_seg.append(site)
                    temp_occupied.add(site)
                
                if not valid:
                    continue
                
                # Check that last position is adjacent to anchor2
                if not any(abs(new_seg[-1][d] - anchor2[d]) == 1 for d in range(dimension)) or \
                   sum(abs(new_seg[-1][d] - anchor2[d]) for d in range(dimension)) != 1:
                    continue
                
                # Check no collision with anchors or outside
                prefix_set = set(chain[:i + 1]) | set(chain[j:])
                if any(p in prefix_set for p in new_seg):
                    continue
                
                new_chain = chain[:i + 1] + new_seg + chain[j:]
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                
                if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                    chain = new_chain
                    occupied = {p: idx for idx, p in enumerate(chain)}
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                success = True
                break
        
        # Periodically recompute energy to avoid drift
        if it % 1000 == 0:
            current_energy = _energy(sequence, chain)
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1  # Will be negative
    
    # Number of restarts based on chain length
    if n <= 48:
        n_restarts = 5
        max_iters = 10000
    elif n <= 80:
        n_restarts = 3
        max_iters = 8000
    else:
        n_restarts = 2
        max_iters = 6000
    
    for r in range(n_restarts):
        chain = _simulate_annealing(sequence, dimension, rng, max_iters=max_iters, t0=1.0, t1=0.01)
        if chain is not None:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
    
    if best_chain is None:
        best_chain = [(i,) + (0,) * (dimension - 1) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
