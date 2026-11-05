"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_energy(sequence, chain):
    """Compute the energy of a chain conformation."""
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
    """One randomized greedy chain growth."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
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
            options.append((gain + 0.1 * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _mc_optimize(sequence, chain, rng, max_steps, deadline):
    """Monte Carlo local search with crank moves to escape local minima."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    energy = _initial_energy(sequence, chain)
    best_energy = energy
    best_chain = list(chain)
    
    # Precompute H positions for quick lookup
    is_h = [c == "H" for c in sequence]
    
    # Temperature schedule
    temp = 2.0
    cooling = 0.999
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Try each of the 6 directions
        for axis in range(3):
            for step_dir in (-1, 1):
                new_pos = list(old_pos)
                new_pos[axis] += step_dir
                new_pos = tuple(new_pos)
                
                # Check if the move is valid (not occupied by another residue)
                if new_pos in where:
                    # Check if it's the adjacent residue (which we're moving away from)
                    occupant = where[new_pos]
                    if abs(occupant - i) == 1:
                        # This would be a self-adjacent move, skip
                        continue
                    else:
                        # Occupied by non-adjacent residue, invalid
                        continue
                
                # Calculate energy delta
                delta = 0
                
                # Remove contacts involving residue i at old position
                if is_h[i]:
                    for q in _neighbours(old_pos):
                        j = where.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                            delta += 1  # Removing a contact increases energy (less negative)
                
                # Add contacts involving residue i at new position
                if is_h[i]:
                    for q in _neighbours(new_pos):
                        j = where.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                            delta -= 1  # Adding a contact decreases energy (more negative)
                
                # Metropolis acceptance
                if delta < 0 or rng.random() < pow(2.718281828, -delta / temp):
                    # Accept the move
                    where.pop(old_pos)
                    where[new_pos] = i
                    chain[i] = new_pos
                    energy += delta
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = list(chain)
                    break  # Only try one direction per step
        
        temp *= cooling
        if temp < 0.01:
            break
    
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, rng, max_steps, deadline):
    """Simulated annealing with pivot and crank moves."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    energy = _initial_energy(sequence, chain)
    best_energy = energy
    best_chain = list(chain)
    is_h = [c == "H" for c in sequence]
    
    temp = 3.0
    cooling = 0.9995
    no_improve = 0
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Randomly choose move type: pivot (60%) or single-residue move (40%)
        if rng.random() < 0.6:
            # Pivot move: pick a random internal point and try to rotate
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            prev_pos = chain[i - 1]
            
            # Find valid new positions that maintain connectivity with prev_pos
            valid_moves = []
            for axis in range(3):
                for step_dir in (-1, 1):
                    new_pos = list(old_pos)
                    new_pos[axis] += step_dir
                    new_pos = tuple(new_pos)
                    # Must be adjacent to prev_pos
                    if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) + abs(new_pos[2] - prev_pos[2]) == 1:
                        if new_pos != old_pos:
                            # Check if new_pos is occupied
                            if new_pos in where:
                                occupant = where[new_pos]
                                if occupant == i + 1:
                                    continue  # Would collide with next residue
                                else:
                                    continue  # Occupied by other
                            valid_moves.append(new_pos)
            
            if not valid_moves:
                continue
            
            new_pos = rng.choice(valid_moves)
            
            # Calculate energy delta for moving residue i
            delta = 0
            if is_h[i]:
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                        delta += 1
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                        delta -= 1
            
            if delta < 0 or rng.random() < pow(2.718281828, -delta / temp):
                where.pop(old_pos)
                where[new_pos] = i
                chain[i] = new_pos
                energy += delta
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            # Single residue move
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            
            for axis in range(3):
                for step_dir in (-1, 1):
                    new_pos = list(old_pos)
                    new_pos[axis] += step_dir
                    new_pos = tuple(new_pos)
                    
                    if new_pos in where:
                        occupant = where[new_pos]
                        if abs(occupant - i) == 1:
                            continue
                        else:
                            continue
                    
                    delta = 0
                    if is_h[i]:
                        for q in _neighbours(old_pos):
                            j = where.get(q)
                            if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                                delta += 1
                        for q in _neighbours(new_pos):
                            j = where.get(q)
                            if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                                delta -= 1
                    
                    if delta < 0 or rng.random() < pow(2.718281828, -delta / temp):
                        where.pop(old_pos)
                        where[new_pos] = i
                        chain[i] = new_pos
                        energy += delta
                        if energy < best_energy:
                            best_energy = energy
                            best_chain = list(chain)
                            no_improve = 0
                        else:
                            no_improve += 1
                        break
        
        temp *= cooling
        if temp < 0.05:
            break
        if no_improve > 500:
            # Restart with higher temperature
            temp = 1.0
            no_improve = 0
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a protein sequence on the cubic lattice using greedy growth + Monte Carlo optimization."""
    start_time = time.time()
    deadline = start_time + 220  # Safety margin before 330s limit
    
    rng = random.Random(42 + len(sequence))
    
    n = len(sequence)
    
    # Generate multiple initial conformations via greedy growth
    initial_chains = []
    max_initials = min(50, max(5, 300 // max(1, n)))
    
    for _ in range(max_initials):
        if time.time() > deadline - 60:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Optimize each initial conformation
    best_chain = None
    best_energy = 0
    
    # Sort by initial energy, optimize best ones first
    initial_chains.sort(key=lambda c: _initial_energy(sequence, c))
    
    # Optimize the top few initial conformations
    num_to_optimize = min(len(initial_chains), max(3, 20 // max(1, n // 20)))
    
    for chain in initial_chains[:num_to_optimize]:
        if time.time() > deadline - 10:
            break
        
        # Use simulated annealing for optimization
        opt_chain, opt_energy = _simulated_annealing(sequence, chain, rng, 
                                                       max_steps=max(1000, 50000 // n),
                                                       deadline=deadline - 10)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
    
    # If we still have time, try more optimizations on the best chain found
    if best_chain is not None and time.time() < deadline - 30:
        # Do additional rounds of SA on the best chain
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            opt_chain, opt_energy = _simulated_annealing(sequence, best_chain, rng,
                                                           max_steps=max(500, 20000 // n),
                                                           deadline=deadline - 10)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain
    
    if best_chain is None:
        best_chain = initial_chains[0]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
