"""HP lattice protein folding using a Memetic Genetic Algorithm."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _is_adjacent(p, q):
    """Check if two points are adjacent (Manhattan distance 1)."""
    return sum(abs(a - b) for a, b in zip(p, q)) == 1


def _count_contacts(sequence, chain, where):
    """Count H-H non-consecutive contacts."""
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts


def _energy_from_contacts(contacts):
    return -contacts


def _initial_chain(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 1:
        return [(0,) * dimension]
    
    for attempt in range(50):
        chain = [(0,) * dimension]
        occupied = {chain[0]}
        success = True
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and len(chain) == n:
            return chain
    
    # Fallback: straight line
    return [(i,) + (0,) * (dimension - 1) for i in range(n)]


def _validate_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            if not _is_adjacent(chain[i], chain[i-1]):
                return False
    return True


def _hill_climb(sequence, chain, where, rng, max_steps=10):
    """
    Perform local search (Hill Climbing) on the chain.
    Returns the improved chain and its energy.
    """
    n = len(sequence)
    current_energy = _energy_from_contacts(_count_contacts(sequence, chain, where), )
    
    # Re-compute where and energy properly
    where = {p: i for i, p in enumerate(chain)}
    contacts = _count_contacts(sequence, chain, where)
    current_energy = _energy_from_contacts(contacts)
    
    for _ in range(max_steps):
        if n <= 2:
            break
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        neighbors = list(_neighbours(old_pos))
        rng.shuffle(neighbors)
        
        moved = False
        for new_pos in neighbors:
            # Must be empty
            if new_pos in where and where[new_pos] != i:
                continue
            # Must be adjacent to neighbors
            if not _is_adjacent(new_pos, chain[i - 1]):
                continue
            if not _is_adjacent(new_pos, chain[i + 1]):
                continue
            
            # Calculate delta energy
            # Old contacts involving i
            old_contacts = 0
            if sequence[i] == 'H':
                p = old_pos
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                        old_contacts += 1
            
            # Temporarily update
            del where[old_pos]
            where[new_pos] = i
            chain[i] = new_pos
            
            # New contacts involving i
            new_contacts = 0
            if sequence[i] == 'H':
                p = new_pos
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                        new_contacts += 1
            
            delta_contacts = new_contacts - old_contacts
            delta_energy = _energy_from_contacts(delta_contacts)
            
            if delta_energy < 0:
                current_energy += delta_energy
                moved = True
                break
            else:
                # Revert
                del where[new_pos]
                where[old_pos] = i
                chain[i] = old_pos
        
        if not moved:
            # Try a few more times before giving up? Or just break?
            # For efficiency, break if no move found in this iteration
            pass
            
    return chain, current_energy


def fold(sequence, dimension):
    """
    Find the best conformation using a Memetic Genetic Algorithm.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + len(sequence))
    time_limit = 8.0
    start_time = time.time()
    
    pop_size = 50
    population = []
    energies = []
    
    # Initialize population
    for _ in range(pop_size):
        chain = _initial_chain(sequence, dimension, rng)
        where = {p: i for i, p in enumerate(chain)}
        contacts = _count_contacts(sequence, chain, where)
        energy = _energy_from_contacts(contacts)
        population.append(chain)
        energies.append(energy)
    
    best_energy = min(energies)
    best_chain = population[energies.index(best_energy)]
    
    while time.time() - start_time < time_limit:
        # Selection: Tournament of 4
        idx1 = rng.randint(0, pop_size - 1)
        idx2 = rng.randint(0, pop_size - 1)
        
        # Find worst to replace
        worst_idx = max(range(pop_size), key=lambda i: energies[i])
        
        if rng.random() < 0.5:
            # Crossover
            # Try to find a valid cut
            offspring = None
            for _ in range(5): # Try 5 random cuts
                k = rng.randint(1, n - 2)
                # Combine P1[0:k] + P2[k:n]
                if idx1 != idx2:
                    p1 = population[idx1]
                    p2 = population[idx2]
                    # Check adjacency at cut
                    if _is_adjacent(p1[k-1], p2[k]):
                        candidate = p1[:k] + p2[k:]
                        if _validate_chain(candidate):
                            offspring = candidate
                            break
                else:
                    # Same parent, just copy?
                    offspring = list(population[idx1])
                    break
            
            if offspring is None:
                # Fallback to mutation if crossover failed
                offspring = list(population[idx1 if energies[idx1] < energies[idx2] else idx2])
        else:
            # Mutation
            parent_idx = idx1 if energies[idx1] < energies[idx2] else idx2
            offspring = list(population[parent_idx])
        
        # Hill Climbing on offspring
        where = {p: i for i, p in enumerate(offspring)}
        offspring, new_energy = _hill_climb(sequence, offspring, where, rng, max_steps=10)
        
        # Replace worst if offspring is better
        if new_energy < energies[worst_idx]:
            population[worst_idx] = offspring
            energies[worst_idx] = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = offspring
    
    # Final verification
    seen = set()
    valid = True
    for p in best_chain:
        if p in seen:
            valid = False
            break
        seen.add(p)
        
    if valid:
        for i in range(1, n):
            if not _is_adjacent(best_chain[i], best_chain[i-1]):
                valid = False
                break
    
    if not valid:
        best_chain = [(i,) + (0,) * (dimension - 1) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
