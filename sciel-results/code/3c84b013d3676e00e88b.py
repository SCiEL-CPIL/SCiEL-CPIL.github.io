"""HP lattice protein folding with replica-exchange simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the energy of a conformation. Energy = -number of H-H contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            for q in _neighbours(chain[i]):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _delta_energy_move(sequence, chain, i, new_pos):
    """Compute the change in energy if residue i is moved to new_pos.
    Returns (delta_energy, success) where success is True if the move is valid."""
    n = len(chain)
    old_pos = chain[i]
    
    # Check if new_pos is already occupied by another residue
    # We need to check all residues except i itself
    for j, p in enumerate(chain):
        if j != i and p == new_pos:
            return 0, False
    
    # Check unit step constraints with neighbours
    if i > 0:
        diff = sum(abs(new_pos[a] - chain[i-1][a]) for a in range(3))
        if diff != 1:
            return 0, False
    if i < n - 1:
        diff = sum(abs(new_pos[a] - chain[i+1][a]) for a in range(3))
        if diff != 1:
            return 0, False
    
    # Compute energy change
    # Remove old contacts involving residue i
    old_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(old_pos):
            for j in range(n):
                if j != i and j != i - 1 and j != i + 1 and chain[j] == q and sequence[j] == 'H':
                    old_contacts += 1
    
    # Add new contacts involving residue i
    new_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(new_pos):
            for j in range(n):
                if j != i and j != i - 1 and j != i + 1 and chain[j] == q and sequence[j] == 'H':
                    new_contacts += 1
    
    delta = -(new_contacts - old_contacts)
    return delta, True


def _grow_greedy(sequence, dimension, rng):
    """Grow a compact initial conformation using greedy placement."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    for i in range(1, n):
        best_pos = None
        best_score = -1000
        
        for d in directions:
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos in occupied:
                continue
            
            # Score: prefer positions that create H-H contacts
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(new_pos):
                    if q in occupied:
                        # Check if the occupant is an H and not adjacent in chain
                        idx = chain.index(q) if q in chain else -1
                        if idx != -1 and idx < i - 1 and sequence[idx] == 'H':
                            score += 1
            
            # Prefer compact positions (closer to origin)
            score -= 0.1 * (abs(new_pos[0]) + abs(new_pos[1]) + abs(new_pos[2]))
            
            if score > best_score:
                best_score = score
                best_pos = new_pos
        
        if best_pos is None:
            # Fallback: try any available direction
            for d in directions:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                if new_pos not in occupied:
                    best_pos = new_pos
                    break
        
        if best_pos is None:
            break
        
        chain.append(best_pos)
        occupied.add(best_pos)
    
    # If we didn't place all residues, extend with a simple pattern
    while len(chain) < n:
        placed = False
        for d in directions:
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                placed = True
                break
        if not placed:
            last = chain[-1]
            for j in range(len(chain), n):
                chain.append((last[0] + (j - len(chain)), last[1], last[2]))
            break
    
    return chain


def _grow_compact(sequence, dimension, rng):
    """Grow a very compact initial conformation using a spiral pattern."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    # Spiral in the xy plane, occasionally moving in z
    dir_seq = [
        (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
        (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
        (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    di = 0
    for i in range(1, n):
        placed = False
        for offset in range(len(dir_seq)):
            d = dir_seq[(di + offset) % len(dir_seq)]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                di = (di + offset + 1) % len(dir_seq)
                placed = True
                break
        
        if not placed:
            # Try all 6 directions
            for d in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied.add(new_pos)
                    placed = True
                    break
        
        if not placed:
            last = chain[-1]
            for j in range(len(chain), n):
                chain.append((last[0] + (j - len(chain)), last[1], last[2]))
            break
    
    return chain


def _random_saw(sequence, dimension, rng):
    """Generate a random self-avoiding walk of the given length."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    for i in range(1, n):
        available = []
        for d in directions:
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                available.append(new_pos)
        
        if not available:
            return None
        
        next_pos = rng.choice(available)
        chain.append(next_pos)
        occupied.add(next_pos)
    
    return chain


def _try_single_move(sequence, chain, rng, time_budget):
    """Try a single-residue move (crankshaft-like)."""
    n = len(chain)
    if n < 4:
        return None, False
    
    # Choose a residue to move (not first or last)
    i = rng.randint(1, n - 2)
    old_pos = chain[i]
    
    # Try moving to a neighbouring position
    candidates = list(_neighbours(old_pos))
    rng.shuffle(candidates)
    
    for new_pos in candidates:
        # Check if new_pos is occupied
        occupied = set(chain)
        occupied.discard(old_pos)
        if new_pos in occupied:
            continue
        
        # Check unit step constraints
        if i > 0:
            diff = sum(abs(new_pos[a] - chain[i-1][a]) for a in range(3))
            if diff != 1:
                continue
        if i < n - 1:
            diff = sum(abs(new_pos[a] - chain[i+1][a]) for a in range(3))
            if diff != 1:
                continue
        
        # Compute energy change
        delta, success = _delta_energy_move(sequence, chain, i, new_pos)
        if success:
            new_chain = list(chain)
            new_chain[i] = new_pos
            return new_chain, delta
    
    return None, False


def _try_2opt_move(sequence, chain, rng):
    """Try a 2-opt move: reverse a segment of the chain."""
    n = len(chain)
    if n < 6:
        return None, False
    
    # Choose two cut points
    i = rng.randint(1, n - 4)
    j = rng.randint(i + 2, n - 1)
    
    # The segment to reverse is chain[i:j+1]
    # After reversal, chain[i] connects to what was chain[i-1] and chain[j] connects to what was chain[j+1]
    # We need to check that the new connections are valid
    
    # New chain: chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    # But we need to check that chain[i-1] is adjacent to chain[j] (new first of reversed segment)
    # and chain[j+1] is adjacent to chain[i] (new last of reversed segment)
    
    if i > 0:
        diff = sum(abs(chain[i-1][a] - chain[j][a]) for a in range(3))
        if diff != 1:
            return None, False
    
    if j < n - 1:
        diff = sum(abs(chain[i][a] - chain[j+1][a]) for a in range(3))
        if diff != 1:
            return None, False
    
    # Build new chain
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None, False
        seen.add(p)
    
    # Compute energy change
    old_energy = _compute_energy(sequence, chain)
    new_energy = _compute_energy(sequence, new_chain)
    delta = new_energy - old_energy
    
    return new_chain, delta


def _local_search_sa(sequence, chain, time_budget, rng, T_start=2.0, T_end=0.05):
    """Run local search with simulated annealing."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = T_start
    iterations = 0
    max_iterations = 200000
    
    while T > T_end and iterations < max_iterations:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Single move
            new_chain, result = _try_single_move(sequence, chain, rng, time_budget)
            if new_chain is None:
                iterations += 1
                T *= 0.9995
                continue
            delta = result
        elif move_type < 0.85:
            # 2-opt move
            new_chain, result = _try_2opt_move(sequence, chain, rng)
            if new_chain is None:
                iterations += 1
                T *= 0.9995
                continue
            delta = result
        else:
            # Small random walk
            new_chain = None
            for _ in range(3):
                mc, res = _try_single_move(sequence, chain, rng, time_budget)
                if mc is not None:
                    chain = mc
                    current_energy += res
            new_chain = chain
            delta = 0
        
        if new_chain is None:
            iterations += 1
            T *= 0.9995
            continue
        
        # Accept or reject
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy += delta if new_chain != chain else 0
            # Recompute energy for safety
            current_energy = _compute_energy(sequence, chain)
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        iterations += 1
        T *= 0.9995
        
        # Time check
        if time_budget > 0 and time.time() > time_budget:
            break
    
    return best_chain, best_energy


def _replica_exchange_sa(sequence, dimension, time_budget, rng, num_replicas=4):
    """Run replica-exchange simulated annealing."""
    n = len(sequence)
    
    # Create initial replicas with different initializations
    replicas = []
    for r in range(num_replicas):
        if r == 0:
            chain = _grow_greedy(sequence, dimension, rng)
        elif r == 1:
            chain = _grow_compact(sequence, dimension, rng)
        else:
            chain = None
            for _ in range(50):
                chain = _random_saw(sequence, dimension, rng)
                if chain is not None and len(chain) == n:
                    break
            if chain is None or len(chain) != n:
                chain = _grow_greedy(sequence, dimension, rng)
        replicas.append(chain)
    
    # Track best
    best_chain = None
    best_energy = 0
    
    for r in range(num_replicas):
        e = _compute_energy(sequence, replicas[r])
        if e < best_energy:
            best_energy = e
            best_chain = list(replicas[r])
    
    # Run replica exchange
    T_values = [5.0, 3.0, 1.5, 0.5][:num_replicas]
    
    while time_budget > 0 and time.time() < time_budget:
        # Run a few iterations of SA on each replica
        for r in range(num_replicas):
            if time.time() >= time_budget:
                break
            
            chain = replicas[r]
            current_energy = _compute_energy(sequence, chain)
            
            # Run a batch of SA iterations
            for _ in range(500):
                if time.time() >= time_budget:
                    break
                
                move_type = rng.random()
                if move_type < 0.6:
                    new_chain, result = _try_single_move(sequence, chain, rng, time_budget)
                    if new_chain is None:
                        continue
                    delta = result
                else:
                    new_chain, result = _try_2opt_move(sequence, chain, rng)
                    if new_chain is None:
                        continue
                    delta = result
                
                if delta < 0 or rng.random() < math.exp(-delta / T_values[r]):
                    chain = new_chain
                    current_energy += delta
            
            replicas[r] = chain
            e = _compute_energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = list(chain)
        
        # Try exchanges between neighbouring replicas
        for r in range(num_replicas - 1):
            if time.time() >= time_budget:
                break
            
            e1 = _compute_energy(sequence, replicas[r])
            e2 = _compute_energy(sequence, replicas[r+1])
            
            # Metropolis criterion for exchange
            delta_E = T_values[r] * e2 - T_values[r+1] * e1
            if delta_E < 0 or rng.random() < math.exp(-delta_E):
                replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
        
        # Cool down temperatures slightly
        for r in range(num_replicas):
            T_values[r] *= 0.999
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold the sequence using replica-exchange simulated annealing."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]
    if n == 3:
        return [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    
    rng = random.Random(42 + len(sequence))
    
    # Use replica exchange SA
    time_budget = deadline - time.time()
    best_chain, best_energy = _replica_exchange_sa(sequence, dimension, deadline, rng, num_replicas=4)
    
    if best_chain is None:
        # Fallback
        best_chain = _grow_greedy(sequence, dimension, rng)
        best_energy = _compute_energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
