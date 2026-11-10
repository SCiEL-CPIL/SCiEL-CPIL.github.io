"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _turns_to_chain(turns, start_dir=0):
    """
    Convert a list of turn directions (0=Left, 1=Right, 2=Straight) to a chain of points.
    start_dir: 0=Up, 1=Right, 2=Down, 3=Left
    Returns list of tuples if valid (self-avoiding), else None.
    """
    if not turns:
        return [(0, 0)]
    
    # Directions: 0=Up, 1=Right, 2=Down, 3=Left
    # dx, dy for each direction
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    
    # Initial position
    x, y = 0, 0
    chain = [(x, y)]
    occupied = {(x, y): 0}
    
    current_dir = start_dir
    
    for i, t in enumerate(turns):
        # t: 0=Left, 1=Right, 2=Straight
        if t == 0:
            current_dir = (current_dir + 3) % 4
        elif t == 1:
            current_dir = (current_dir + 1) % 4
        # t == 2: Straight, no change
        
        x += dx[current_dir]
        y += dy[current_dir]
        pos = (x, y)
        
        if pos in occupied:
            return None  # Self-avoidance violated
        
        occupied[pos] = i + 1
        chain.append(pos)
        
    return chain


def _random_turns(n_residues, rng):
    """Generate a random turn vector for a chain of n_residues residues."""
    # Number of turns = n_residues - 1
    n_turns = max(0, n_residues - 1)
    return [rng.randint(0, 2) for _ in range(n_turns)]


def _evaluate_individual(turns, sequence, start_dir=0):
    """
    Evaluate a turn vector: returns (energy, is_valid).
    If invalid, energy is a large positive number (penalty).
    """
    chain = _turns_to_chain(turns, start_dir)
    if chain is None:
        return 1000, False  # Penalty for invalid
    energy = _compute_energy(sequence, chain)
    return energy, True


def _crossover(turns1, turns2, rng):
    """Single-point crossover."""
    if not turns1 or not turns2:
        return turns1, turns2
    point = rng.randint(1, len(turns1) - 1) if len(turns1) > 1 else 0
    child1 = turns1[:point] + turns2[point:]
    child2 = turns2[:point] + turns1[point:]
    return child1, child2


def _mutate(turns, rng, mutation_rate=0.1):
    """Mutation: randomly change some turn directions."""
    mutated = list(turns)
    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            mutated[i] = rng.randint(0, 2)
    return mutated


def _greedy_init(sequence, rng):
    """
    Generate a valid initial population using greedy growth, then encode as turns.
    """
    n = len(sequence)
    if n < 2:
        return [[], (0, 0)]
    
    # Simple greedy growth to get a valid chain
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    current_dir = 0  # Up
    
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    
    for i in range(1, n):
        last = chain[-1]
        # Try all 4 directions, prefer those that don't collide
        options = []
        for d in range(4):
            nx = last[0] + dx[d]
            ny = last[1] + dy[d]
            pos = (nx, ny)
            if pos not in occupied:
                options.append(d)
        
        if not options:
            return None, None
        
        # Pick randomly from valid options
        d = rng.choice(options)
        nx = last[0] + dx[d]
        ny = last[1] + dy[d]
        pos = (nx, ny)
        occupied[pos] = i
        chain.append(pos)
        
        # Determine turn relative to current_dir
        if d == current_dir:
            turn = 2  # Straight
        elif (d - current_dir) % 4 == 1:
            turn = 1  # Right
        elif (d - current_dir) % 4 == 3:
            turn = 0  # Left
        else:
            turn = 2  # Should not happen, but safe fallback
        
        current_dir = d
    
    # Now encode chain as turns
    turns = []
    for i in range(1, n):
        prev = chain[i-1]
        curr = chain[i]
        # Determine direction from prev to curr
        dx_val = curr[0] - prev[0]
        dy_val = curr[1] - prev[1]
        if dx_val == 0 and dy_val == 1:
            d = 0
        elif dx_val == 1 and dy_val == 0:
            d = 1
        elif dx_val == 0 and dy_val == -1:
            d = 2
        elif dx_val == -1 and dy_val == 0:
            d = 3
        else:
            d = 0  # Should not happen
        
        if i == 1:
            # First turn is relative to start_dir=0
            if d == 0:
                turn = 2
            elif d == 1:
                turn = 1
            elif d == 3:
                turn = 0
            else:
                turn = 2
        else:
            # Get previous direction
            prev_prev = chain[i-2]
            dx_prev = prev[0] - prev_prev[0]
            dy_prev = prev[1] - prev_prev[1]
            if dx_prev == 0 and dy_prev == 1:
                d_prev = 0
            elif dx_prev == 1 and dy_prev == 0:
                d_prev = 1
            elif dx_prev == 0 and dy_prev == -1:
                d_prev = 2
            elif dx_prev == -1 and dy_prev == 0:
                d_prev = 3
            else:
                d_prev = 0
            
            if d == d_prev:
                turn = 2
            elif (d - d_prev) % 4 == 1:
                turn = 1
            elif (d - d_prev) % 4 == 3:
                turn = 0
            else:
                turn = 2
        
        turns.append(turn)
    
    return turns, chain


def _ga_fold(sequence, rng):
    """
    Genetic Algorithm on turn vectors.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)]
    
    pop_size = 100
    max_generations = 5000
    mutation_rate = 0.05
    elitism = 5
    
    # Initialize population
    population = []
    # Start with a few greedy inits
    for _ in range(min(10, pop_size)):
        turns, chain = _greedy_init(sequence, rng)
        if turns is not None:
            population.append(turns)
    
    # Fill rest with random valid turns (retry until valid or give up)
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 10:
        turns = _random_turns(n, rng)
        chain = _turns_to_chain(turns, 0)
        if chain is not None:
            population.append(turns)
        attempts += 1
    
    if not population:
        # Fallback: straight line
        return [(i, 0) for i in range(n)]
    
    # Evaluate initial population
    def eval_pop(pop):
        results = []
        for turns in pop:
            energy, valid = _evaluate_individual(turns, sequence, 0)
            results.append((energy, valid, turns))
        return results
    
    evaluated = eval_pop(population)
    
    best_energy = min(e for e, v, t in evaluated if v)
    best_turns = None
    for e, v, t in evaluated:
        if v and e < best_energy:
            best_energy = e
            best_turns = t
    
    for gen in range(max_generations):
        if time.time() > _DEADLINE - 20:
            break
        
        # Sort by energy (valid first, then invalid)
        evaluated.sort(key=lambda x: (not x[1], x[0]))
        
        # Elitism
        elites = [t for e, v, t in evaluated[:elitism]]
        
        # Selection and reproduction
        new_population = list(elites)
        
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                t1 = rng.randint(0, len(evaluated) - 1)
                t2 = rng.randint(0, len(evaluated) - 1)
                if evaluated[t1][1] and not evaluated[t2][1]:
                    return evaluated[t1][2]
                if evaluated[t2][1] and not evaluated[t1][1]:
                    return evaluated[t2][2]
                if evaluated[t1][0] <= evaluated[t2][0]:
                    return evaluated[t1][2]
                return evaluated[t2][2]
            
            parent1 = select()
            parent2 = select()
            
            child1, child2 = _crossover(parent1, parent2, rng)
            child1 = _mutate(child1, rng, mutation_rate)
            child2 = _mutate(child2, rng, mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
        evaluated = eval_pop(population)
        
        # Track best
        for e, v, t in evaluated:
            if v and e < best_energy:
                best_energy = e
                best_turns = t
    
    # Convert best turns to chain
    if best_turns is not None:
        chain = _turns_to_chain(best_turns, 0)
        if chain is not None:
            return chain
    
    # Fallback
    return [(i, 0) for i in range(n)]


def _local_improve(sequence, chain, rng, max_moves=2000):
    """
    Improve conformation via single-residue moves that preserve connectivity.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 10:
            break
        
        i = rng.randint(1, n - 2)
        p = chain[i]
        tried = False

        for np in _neighbours(p):
            if np == p:
                continue
            if np in occupied and occupied[np] != i:
                continue
            
            # Check if moving to np preserves connectivity
            prev, nxt = chain[i-1], chain[i+1]
            dist_prev = abs(np[0]-prev[0]) + abs(np[1]-prev[1])
            dist_next = abs(np[0]-nxt[0]) + abs(np[1]-nxt[1])
            
            if dist_prev != 1 or dist_next != 1:
                continue
            
            # Perform move
            del occupied[p]
            chain[i] = np
            occupied[np] = i
            new_energy = _compute_energy(sequence, chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                no_improve_count = 0
            else:
                # Revert
                del occupied[np]
                chain[i] = p
                occupied[p] = i
                no_improve_count += 1
            
            tried = True
            break
        
        if not tried:
            no_improve_count += 1
        
        if no_improve_count > 500:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using a Genetic Algorithm on turn vectors,
    followed by local improvement.
    """
    rng = random.Random(42)
    
    n = len(sequence)
    if n < 2:
        return [[0, 0]]
    
    # Phase 1: Genetic Algorithm on turn vectors
    chain = _ga_fold(sequence, rng)
    
    # Phase 2: Local improvement
    if chain is not None:
        improved_chain, improved_energy = _local_improve(sequence, chain, rng, max_moves=5000)
        chain = improved_chain
    
    # Fallback: straight line
    if chain is None:
        chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
