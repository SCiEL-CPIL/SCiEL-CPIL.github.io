"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0

# Directions: 0:R, 1:D, 2:L, 3:U
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
VEC_TO_IDX = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _chain_to_steps(chain):
    steps = []
    for i in range(1, len(chain)):
        dx = chain[i][0] - chain[i-1][0]
        dy = chain[i][1] - chain[i-1][1]
        steps.append(VEC_TO_IDX[(dx, dy)])
    return steps

def _steps_to_chain(steps):
    chain = [(0, 0)]
    for d in steps:
        dx, dy = DIRS[d]
        chain.append((chain[-1][0] + dx, chain[-1][1] + dy))
    return chain

def _generate_random_saw(n, rng):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
        
    start_dir = rng.randint(0, 3)
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        options = []
        for d in range(4):
            dx, dy = DIRS[d]
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pos not in occupied:
                options.append(d)
                
        if not options:
            return None
            
        next_dir = rng.choice(options)
        dx, dy = DIRS[next_dir]
        new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
        chain.append(new_pos)
        occupied.add(new_pos)
        
    return chain

def _pivot_move(chain, bond_idx, rng):
    """
    Perform a pivot move: rotate the segment from bond_idx onwards.
    bond_idx is the index of the bond to rotate (0 to n-2).
    """
    n = len(chain)
    if bond_idx < 0 or bond_idx >= n - 1:
        return None
        
    steps = _chain_to_steps(chain)
    rot = rng.choice([1, 2, 3])
    
    new_steps = steps[:]
    for i in range(bond_idx, len(new_steps)):
        new_steps[i] = (new_steps[i] + rot) % 4
        
    new_chain = _steps_to_chain(new_steps)
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
        
    return new_chain

def _local_improve_pivot(sequence, chain, rng, max_iters=2000, start_temp=3.0, cool_rate=0.997):
    """
    Improve conformation using pivot moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
        
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.05:
            break
            
        # Pick a random bond to pivot
        b = rng.randint(0, n - 2)
        
        new_chain = _pivot_move(chain, b, rng)
        
        if new_chain is None:
            temp *= cool_rate
            no_improve += 1
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        temp *= cool_rate
        
        if no_improve > 500:
            break
            
    return best_chain, best_energy

def _ga_crossover(sequence, chain1, chain2, rng):
    """
    Genetic Algorithm crossover with error repair.
    1. Convert chains to step sequences.
    2. Perform uniform crossover on steps.
    3. Reconstruct chain.
    4. If invalid (collision), try to repair by local moves or return None.
    """
    n = len(sequence)
    
    steps1 = _chain_to_steps(chain1)
    steps2 = _chain_to_steps(chain2)
    
    # Uniform crossover
    new_steps = []
    for i in range(len(steps1)):
        if rng.random() < 0.5:
            new_steps.append(steps1[i])
        else:
            new_steps.append(steps2[i])
            
    # Mutation
    for i in range(len(new_steps)):
        if rng.random() < 0.1:
            new_steps[i] = rng.randint(0, 3)
            
    new_chain = _steps_to_chain(new_steps)
    
    # Check validity
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
        
    return new_chain

def _repair_chain(sequence, chain, rng):
    """
    Error repair strategy:
    If the chain has collisions, try to resolve them by moving the later residue
    to a nearby free site, or by regrowing the tail.
    """
    n = len(chain)
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            # Collision at i. Try to move residue i to a nearby free site.
            # Or, simpler: regrow the tail from i-1.
            # Let's try a simple local move first.
            moved = False
            for _ in range(10):
                # Try to move residue i to a random neighbour of its original position
                # that is free and not occupied by 0..i-1.
                x, y = p
                options = []
                for dx, dy in DIRS:
                    np = (x + dx, y + dy)
                    if np not in seen:
                        # Check if np is adjacent to i-1 (must be, since it's a walk)
                        # Actually, moving i might break the bond with i-1.
                        # So we must move i such that it is adjacent to i-1.
                        # i-1 is chain[i-1].
                        prev = chain[i-1]
                        if np in [(prev[0]+1, prev[1]), (prev[0]-1, prev[1]), (prev[0], prev[1]+1), (prev[0], prev[1]-1)]:
                            options.append(np)
                            
                if not options:
                    break
                    
                new_p = rng.choice(options)
                # Check if new_p is free in seen (excluding i itself, which we are moving)
                # seen contains 0..i-1 and maybe i? No, we broke out of loop when i was in seen.
                # So seen contains 0..i-1.
                if new_p not in seen:
                    chain[i] = new_p
                    moved = True
                    break
                    
            if not moved:
                # Regrow tail from i-1
                head = chain[:i]
                tail_len = n - i
                # Generate random SAW of length tail_len starting from head[-1]
                # This is complex. Let's just return None if we can't fix it easily.
                return None
            # Update seen? No, we just return the modified chain.
            # We need to re-check the rest of the chain.
            # This simple repair might not be enough.
            # Let's just return the chain and let the caller verify.
            return chain
        seen.add(p)
        
    return chain

def _continuous_fold(sequence, n, rng):
    """
    Formulate the folding problem as a continuous 2D vector field optimization.
    The chain is represented by a differentiable path in 2D.
    Uses a soft self-avoidance penalty and a hard H-H contact reward.
    After converging the continuous solution, snap the coordinates to the nearest
    valid self-avoiding walk on the integer lattice.
    """
    # Initialize with a random walk
    x = np.zeros((n, 2))
    x[0] = [0.0, 0.0]
    for i in range(1, n):
        dx, dy = rng.choice([(1, 0), (0, -1), (-1, 0), (0, 1)])
        x[i] = x[i-1] + [dx, dy]
    
    # Optimization parameters
    lr = 0.01
    penalty_weight = 5.0
    contact_weight = 1.0
    max_iters = 500
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 20:
            break
        
        # Compute soft self-avoidance penalty
        # Penalty for pairs (i, j) with |i - j| > 1 that are too close
        penalty = 0.0
        grad_penalty = np.zeros_like(x)
        
        for i in range(n):
            for j in range(i + 2, n):
                dist_sq = np.sum((x[i] - x[j]) ** 2)
                if dist_sq < 2.0:  # Too close
                    # Gradient of penalty: push apart
                    diff = x[i] - x[j]
                    norm = np.sqrt(dist_sq) + 1e-6
                    force = penalty_weight / (norm * norm)
                    grad_penalty[i] += force * diff
                    grad_penalty[j] -= force * diff
                    penalty += penalty_weight * (2.0 - dist_sq)
        
        # Compute H-H contact reward
        # Reward for pairs (i, j) with |i - j| > 1 where both are H and distance is ~1
        reward = 0.0
        grad_reward = np.zeros_like(x)
        
        for i in range(n):
            if sequence[i] == 'H':
                for j in range(i + 2, n):
                    if sequence[j] == 'H':
                        dist = np.sum((x[i] - x[j]) ** 2)
                        # Reward if distance is close to 1 (unit step)
                        if 0.5 < dist < 1.5:
                            reward += contact_weight
                            # Gradient: pull together if slightly too far, push apart if slightly too close
                            diff = x[i] - x[j]
                            norm = np.sqrt(dist) + 1e-6
                            # We want dist ~ 1, so force proportional to (1 - dist)
                            force = 0.1 * (1.0 - dist) / norm
                            grad_reward[i] += force * diff
                            grad_reward[j] -= force * diff
        
        # Update positions
        grad = -grad_penalty + grad_reward
        x = x + lr * grad
        
        # Enforce unit step constraint softly
        for i in range(1, n):
            diff = x[i] - x[i-1]
            dist = np.sqrt(np.sum(diff ** 2)) + 1e-6
            # Project back to unit length
            if dist > 0:
                x[i] = x[i-1] + diff / dist
    
    # Snap to lattice and repair
    # Round to nearest integer
    x_snapped = np.round(x).astype(int)
    
    # Convert to list of tuples
    chain = [(int(x_snapped[i][0]), int(x_snapped[i][1])) for i in range(n)]
    
    # Check self-avoidance
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            # Collision: try to repair
            # Use a greedy repair
            for _ in range(20):
                # Try to move residue i to a nearby free site
                x, y = p
                options = []
                for dx, dy in DIRS:
                    np = (x + dx, y + dy)
                    if np not in seen:
                        # Check if adjacent to i-1
                        if i > 0:
                            prev = chain[i-1]
                            if np in [(prev[0]+1, prev[1]), (prev[0]-1, prev[1]), (prev[0], prev[1]+1), (prev[0], prev[1]-1)]:
                                options.append(np)
                        else:
                            options.append(np)
                if options:
                    new_p = rng.choice(options)
                    chain[i] = new_p
                    break
            else:
                # Failed to repair
                return None
        seen.add(p)
    
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence using a hybrid approach:
    1. Continuous optimization to find a good starting point.
    2. Pivot-based simulated annealing to refine.
    3. Genetic algorithm with error repair for global search.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)

    n = len(sequence)
    
    # Adaptive time allocation
    if n < 30:
        cont_time = 20.0
        sa_time = 80.0
        ga_time = 50.0
    else:
        cont_time = 30.0
        sa_time = 70.0
        ga_time = 40.0
        
    # Phase 1: Continuous optimization
    phase1_end = time.time() + cont_time
    
    while time.time() < phase1_end:
        chain = _continuous_fold(sequence, n, rng)
        if chain is not None:
            # Refine with Pivot SA
            sa_chain, sa_energy = _local_improve_pivot(
                sequence, chain, rng, max_iters=1000, start_temp=3.0, cool_rate=0.995)
                
            if best_chain is None or sa_energy < best_energy:
                best_chain = sa_chain
                best_energy = sa_energy
    
    # Phase 2: Random SAW + Pivot SA
    phase2_end = time.time() + sa_time
    
    attempts = 0
    while time.time() < phase2_end and attempts < 100:
        attempts += 1
        chain = _generate_random_saw(n, rng)
        if chain is None:
            continue
            
        # Refine with Pivot SA
        sa_chain, sa_energy = _local_improve_pivot(
            sequence, chain, rng, max_iters=1000, start_temp=3.0, cool_rate=0.995)
            
        if best_chain is None or sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
            
    # Fallback if no SAW generated
    if best_chain is None:
        def _greedy_grow_fallback(sequence, rng):
            start = (0, 0)
            chain = [start]
            occupied = {start: 0}
            n = len(sequence)
            
            for i in range(1, n):
                last = chain[-1]
                options = []
                for site in _neighbours(last):
                    if site in occupied:
                        continue
                    gain = 0
                    if sequence[i] == 'H':
                        for q in _neighbours(site):
                            j = occupied.get(q)
                            if j is not None and j < i - 1 and sequence[j] == 'H':
                                gain += 1
                    free = sum(1 for q in _neighbours(site) if q not in occupied)
                    score = gain * 10.0 + free * 0.1 + rng.random() * 1.0
                    options.append((score, site))
                    
                if not options:
                    return None
                    
                best_score, best_site = max(options, key=lambda x: x[0])
                occupied[best_site] = i
                chain.append(best_site)
                
            return chain

        chain = _greedy_grow_fallback(sequence, rng)
        if chain:
            best_chain = chain
            best_energy = _compute_energy(sequence, chain)
            # Refine it
            sa_chain, sa_energy = _local_improve_pivot(
                sequence, chain, rng, max_iters=2000, start_temp=3.0, cool_rate=0.995)
            if sa_energy < best_energy:
                best_chain = sa_chain
                best_energy = sa_energy

    # Phase 3: Genetic Algorithm with Error Repair
    if best_chain is not None:
        phase3_end = time.time() + ga_time
        population = [best_chain]
        
        # Add some diversity
        for _ in range(5):
            c = _generate_random_saw(n, rng)
            if c:
                population.append(c)
                
        for gen in range(30):
            if time.time() > phase3_end:
                break
                
            new_pop = []
            
            # Elitism
            new_pop.append(best_chain)
            
            # Crossover and Mutation with Error Repair
            while len(new_pop) < 10:
                p1 = rng.choice(population)
                p2 = rng.choice(population)
                child = _ga_crossover(sequence, p1, p2, rng)
                
                if child is None:
                    # Error Repair: Try to repair a random child from population
                    # Pick a random parent and mutate it heavily, then repair
                    parent = rng.choice(population)
                    steps = _chain_to_steps(parent)
                    for i in range(len(steps)):
                        if rng.random() < 0.3:
                            steps[i] = rng.randint(0, 3)
                    child = _steps_to_chain(steps)
                    
                    # Check if valid, if not, try _repair_chain
                    seen = set()
                    valid = True
                    for p in child:
                        if p in seen:
                            valid = False
                            break
                        seen.add(p)
                            
                    if not valid:
                        repaired = _repair_chain(sequence, child, rng)
                        if repaired:
                            # Verify repair
                            seen = set()
                            for p in repaired:
                                if p in seen:
                                    valid = False
                                    break
                                seen.add(p)
                            if valid:
                                child = repaired
                            else:
                                child = None
                        else:
                            child = None
                    else:
                        child = child # Already valid
                        
                if child is not None:
                    new_pop.append(child)
                    
            # Evaluate and Select
            scored = []
            for c in new_pop:
                e = _compute_energy(sequence, c)
                scored.append((e, c))
                
            scored.sort(key=lambda x: x[0])
            
            # Keep top 10
            new_pop = [c for e, c in scored[:10]]
            
            # Update best
            if scored[0][0] < best_energy:
                best_energy = scored[0][0]
                best_chain = scored[0][1]
                
            population = new_pop
            
    # Final Local Improvement
    if best_chain is not None:
        improved_chain, improved_energy = _local_improve_pivot(
            sequence, best_chain, rng, max_iters=2000, start_temp=1.0, cool_rate=0.999)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
