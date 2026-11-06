"""
HP lattice protein folding on the cubic lattice.
Strategy: Monte Carlo growth with backtracking and local optimization.
"""
import random
import sys
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _precompute_adjacent(seq):
    """Precompute which residues are H for fast lookup."""
    return [c == 'H' for c in seq]


def _energy_from_contacts(seq, chain, contacts_set):
    """Compute energy given a set of contact pairs (i, j) where i < j."""
    count = 0
    for i, j in contacts_set:
        if seq[i] == 'H' and seq[j] == 'H':
            count += 1
    return -count


def _compute_contacts(seq, chain):
    """Compute all non-consecutive contacts in the chain."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = set()
    n = len(chain)
    for i in range(n):
        if seq[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and abs(i - j) > 1:
                pair = (min(i, j), max(i, j))
                if seq[j] == 'H':
                    contacts.add(pair)
    return contacts


def _energy(seq, chain):
    """Compute energy of a chain."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if seq[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                if i < j:
                    contacts += 1
    return -contacts


def _local_optimize(seq, chain, max_moves=5000, rng=None):
    """
    Perform local optimization by moving individual residues or small segments.
    Uses a simple Monte Carlo approach: try moving a residue to a neighboring site,
    accept if energy improves.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 4:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(seq, current_chain)
    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    for _ in range(max_moves):
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        p = current_chain[i]
        
        # Try each neighbor
        for q in _neighbours(p):
            if q in pos_to_idx:
                continue
            # Check if moving i to q creates valid chain (no self-intersection)
            # The new position q must not be occupied by any other residue
            # Since we're checking pos_to_idx, and q is not in it, we're good
            # But we need to make sure the chain remains connected (it does, since we're 
            # just moving one point, and its neighbors in the chain are still adjacent 
            # to it? No! Moving i to q means i-1 and i+1 must still be adjacent to q.
            # Actually, the chain connectivity is: ... i-1, i, i+1 ...
            # If we move i to q, then i-1 must be adjacent to q and i+1 must be adjacent to q.
            
            prev_p = current_chain[i - 1]
            next_p = current_chain[i + 1]
            
            # Check adjacency
            if abs(prev_p[0] - q[0]) + abs(prev_p[1] - q[1]) + abs(prev_p[2] - q[2]) != 1:
                continue
            if abs(next_p[0] - q[0]) + abs(next_p[1] - q[1]) + abs(next_p[2] - q[2]) != 1:
                continue
            
            # Move the residue
            pos_to_idx.pop(p)
            pos_to_idx[q] = i
            current_chain[i] = q
            
            new_energy = _energy(seq, current_chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
            else:
                # Revert
                pos_to_idx.pop(q)
                pos_to_idx[p] = i
                current_chain[i] = p
    
    return best_chain


def _grow_with_backtracking(seq, dimension, rng, max_depth=None, time_limit=None):
    """
    Grow the chain using a backtracking search with heuristic ordering.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    if max_depth is None:
        max_depth = len(seq)
    
    start_time = time.time() if time_limit else None
    
    def _heuristic_score(i, site):
        """Score a candidate site for position i."""
        if i >= len(seq):
            return 0
        score = 0.0
        if seq[i] == 'H':
            for q in _neighbours(site):
                j = occupied.get(q)
                if j is not None and j < i - 1 and seq[j] == 'H':
                    score += 10
        # Prefer sites with more free space to avoid trapping
        free = sum(1 for q in _neighbours(site) if q not in occupied)
        score += 0.1 * free
        score += rng.random() * 0.5
        return score
    
    def _dfs(depth):
        if depth == len(seq):
            return True
        if time_limit and time.time() - start_time > time_limit:
            return False
        
        current_site = chain[-1]
        candidates = []
        for site in _neighbours(current_site):
            if site in occupied:
                continue
            score = _heuristic_score(depth, site)
            candidates.append((score, site))
        
        candidates.sort(key=lambda x: -x[0])
        
        for score, site in candidates:
            chain.append(site)
            occupied[site] = depth
            
            if _dfs(depth + 1):
                return True
            
            chain.pop()
            del occupied[site]
        
        return False
    
    if _dfs(1):
        return chain
    
    return None


def _simulated_annealing_fold(seq, dimension, iterations=10000, rng=None):
    """
    Simulated annealing on the chain representation.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(seq)
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Start with a random walk or straight line
    # Use a random walk as initial state
    chain = [tuple([0] * dimension)]
    occupied = {chain[0]: 0}
    last_dir = None
    
    for i in range(1, n):
        valid_dirs = []
        for axis in range(dimension):
            for step in (-1, 1):
                if last_dir is not None and last_dir[0] == axis and last_dir[1] == -step:
                    continue
                q = list(chain[-1])
                q[axis] += step
                q = tuple(q)
                if q not in occupied:
                    valid_dirs.append((axis, step))
        
        if not valid_dirs:
            # Try any direction
            for axis in range(dimension):
                for step in (-1, 1):
                    q = list(chain[-1])
                    q[axis] += step
                    q = tuple(q)
                    if q not in occupied:
                        valid_dirs.append((axis, step))
        
        if not valid_dirs:
            # Cannot continue, return what we have
            return None
        
        axis, step = rng.choice(valid_dirs)
        last_dir = (axis, step)
        q = list(chain[-1])
        q[axis] += step
        q = tuple(q)
        chain.append(q)
        occupied[q] = i
    
    if len(chain) < n:
        return None
    
    current_energy = _energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0
    cooling = 0.999
    
    for it in range(iterations):
        T *= cooling
        if T < 0.01:
            break
        
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        p = chain[i]
        
        # Try a random neighbor
        axis = rng.randint(0, dimension - 1)
        step = rng.choice((-1, 1))
        q = list(p)
        q[axis] += step
        q = tuple(q)
        
        if q in occupied:
            continue
        
        prev_p = chain[i - 1]
        next_p = chain[i + 1]
        
        # Check adjacency
        if abs(prev_p[0] - q[0]) + abs(prev_p[1] - q[1]) + abs(prev_p[2] - q[2]) != 1:
            continue
        if abs(next_p[0] - q[0]) + abs(next_p[1] - q[1]) + abs(next_p[2] - q[2]) != 1:
            continue
        
        # Move
        occupied.pop(p)
        occupied[q] = i
        chain[i] = q
        
        new_energy = _energy(seq, chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            # Revert
            occupied.pop(q)
            occupied[p] = i
            chain[i] = p
    
    return best_chain


def _hybrid_fold(seq, dimension, time_budget=10.0):
    """
    Hybrid approach: multiple growth strategies + local optimization.
    """
    rng = random.Random(42 + len(seq))
    n = len(seq)
    
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    best_chain = None
    best_energy = float('inf')
    
    start_time = time.time()
    
    # Strategy 1: Multiple greedy growths with different seeds
    for trial in range(50):
        if time.time() - start_time > time_budget * 0.4:
            break
        
        seed = 1000 + trial
        rng_trial = random.Random(seed)
        
        # Greedy growth with randomization
        chain = [tuple([0] * dimension)]
        occupied = {chain[0]: 0}
        success = True
        
        for i in range(1, n):
            candidates = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                score = 0.0
                if seq[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and seq[j] == 'H':
                            score += 5
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score += 0.05 * free
                score += rng_trial.random() * 2.0
                candidates.append((score, site))
            
            if not candidates:
                success = False
                break
            
            candidates.sort(key=lambda x: -x[0])
            # Sometimes pick a non-best option for diversity
            if rng_trial.random() < 0.3 and len(candidates) > 1:
                idx = rng_trial.randint(1, min(3, len(candidates) - 1))
            else:
                idx = 0
            
            site = candidates[idx][1]
            chain.append(site)
            occupied[site] = i
        
        if not success or len(chain) < n:
            continue
        
        energy = _energy(seq, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Local optimization
        if time.time() - start_time < time_budget * 0.8:
            opt_chain = _local_optimize(seq, chain, max_moves=2000, rng=rng_trial)
            opt_energy = _energy(seq, opt_chain)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_chain = opt_chain
    
    # Strategy 2: Simulated annealing
    if time.time() - start_time < time_budget * 0.9:
        sa_chain = _simulated_annealing_fold(seq, dimension, iterations=5000, rng=rng)
        if sa_chain is not None and len(sa_chain) == n:
            sa_energy = _energy(seq, sa_chain)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_chain = sa_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n < 4:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Determine time budget based on sequence length
    # Total budget is 240s for all sequences, so ~9s per sequence
    time_budget = max(3.0, min(15.0, 240.0 / 26.0 * 0.8))
    
    chain = _hybrid_fold(sequence, dimension, time_budget=time_budget)
    
    # Ensure the chain is valid
    if chain is None or len(chain) != n:
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
