"""HP lattice protein folding using contact-guided backtracking with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    """Yield the 4 lattice neighbours of a 2D point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(sequence, chain):
    """Compute the HP energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    where = {tuple(p): i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            for q in _neighbours(tuple(chain[i])):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2  # Each contact counted twice


def _try_local_move(sequence, chain, rng, max_moves=100):
    """Try local single-residue moves to improve energy."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_moves):
        i = rng.randint(1, n - 2)  # Don't move endpoints
        old_pos = tuple(chain[i])
        for new_pos in _neighbours(old_pos):
            if new_pos == tuple(chain[i-1]) or new_pos == tuple(chain[i+1]):
                continue
            # Check self-avoidance
            if any(tuple(p) == new_pos for j, p in enumerate(chain) if j != i):
                continue
            # Try the move
            chain[i] = list(new_pos)
            new_energy = _compute_energy(sequence, chain)
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
            else:
                chain[i] = list(old_pos)
        chain[i] = list(old_pos)
    
    return best_chain, best_energy


def _backtrack_fold(sequence, dimension, rng, deadline):
    """Contact-guided backtracking to find a valid conformation."""
    n = len(sequence)
    if n == 0:
        return None
    if n == 1:
        return [(0, 0)]
    
    # Start with first two residues
    chain = [(0, 0)]
    # Choose initial direction randomly
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rng.shuffle(dirs)
    first_dir = dirs[0]
    chain.append((first_dir[0], first_dir[1]))
    occupied = {tuple(p): i for i, p in enumerate(chain)}
    
    def backtrack(pos, chain, occupied):
        if time.time() > deadline:
            return None
        if pos == n:
            return list(chain)
        
        # Get candidate positions
        candidates = []
        last = chain[-1]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos in occupied:
                continue
            # Don't go back to previous position
            if pos >= 2 and new_pos == chain[-2]:
                continue
            candidates.append(new_pos)
        
        if not candidates:
            return None
        
        # Score candidates by potential H-H contacts
        def score_candidate(pos_idx, cand):
            score = 0
            if sequence[pos_idx] == 'H':
                for q in _neighbours(cand):
                    if q in occupied:
                        j = occupied[q]
                        if j < pos_idx - 1 and sequence[j] == 'H':
                            score += 1
            # Add small random tiebreaker
            score += rng.random() * 0.01
            return score
        
        # Sort by score descending
        candidates.sort(key=lambda c: -score_candidate(pos, c))
        
        # Try candidates in order
        for cand in candidates:
            chain.append(cand)
            occupied[cand] = pos
            result = backtrack(pos + 1, chain, occupied)
            if result is not None:
                return result
            chain.pop()
            del occupied[cand]
        
        return None
    
    return backtrack(2, chain, occupied)


def fold(sequence, dimension):
    """Find the best conformation within the time budget."""
    start_time = time.time()
    # Leave safety margin: aim to finish by 180s, hard kill at 210s
    deadline = start_time + 170  # Safety margin
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    
    best_chain = None
    best_energy = 0  # We want to minimize (more negative is better)
    
    # First, get a baseline conformation
    rng = random.Random(42)
    baseline = _backtrack_fold(sequence, dimension, rng, deadline)
    if baseline is None:
        # Fallback: straight line
        baseline = [(i, 0) for i in range(n)]
    
    best_chain = baseline
    best_energy = _compute_energy(sequence, best_chain)
    
    # Try multiple backtracking attempts with different seeds
    attempts = 0
    while time.time() < deadline - 5:  # Leave 5s for final refinement
        seed = 12345 + attempts
        rng = random.Random(seed)
        chain = _backtrack_fold(sequence, dimension, rng, deadline)
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
        attempts += 1
        if attempts > 50:  # Cap on number of backtracking attempts
            break
    
    # Local search refinement on the best chain found
    if time.time() < deadline - 2:
        rng = random.Random(99999)
        best_chain, best_energy = _try_local_move(sequence, best_chain, rng, max_moves=200)
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
