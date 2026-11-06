"""HP lattice protein folding on the square lattice with simulated annealing."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

_DEADLINE = 200.0  # safety margin before 210s kill
_START_TIME = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    """Yield the 4 orthogonal neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _is_valid(chain, sequence):
    """Check self-avoidance and unit steps."""
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(p[0] - chain[i-1][0])
            dy = abs(p[1] - chain[i-1][1])
            if dx + dy != 1:
                return False
    return True


def _compute_energy(chain, sequence):
    """Compute the HP contact energy (negative number of H-H contacts)."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _init_chain(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites that create H-H contacts
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 10
            # Slight preference for free space to avoid dead ends
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free
            candidates.append((score + rng.random(), site))
        
        if not candidates:
            return None
        
        candidates.sort(reverse=True)
        # Pick from top candidates with some randomness
        top_k = min(3, len(candidates))
        _, site = candidates[rng.randint(0, top_k - 1)]
        chain.append(site)
        occupied[site] = i
    
    return chain


def _sa_fold(sequence, dimension, rng, time_budget):
    """Simulated annealing to minimize HP energy."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] * n
    
    # Initialize
    chain = None
    for attempt in range(50):
        chain = _init_chain(sequence, dimension, rng)
        if chain is not None:
            break
    
    if chain is None:
        # Fallback: straight line
        return [(i, 0) for i in range(n)]
    
    current_energy = _compute_energy(chain, sequence)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T_start = 2.0
    T_end = 0.01
    alpha = (T_end / T_start) ** (1.0 / max(1, int(time_budget * 500)))
    
    T = T_start
    steps = 0
    max_steps = int(time_budget * 1000)
    
    while steps < max_steps and _remaining() > 2.0:
        steps += 1
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Pivot move: rotate a subchain
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Try rotating the subchain chain[i:j+1] by 90 or 180 degrees
            subchain = chain[i:j+1]
            pivot = chain[i]
            
            # Generate rotated versions
            rotations = []
            # 90 degree clockwise
            rot1 = [(pivot[0] + (p[1] - pivot[1]), pivot[1] - (p[0] - pivot[0])) for p in subchain]
            # 90 degree counter-clockwise
            rot2 = [(pivot[0] - (p[1] - pivot[1]), pivot[1] + (p[0] - pivot[0])) for p in subchain]
            # 180 degree
            rot3 = [(2 * pivot[0] - p[0], 2 * pivot[1] - p[1]) for p in subchain]
            
            for new_sub in [rot1, rot2, rot3]:
                # Check validity
                new_chain = chain[:i] + new_sub + chain[j+1:]
                # Check self-avoidance for the changed part
                valid = True
                occupied_set = set(new_chain[:i])
                for k in range(i, len(new_chain)):
                    p = new_chain[k]
                    if p in occupied_set:
                        valid = False
                        break
                    occupied_set.add(p)
                if not valid:
                    continue
                # Check unit steps at boundaries
                if i > 0:
                    dx = abs(new_chain[i][0] - new_chain[i-1][0])
                    dy = abs(new_chain[i][1] - new_chain[i-1][1])
                    if dx + dy != 1:
                        continue
                if j + 1 < n:
                    dx = abs(new_chain[j+1][0] - new_chain[j][0])
                    dy = abs(new_chain[j+1][1] - new_chain[j][1])
                    if dx + dy != 1:
                        continue
                
                new_energy = _compute_energy(new_chain, sequence)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    break
            else:
                continue
        
        elif move_type < 0.6:
            # Slide move: move a single residue
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            
            candidates = []
            for site in _neighbours(old_pos):
                if site == chain[i-1] or site == chain[i+1]:
                    continue
                # Check if site is occupied by other residues
                occupied = set(chain[:i]) | set(chain[i+1:])
                if site in occupied:
                    continue
                # Check unit steps
                dx1 = abs(site[0] - chain[i-1][0])
                dy1 = abs(site[1] - chain[i-1][1])
                dx2 = abs(site[0] - chain[i+1][0])
                dy2 = abs(site[1] - chain[i+1][1])
                if dx1 + dy1 != 1 or dx2 + dy2 != 1:
                    continue
                candidates.append(site)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            new_chain = list(chain)
            new_chain[i] = new_pos
            
            new_energy = _compute_energy(new_chain, sequence)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        else:
            # Reversal move: reverse a subchain
            if n < 4:
                continue
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            
            # Reverse chain[i:j+1]
            new_sub = chain[i:j+1][::-1]
            new_chain = chain[:i] + new_sub + chain[j+1:]
            
            # Check validity
            valid = True
            occupied_set = set(new_chain[:i])
            for k in range(i, len(new_chain)):
                p = new_chain[k]
                if p in occupied_set:
                    valid = False
                    break
                occupied_set.add(p)
            if not valid:
                continue
            # Check unit steps at boundaries
            if i > 0:
                dx = abs(new_chain[i][0] - new_chain[i-1][0])
                dy = abs(new_chain[i][1] - new_chain[i-1][1])
                if dx + dy != 1:
                    continue
            if j + 1 < n:
                dx = abs(new_chain[j+1][0] - new_chain[j][0])
                dy = abs(new_chain[j+1][1] - new_chain[j][1])
                if dx + dy != 1:
                    continue
            
            new_energy = _compute_energy(new_chain, sequence)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        # Cool down
        T *= alpha
    
    return best_chain


def fold(sequence, dimension):
    """Fold an HP sequence into a low-energy conformation on the square lattice."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    if dimension != 2:
        # Fallback for non-2D: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Allocate time budget
    time_budget = min(15.0, _remaining() * 0.8)
    if time_budget < 1.0:
        # Very little time left, just return a simple conformation
        chain = _init_chain(sequence, dimension, random.Random(42))
        if chain is None:
            return [[i, 0] for i in range(n)]
        return [list(p) for p in chain]
    
    rng = random.Random(12345 + len(sequence))
    
    # Multiple restarts with simulated annealing
    best_chain = None
    best_energy = float('inf')
    
    num_restarts = max(1, int(time_budget / 5.0))
    per_restart = time_budget / num_restarts
    
    for r in range(num_restarts):
        if _remaining() < 3.0:
            break
        
        chain = _sa_fold(sequence, dimension, rng, per_restart)
        if chain is None:
            continue
        
        energy = _compute_energy(chain, sequence)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback
        best_chain = _init_chain(sequence, dimension, rng)
        if best_chain is None:
            return [[i, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
