"""
HP lattice protein folding using Monte Carlo with simulated annealing.
This uses a fundamentally different strategy: instead of greedy growth + local search,
it uses random initialization and simulated annealing with more sophisticated moves.
"""
import random
import time
import math
from collections import defaultdict

# Precompute neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def _neighbors(point):
    """Return list of 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z),
            (x, y, z+1), (x, y, z-1)]


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-consecutive residues)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in neighbors(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_move(sequence, chain, pos_map, i, new_pos):
    """Check if moving residue i to new_pos is valid (self-avoiding, connected)."""
    n = len(chain)
    if i == 0 or i == n - 1:
        return False
    
    old_pos = chain[i]
    
    # new_pos must be free
    if new_pos in pos_map:
        return False
    
    # new_pos must be adjacent to both i-1 and i+1
    prev_pos = chain[i-1]
    next_pos = chain[i+1]
    
    dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) + abs(new_pos[2] - prev_pos[2])
    dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) + abs(new_pos[2] - next_pos[2])
    
    if dist_prev != 1 or dist_next != 1:
        return False
    
    # Check that removing old_pos doesn't break connectivity
    # old_pos was adjacent to i-1 and i+1, so we need to verify the chain stays connected
    # This is implicitly checked by the adjacency requirements above
    
    return True


def _make_move(sequence, chain, pos_map, i, new_pos):
    """Execute a move: relocate residue i to new_pos."""
    old_pos = chain[i]
    del pos_map[old_pos]
    chain[i] = new_pos
    pos_map[new_pos] = i


def _revert_move(sequence, chain, pos_map, i, old_pos):
    """Revert a move: put residue i back to old_pos."""
    new_pos = chain[i]
    del pos_map[new_pos]
    chain[i] = old_pos
    pos_map[old_pos] = i


def _delta_energy(sequence, chain, pos_map, i, new_pos):
    """Compute energy change for moving residue i to new_pos without actually moving."""
    old_pos = chain[i]
    delta = 0
    
    # Remove old contacts involving residue i
    if sequence[i] == 'H':
        for q in neighbors(old_pos):
            j = pos_map.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                delta += 1  # losing a contact
            elif j is not None and j < i - 1 and sequence[j] == 'H':
                delta += 1  # losing a contact
    
    if sequence[i] == 'P':
        # P residues don't contribute to contacts directly, but check if they were blocking
        # Actually, only H-H contacts count, so P moves only affect via position changes
        pass
    
    # Add new contacts involving residue i at new_pos
    if sequence[i] == 'H':
        for q in neighbors(new_pos):
            j = pos_map.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                delta -= 1  # gaining a contact
            elif j is not None and j < i - 1 and sequence[j] == 'H':
                delta -= 1  # gaining a contact
    
    return delta


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk as initial configuration."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    pos_map = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for offset in _NEIGHBOR_OFFSETS:
            new_pos = (last[0] + offset[0], last[1] + offset[1], last[2] + offset[2])
            if new_pos not in pos_map:
                options.append(new_pos)
        
        if not options:
            # Stuck - restart
            return None
        
        new_pos = rng.choice(options)
        chain.append(new_pos)
        pos_map[new_pos] = i
    
    return chain


def _simulated_annealing(sequence, chain, pos_map, steps, rng, start_temp=10.0, end_temp=0.1):
    """Perform simulated annealing on the chain."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    best_pos_map = dict(pos_map)
    
    for step in range(steps):
        # Linear cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (step / max(steps - 1, 1))
        
        # Choose a random interior residue to move
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Get valid new positions
        candidates = []
        for new_pos in neighbors(old_pos):
            if new_pos not in pos_map:
                # Check if move is valid (adjacent to both neighbors)
                prev_pos = chain[i-1]
                next_pos = chain[i+1]
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) + abs(new_pos[2] - prev_pos[2])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) + abs(new_pos[2] - next_pos[2])
                if dist_prev == 1 and dist_next == 1:
                    candidates.append(new_pos)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        de = _delta_energy(sequence, chain, pos_map, i, new_pos)
        
        # Metropolis criterion
        if de < 0 or rng.random() < math.exp(-de / max(temp, 1e-10)):
            # Accept the move
            _make_move(sequence, chain, pos_map, i, new_pos)
            current_energy += de
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                best_pos_map = dict(pos_map)
    
    return best_chain, best_energy, best_pos_map


def fold(sequence, dimension):
    """Fold the sequence using random initialization + simulated annealing."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is worst
    
    rng = random.Random(42 + n)
    
    # Number of restarts based on time budget
    num_restarts = min(50, max(10, int(120 * n / 50)))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate random initial walk
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            continue
        
        pos_map = {p: i for i, p in enumerate(chain)}
        energy = _compute_energy(sequence, chain)
        
        # Simulated annealing
        steps = min(20000, max(5000, int(150 * n)))
        chain, energy, pos_map = _simulated_annealing(
            sequence, chain, pos_map, steps, rng,
            start_temp=5.0, end_temp=0.05
        )
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check if we've found a good solution
        if best_energy <= -n // 3:
            # Good enough, try a few more restarts if time permits
            if time.time() > deadline - 60:
                break
    
    # If we have time, do more SA on the best chain
    if best_chain is not None and time.time() < deadline - 10:
        pos_map = {p: i for i, p in enumerate(best_chain)}
        steps = min(30000, max(10000, int(200 * n)))
        chain, energy, pos_map = _simulated_annealing(
            sequence, best_chain, pos_map, steps, rng,
            start_temp=2.0, end_temp=0.01
        )
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Fallback: straight line if no valid chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0
    
    return [list(p) for p in best_chain]
