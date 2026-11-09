"""
HP lattice protein folding on the cubic lattice.
Strategy: Multi-start with greedy growth seeds + simulated annealing with pivot moves.
Time-budgeted: uses remaining wall-clock to allocate SA steps per sequence.
"""
import random
import time
import math
import sys

# Global deadline: evaluator kills at 330s wall-clock; use 230s for safety
_TOTAL_DEADLINE = 230.0
_start_time = time.time()


def _remaining_time():
    return max(0.0, _TOTAL_DEADLINE - (time.time() - _start_time))


def _neighbours(point):
    """Yield all 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute HP energy: -number of non-adjacent H-H contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            for q in _neighbours(chain[i]):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Greedy/randomized chain growth.
    temperature: 0 = pure greedy, >0 = add randomness.
    Returns a list of tuples (lattice points) or None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Score: prefer sites with H-H contact potential, then free space
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            # Count free neighbours for entropy
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            score = gain * 2.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best (or random among top if temperature > 0)
        options.sort(key=lambda x: -x[0])
        if temperature > 0 and len(options) > 1:
            # Boltzmann-like selection from top few
            k = min(len(options), max(2, int(temperature * 3)))
            top = options[:k]
            weights = [math.exp(s / max(1e-6, temperature)) for s, _ in top]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = top[-1][1]
            for w, (_, site) in zip(weights, top):
                cum += w
                if r <= cum:
                    chosen = site
                    break
        else:
            chosen = options[0][1]
        
        occupied[chosen] = i
        chain.append(chosen)
    
    return chain


def _pivot_move(chain, rng, dimension):
    """
    Attempt a pivot move: rotate a sub-chain by 90 degrees around an axis.
    Returns (new_chain, delta_energy_info) or (None, 0) if invalid.
    We only check self-avoidance here; energy delta is computed externally.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Pick pivot point
    pivot_idx = rng.randint(1, n - 2)
    
    # Pick axis of rotation
    axis = rng.randint(0, dimension - 1)
    
    # Pick direction (+1 or -1)
    direction = rng.choice([-1, 1])
    
    # Build rotation: for each point in sub-chain, rotate around pivot
    # We need to rotate the vector (point - pivot) around the axis
    sub_chain = list(chain[pivot_idx:])
    pivot = chain[pivot_idx]
    
    # Rotation matrix for 90-degree rotation around axis
    # In 3D, rotating (x,y,z) 90 deg around z: (x,y,z) -> (-y,x,z)
    # Around x: (x,y,z) -> (x,-z,y)
    # Around y: (x,y,z) -> (z,x,-y)
    
    new_sub = []
    valid = True
    for pt in sub_chain:
        vec = [pt[k] - pivot[k] for k in range(dimension)]
        if axis == 0:  # x-axis
            new_vec = [vec[0], -direction * vec[2], direction * vec[1]]
        elif axis == 1:  # y-axis
            new_vec = [direction * vec[2], vec[1], -direction * vec[0]]
        else:  # z-axis
            new_vec = [-direction * vec[1], direction * vec[0], vec[2]]
        new_pt = tuple(pivot[k] + new_vec[k] for k in range(dimension))
        new_sub.append(new_pt)
    
    # Check self-avoidance: new sub-chain points must not collide with fixed part or each other
    fixed_part = set(chain[:pivot_idx + 1])  # include pivot
    # Remove pivot from fixed part for collision check with new sub
    # Actually, the first point of new_sub is the pivot itself
    # Check new_sub[1:] against fixed_part and among themselves
    seen = set(fixed_part)
    for i, pt in enumerate(new_sub):
        if i == 0:
            continue  # pivot stays
        if pt in seen:
            valid = False
            break
        seen.add(pt)
    
    if not valid:
        return None
    
    new_chain = chain[:pivot_idx] + new_sub
    return new_chain


def _slide_move(chain, rng, dimension):
    """
    Attempt a slide move: move one internal residue to an adjacent empty site.
    Returns new_chain or None.
    """
    n = len(chain)
    if n < 3:
        return None
    
    idx = rng.randint(1, n - 2)
    pt = chain[idx]
    
    for new_pt in _neighbours(pt):
        if new_pt == chain[idx - 1] or new_pt == chain[idx + 1]:
            continue
        # Check if new_pt is occupied by any other part of chain
        # Build occupied set excluding idx
        occupied = set(chain)
        occupied.discard(pt)
        if new_pt in occupied:
            continue
        # Also check that moving doesn't cause adjacency issues
        # The new position must not be adjacent to chain[idx-1] or chain[idx+1] 
        # in a way that creates a 180-degree turn (which is fine, just no overlap)
        # Actually, we just need self-avoidance
        new_chain = list(chain)
        new_chain[idx] = new_pt
        return new_chain
    
    return None


def _sa_fold(sequence, dimension, rng, time_budget):
    """
    Simulated annealing on a single initial conformation.
    Returns (best_chain, best_energy).
    """
    n = len(sequence)
    
    # Generate initial conformation
    chain = None
    for temp in [0.0, 0.5, 1.0, 2.0]:
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is not None:
            break
    
    if chain is None:
        # Fallback: straight line
        chain = [(i, 0, 0) for i in range(n)]
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing
    # t0 scaled to initial energy magnitude
    t0 = max(1.0, abs(current_energy) * 0.5)
    # Cooling: geometric, rate chosen so that t_end ~ 0.01
    # We'll estimate number of steps from time budget
    # Each step is roughly O(n) for energy computation, but we optimize
    
    # Build position map for fast energy delta computation
    def build_map(chain):
        m = {}
        for i, p in enumerate(chain):
            m[p] = i
        return m
    
    cur_map = build_map(chain)
    
    # Precompute H positions
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    
    # Estimate steps: assume ~1000 steps per second (conservative)
    elapsed = time.time() - _start_time
    time_left = max(0.1, time_budget - (elapsed - (_start_time if _start_time > 0 else time.time())))
    # Actually, let's just use a deadline check inside the loop
    deadline = time.time() + time_budget
    
    # Number of restarts within this SA call is 1; outer loop handles restarts
    # Set cooling rate based on estimated iterations
    # We'll do a fixed number of iterations based on time
    # Let's target ~50000 iterations per second for simple moves
    target_rate = 0.9995  # slow cooling
    
    t = t0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Choose move type
        if rng.random() < 0.7:
            new_chain = _pivot_move(chain, rng, dimension)
        else:
            new_chain = _slide_move(chain, rng, dimension)
        
        if new_chain is None:
            t *= target_rate
            if t < 0.01:
                break
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0:
            # Accept
            chain = new_chain
            current_energy = new_energy
            cur_map = build_map(chain)
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Metropolis acceptance
            if t > 0:
                prob = math.exp(-delta / t)
                if rng.random() < prob:
                    chain = new_chain
                    current_energy = new_energy
                    cur_map = build_map(chain)
        
        t *= target_rate
        if t < 0.01:
            break
        
        if no_improve > max_no_improve:
            # Restart with new seed
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using multi-start SA with time budgeting.
    Returns list of lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    
    # Allocate time budget for this sequence
    # We don't know how many sequences are left, so use a heuristic:
    # Give each sequence a share based on remaining time
    # For simplicity, use a fixed fraction of remaining time
    time_left = _remaining_time()
    
    # Reserve 5s for final cleanup
    usable_time = max(1.0, time_left - 5.0)
    
    # Number of restarts: more for shorter sequences
    # Each restart takes roughly proportional to n
    # Estimate: ~1000 SA iterations per second per restart
    # Let's do at least 3 restarts, more if time allows
    min_restarts = 3
    # Estimate time per restart: ~n * 0.001 seconds (very rough)
    est_time_per_restart = max(0.5, n * 0.002)
    max_restarts = min(20, max(min_restarts, int(usable_time / est_time_per_restart)))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is "worst" (unfolded)
    
    for restart in range(max_restarts):
        if _remaining_time() < 5.0:
            break
        
        # Time budget for this restart
        remaining = _remaining_time() - 5.0
        if remaining <= 0:
            break
        
        # Distribute remaining time among remaining restarts
        restarts_left = max_restarts - restart
        time_for_this = remaining / restarts_left
        
        chain, energy = _sa_fold(sequence, dimension, rng, time_for_this)
        
        if chain is not None:
            # Verify self-avoidance
            if len(set(chain)) == len(chain):
                # Verify unit steps
                valid = True
                for i in range(1, len(chain)):
                    diff = sum(abs(chain[i][k] - chain[i-1][k]) for k in range(dimension))
                    if diff != 1:
                        valid = False
                        break
                if valid and energy < best_energy:
                    best_energy = energy
                    best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
