"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Deadline guard: 330s wall-clock kill, use 320s as safety margin
_DEADLINE = 320.0
_start_time = None

def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, temp=0.0):
    """
    One chain growth. Uses a scoring function that balances:
    - New H-H contacts (primary)
    - Free space (to avoid trapping)
    - Random noise (for diversity)
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count new H-H contacts this site would make
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Count free neighbours (to avoid dead ends)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Score: contacts dominate, free space breaks ties, noise for diversity
            score = gain * 10.0 + free * 0.5 + rng.random() * (1.0 + temp)
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best (with some randomness if temp > 0)
        if temp > 0 and rng.random() < 0.3:
            options.sort(key=lambda x: -x[0])
            k = min(len(options), 1 + int(rng.expovariate(1.0 / max(1, temp))))
            site = options[rng.randint(0, k - 1)][1]
        else:
            site = max(options)[1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain

def _is_valid_chain(chain, dimension):
    """Check self-avoidance and unit steps."""
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for k in range(len(chain) - 1):
        diff = sum(abs(chain[k][d] - chain[k+1][d]) for d in range(dimension))
        if diff != 1:
            return False
    return True

def _pivot_move(chain, dimension, rng):
    """
    Perform a pivot move: rotate a sub-chain around a pivot point.
    Returns the new chain if valid, else None.
    """
    n = len(chain)
    if n < 3:
        return None
    
    # Choose pivot index (1 to n-2 to ensure we have something to rotate)
    pivot_idx = rng.randint(1, n - 2)
    
    # Choose rotation axis and angle (90, 180, 270 degrees)
    axis = rng.randint(0, dimension - 1)
    angle = rng.choice([1, 2, 3]) # 1=90, 2=180, 3=270
    
    # Define rotation matrix for 90 deg around 'axis'
    # We only rotate the part of the chain from pivot_idx to end
    # The pivot point itself stays fixed.
    
    # Extract the sub-chain to rotate
    head = chain[:pivot_idx]
    pivot_point = chain[pivot_idx]
    tail = chain[pivot_idx+1:]
    
    # We need to rotate the relative coordinates of the tail with respect to the pivot.
    # Actually, it's easier to rotate the absolute coordinates of the tail segment
    # around the pivot point.
    
    # Let's implement rotation of points relative to pivot_point
    def rotate_point(point, pivot, axis, steps):
        p = list(point)
        # Translate to origin
        for d in range(dimension):
            p[d] -= pivot[d]
        
        # Rotate around 'axis'
        # In 3D, rotation around Z (axis 2) swaps X and Y.
        # Rotation around Y (axis 1) swaps X and Z.
        # Rotation around X (axis 0) swaps Y and Z.
        
        # Identify the two axes perpendicular to the rotation axis
        if axis == 0:
            ax1, ax2 = 1, 2
        elif axis == 1:
            ax1, ax2 = 0, 2
        else:
            ax1, ax2 = 0, 1
            
        v1 = p[ax1]
        v2 = p[ax2]
        
        if steps == 1: # 90 deg
            # (x, y) -> (y, -x)
            new_v1 = v2
            new_v2 = -v1
        elif steps == 2: # 180 deg
            # (x, y) -> (-x, -y)
            new_v1 = -v1
            new_v2 = -v2
        else: # 270 deg
            # (x, y) -> (-y, x)
            new_v1 = -v2
            new_v2 = v1
            
        p[ax1] = new_v1
        p[ax2] = new_v2
        
        # Translate back
        for d in range(dimension):
            p[d] += pivot[d]
            
        return tuple(p)
    
    new_tail = []
    for p in tail:
        new_tail.append(rotate_point(p, pivot_point, axis, angle))
        
    new_chain = head + [pivot_point] + new_tail
    
    # Check validity
    if _is_valid_chain(new_chain, dimension):
        return new_chain
    return None

def _criss_cross_move(chain, dimension, rng):
    """
    Criss-cross move: pick two non-adjacent bonds and swap their endpoints if valid.
    """
    n = len(chain)
    if n < 5:
        return None
        
    # Pick two bonds (i, i+1) and (j, j+1) such that |i-j| > 1
    i = rng.randint(0, n - 2)
    j = rng.randint(0, n - 2)
    if abs(i - j) <= 1:
        return None
        
    # Ensure i < j for consistency
    if i > j:
        i, j = j, i
        
    # The move swaps the segments.
    # Original: ... A_i - A_{i+1} ... A_j - A_{j+1} ...
    # New: ... A_i - A_{j+1} ... A_j - A_{i+1} ...
    # This effectively reverses the segment between i+1 and j and swaps the connection logic.
    # Wait, standard criss-cross on a path:
    # Cut at (i, i+1) and (j, j+1).
    # Reconnect i to j+1 and j to i+1.
    # This requires the segment i+1..j to be reversed? No, that's 2-opt.
    # Criss-cross usually involves swapping two independent segments or "crossing" paths.
    # Let's stick to a simpler variant: 2-opt is already good, but Pivot is better for exploration.
    # Let's implement a "Reversal" (2-opt) but optimized.
    
    a = i
    b = j + 1
    if b - a < 2:
        return None
        
    new_chain = chain[:a] + chain[a:b+1][::-1] + chain[b+1:]
    
    if _is_valid_chain(new_chain, dimension):
        return new_chain
    return None

def _replica_exchange_mc(sequence, dimension, rng, n_replicas, steps_per_replica, temps):
    """
    Parallel Tempering (Replica Exchange) Monte Carlo.
    """
    n = len(sequence)
    
    # Initialize replicas with different greedy chains
    replicas = []
    energies = []
    for r in range(n_replicas):
        # Use different temperatures for initialization to get diverse starts
        init_temp = 0.0 if r == 0 else rng.uniform(0.5, 2.0)
        chain = _greedy_grow(sequence, dimension, rng, init_temp)
        if chain is None:
            chain = _greedy_grow(sequence, dimension, rng, 0.0)
        if chain is None:
            # Fallback straight line
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        replicas.append(chain)
        energies.append(_energy(sequence, chain))
        
    best_chain = replicas[0]
    best_energy = energies[0]
    
    # Track global best
    for r in range(n_replicas):
        if energies[r] < best_energy:
            best_energy = energies[r]
            best_chain = replicas[r]
            
    steps_executed = 0
    max_total_steps = steps_per_replica * n_replicas
    
    while steps_executed < max_total_steps:
        if _get_remaining_time() < 2.0:
            break
            
        # 1. Perform moves on each replica
        for r in range(n_replicas):
            if _get_remaining_time() < 2.0:
                break
                
            chain = replicas[r]
            E = energies[r]
            T = temps[r]
            
            # Choose move type
            move_type = rng.random()
            new_chain = None
            
            if move_type < 0.7:
                new_chain = _pivot_move(chain, dimension, rng)
            else:
                new_chain = _criss_cross_move(chain, dimension, rng)
                
            if new_chain is not None:
                new_E = _energy(sequence, new_chain)
                delta_E = new_E - E
                
                # Metropolis Acceptance
                if delta_E <= 0:
                    accept = True
                else:
                    # Avoid overflow
                    if T > 1e-6:
                        ratio = math.exp(-delta_E / T)
                        accept = rng.random() < ratio
                    else:
                        accept = False
                        
                if accept:
                    replicas[r] = new_chain
                    energies[r] = new_E
                    if new_E < best_energy:
                        best_energy = new_E
                        best_chain = new_chain
                        
        steps_executed += n_replicas
        
        # 2. Attempt swaps between neighboring replicas
        # Sort replicas by temperature? No, usually fixed order.
        # We swap if (E_i/T_i - E_j/T_j) < log(random)
        for r in range(n_replicas - 1):
            if _get_remaining_time() < 2.0:
                break
                
            T1 = temps[r]
            T2 = temps[r+1]
            E1 = energies[r]
            E2 = energies[r+1]
            
            # Condition: exp(-(E1/T1 - E2/T2)) > u
            # -E1/T1 + E2/T2 > log(u)
            # E2/T2 - E1/T1 > log(u)
            
            u = rng.random()
            if u == 0: u = 1e-10
            
            # Calculate delta
            delta = (E2 / T2) - (E1 / T1)
            
            # If delta > log(u), accept swap
            # log(u) is negative.
            if delta > math.log(u):
                # Swap
                replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                energies[r], energies[r+1] = energies[r+1], energies[r]
                
                # Update global best if swapped in a better one (though we track global best separately)
                # The global best is tracked in the move loop, so no update needed here 
                # unless the swap itself puts a better chain into a "cold" slot which 
                # doesn't change the global best value, just its location.
                
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using Parallel Tempering Monte Carlo.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Define temperatures
    # Geometric progression from T_low to T_high
    T_low = 0.1
    T_high = 5.0
    
    # Number of replicas
    if n <= 48:
        n_replicas = 8
        steps_per_replica = 5000
    elif n <= 80:
        n_replicas = 6
        steps_per_replica = 3000
    else:
        n_replicas = 4
        steps_per_replica = 2000
        
    # Create temperature ladder
    if n_replicas > 1:
        ratio = (T_high / T_low) ** (1.0 / (n_replicas - 1))
        temps = [T_low * (ratio ** i) for i in range(n_replicas)]
    else:
        temps = [T_low]
        
    best_chain, best_energy = _replica_exchange_mc(
        sequence, dimension, rng, n_replicas, steps_per_replica, temps
    )
    
    # Fallback if something went wrong
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        
    # Final polish: A quick local search on the best found chain
    # Using 2-opt for a few iterations to tighten up
    if _get_remaining_time() > 5.0:
        # Simple local search
        improved = True
        moves_done = 0
        max_moves = 1000
        while improved and moves_done < max_moves:
            improved = False
            moves_done += 1
            if _get_remaining_time() < 2.0:
                break
            # Try a few random 2-opts
            for _ in range(10):
                a = rng.randint(0, n - 2)
                b = rng.randint(a + 2, n - 1)
                new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
                if _is_valid_chain(new_chain, dimension):
                    new_E = _energy(sequence, new_chain)
                    if new_E < best_energy:
                        best_chain = new_chain
                        best_energy = new_E
                        improved = True
                        break
                        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
