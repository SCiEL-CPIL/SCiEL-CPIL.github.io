"""
Size-efficient sorting networks on 17 wires via bit-vector simulation and Simulated Annealing.
"""
import time
import random
import sys

N = 17
NUM_INPUTS = 1 << N
# Safety margin: The evaluator kills at 90s. We stop searching at 55s to ensure 
# clean exit and result serialization.
DEADLINE = 55.0

# Precompute bit-vector representation of initial states
# state[wire] is an integer where bit k is the value on that wire for input k.
# There are 2^N inputs, so we need 2^N bits per wire.
# Python big integers handle this efficiently.
def _init_bit_vectors():
    states = [0] * N
    # Optimization: Construct bit vectors using bit manipulation patterns
    # For N=17, we can use the fact that bit w of input k is (k>>w)&1
    # Use a more efficient construction
    for w in range(N):
        # Build the bit vector for wire w: bit k is 1 if (k>>w)&1
        # This is a pattern of 2^w zeros followed by 2^w ones, repeated
        period = 1 << (w + 1)
        half = 1 << w
        # Create one period: lower half 0, upper half 1
        # We need to tile this across all NUM_INPUTS bits
        # Efficient approach: build using shifts
        vec = 0
        for start in range(0, NUM_INPUTS, period):
            # Set bits from start+half to start+period-1
            mask = ((1 << half) - 1) << (start + half)
            vec |= mask
        states[w] = vec
    return states

INITIAL_STATES = _init_bit_vectors()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_network(comps):
    """
    Verify network using bit-vector simulation.
    comps: list of (i, j) tuples.
    Returns True if valid.
    """
    if not comps:
        return N <= 1
    
    # Copy initial states
    state = INITIAL_STATES[:]
    
    for i, j in comps:
        wi = state[i]
        wj = state[j]
        
        # Comparator (i, j): min to i, max to j.
        # min(wi, wj) = wi & wj
        # max(wi, wj) = wi | wj
        state[i] = wi & wj
        state[j] = wi | wj
        
    # Check if sorted: for all rows, state[:, k] <= state[:, k+1]
    # This means for all k, if state[w] has bit k set, state[w+1] must also have bit k set.
    # i.e., state[w] & ~state[w+1] must be 0 for all w.
    for w in range(N - 1):
        if state[w] & ~state[w + 1]:
            return False
            
    return True

def make_seed():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def greedy_remove(comps, rng, time_budget=2.0):
    """
    Try to remove each comparator if the network remains valid.
    Uses randomized order to escape local minima.
    Returns the shrunk network.
    """
    n = len(comps)
    improved = True
    while improved:
        improved = False
        if _time_left() < time_budget:
            return comps
            
        indices = list(range(n))
        rng.shuffle(indices)
        for idx in indices:
            if _time_left() < time_budget:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if verify_network(candidate):
                comps = candidate
                n = len(comps)
                improved = True
                break # Restart scan
    return comps

def add_remove_search(comps, rng):
    """
    Perform a local search step: add a random comparator, then greedily remove.
    This helps escape local minima.
    """
    if _time_left() < 2.0:
        return comps
    
    n = len(comps)
    # Try adding a random comparator
    i = rng.randrange(N)
    j = rng.randrange(N)
    if i == j:
        return comps
    if i > j:
        i, j = j, i
    
    # Insert at a random position
    pos = rng.randrange(n + 1)
    candidate = comps[:pos] + [(i, j)] + comps[pos:]
    
    # Greedily remove from this new candidate
    candidate = greedy_remove(candidate, rng)
    
    return candidate

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Start with odd-even transposition
    comps = make_seed()
    
    # First pass: aggressive greedy removal
    comps = greedy_remove(comps, rng)
    
    best = list(comps)
    best_len = len(best)
    current = list(comps)
    
    # Simulated Annealing Parameters
    # T is the maximum allowed size increase we are willing to consider.
    # We start high to allow exploration, then cool down.
    T = 20.0  
    cooling = 0.995
    min_T = 0.5
    
    # Track stagnation
    stagnant_count = 0
    
    # Main SA Loop
    while _time_left() > 5.0:
        # Local search on current to ensure it's locally optimal for its size
        current = greedy_remove(current, rng)
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
            # Reset temperature on improvement to explore more
            T = max(T, 10.0)
            stagnant_count = 0
        else:
            stagnant_count += 1
        
        # If stagnant for a while, do a stronger kick
        if stagnant_count > 50:
            # Strong exploration: add multiple random comparators
            n = len(current)
            for _ in range(rng.randint(2, 4)):
                i = rng.randrange(N)
                j = rng.randrange(N)
                if i != j:
                    if i > j:
                        i, j = j, i
                    pos = rng.randrange(len(current) + 1)
                    current.insert(pos, (i, j))
            current = greedy_remove(current, rng)
            stagnant_count = 0
            T = max(T, 12.0)
        
        # Generate neighbor
        neighbor = None
        r = rng.random()
        
        if r < 0.3:
            # Add-Remove kick (strong exploration)
            neighbor = add_remove_search(current, rng)
        else:
            # Standard mutation
            neighbor = list(current)
            n = len(neighbor)
            op = rng.choice(['remove', 'swap', 'move', 'insert', 'insert_double'])
            
            if op == 'remove' and n > 0:
                idx = rng.randrange(n)
                neighbor.pop(idx)
            elif op == 'swap' and n >= 2:
                i1 = rng.randrange(n)
                i2 = rng.randrange(n)
                if i1 != i2:
                    neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
            elif op == 'move' and n >= 2:
                i1 = rng.randrange(n)
                i2 = rng.randrange(n)
                if i1 != i2:
                    elem = neighbor.pop(i1)
                    neighbor.insert(i2, elem)
            elif op == 'insert':
                i = rng.randrange(N)
                j = rng.randrange(N)
                if i > j: 
                    i, j = j, i
                if i != j:
                    pos = rng.randrange(n + 1)
                    neighbor.insert(pos, (i, j))
            elif op == 'insert_double':
                # Insert two comparators to allow for more complex structural changes
                i1 = rng.randrange(N)
                j1 = rng.randrange(N)
                if i1 > j1: 
                    i1, j1 = j1, i1
                i2 = rng.randrange(N)
                j2 = rng.randrange(N)
                if i2 > j2: 
                    i2, j2 = j2, i2
                
                pos1 = rng.randrange(n + 1)
                pos2 = rng.randrange(n + 2)
                if pos2 > pos1:
                    pos2 += 1
                else:
                    pos2 += 1 # Adjust for first insert

                neighbor.insert(pos1, (i1, j1))
                neighbor.insert(pos2, (i2, j2))

            # Verify and shrink
            if verify_network(neighbor):
                neighbor = greedy_remove(neighbor, rng)
            else:
                # If invalid, revert to current (or just skip this neighbor)
                neighbor = list(current)
        
        # Acceptance criterion
        if len(neighbor) <= len(current):
            current = neighbor
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = len(neighbor) - len(current)
            if delta <= T:
                # Use a simpler probability model for speed
                prob = 0.5 ** (delta / max(T, 1e-6))
                if rng.random() < prob:
                    current = neighbor
        
        T *= cooling
        if T < min_T:
            T = min_T

    # Final greedy pass on best with more time
    best = greedy_remove(best, rng, time_budget=1.0)
    
    # One more careful pass: try removing each comparator in sequence
    n = len(best)
    for idx in range(n - 1, -1, -1):
        if _time_left() < 0.5:
            break
        candidate = best[:idx] + best[idx+1:]
        if verify_network(candidate):
            best = candidate
    
    return best

# Keep the old function name as a wrapper if required by strict interface checks, 
# though the prompt says "Keep the function names and signatures of the seed program".
# The seed had verify_network and verify_network_fast. 
# We define verify_network above. We alias verify_network_fast to it.
def verify_network_fast(network):
    return verify_network(network)

if __name__ == "__main__":
    # Quick self-test if run directly
    start = time.time()
    net = construct()
    print(f"Network size: {len(net)}, Time: {time.time()-start:.2f}s")
    print(f"Valid: {verify_network(net)}")
