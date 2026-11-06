import time
import random

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85.0  # Safety margin from 90s limit

# Pre-compute masks at module load time to avoid doing it in construct()
_INITIAL_MASKS = [0] * N
for _w in range(N):
    block = 1 << _w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = (1 << N) // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    _INITIAL_MASKS[_w] = m

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Start with the seed network
    current = _odd_even_transposition()
    best = current[:]
    best_len = len(best)
    
    # Initial verification of seed (should be true)
    if not _verify(best):
        return best
        
    # Phase 1: Greedy removal of redundant comparators
    # Try to remove each comparator one by one. If the network is still valid, keep it removed.
    # This is a heuristic to quickly find a smaller network.
    changed = True
    while changed:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing current[i]
            neighbor = current[:i] + current[i+1:]
            if _verify(neighbor):
                current = neighbor
                changed = True
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
    
    # Phase 2: Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.99
    
    rng = random.Random(42)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline and T > T_min:
        # Weighted move selection favoring removal
        move_type = rng.choices(['remove', 'add', 'swap', 'move'], weights=[5, 2, 1, 1])[0]
        neighbor = current[:]
        
        if move_type == 'remove' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            neighbor.pop(idx)
            
        elif move_type == 'add':
            comp = rng.choice(all_comparators)
            pos = rng.randrange(len(neighbor) + 1)
            neighbor.insert(pos, comp)
            
        elif move_type == 'swap' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
        elif move_type == 'move' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                comp = neighbor.pop(i)
                if i < j:
                    j -= 1
                neighbor.insert(j, comp)

        if _verify(neighbor):
            delta = len(neighbor) - len(current)
            # Accept if better or with probability exp(-delta/T)
            if delta <= 0 or rng.random() < 2.718281828459045 ** (-delta / T):
                current = neighbor
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)

        T *= cooling_rate
        
    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best
