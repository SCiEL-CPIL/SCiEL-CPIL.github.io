"""Size-efficient sorting networks on 17 wires via structural optimization."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 58.0  # Safety margin before 60s limit
START_TIME = time.time()

# Precompute initial states for all 2^N inputs using NumPy.
# Shape: (NUM_INPUTS, N), dtype: uint8
def _init_states():
    inputs = np.arange(NUM_INPUTS, dtype=np.uint32)
    states = (inputs[:, None] >> np.arange(N)) & 1
    return states.astype(np.uint8)

INITIAL_STATES = _init_states()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_network(comps):
    """
    Verify network using NumPy vectorized simulation.
    comps: list of (i, j) tuples.
    Returns True if valid.
    """
    if not comps:
        return False
        
    # Copy initial states
    state = INITIAL_STATES.copy()
    
    for i, j in comps:
        wi = state[:, i]
        wj = state[:, j]
        
        # Comparator (i, j): min to i, max to j.
        new_wi = np.minimum(wi, wj)
        new_wj = np.maximum(wi, wj)
        
        state[:, i] = new_wi
        state[:, j] = new_wj
        
    # Check if sorted: for all rows, state[:, k] <= state[:, k+1]
    # Since values are 0 or 1:
    # sorted if state[:, k] <= state[:, k+1] for all k.
    # This is true if there is no index where state[:, k] == 1 and state[:, k+1] == 0.
    
    # Cast to signed integer to avoid wrap-around in subtraction
    state_signed = state.astype(np.int8)
    diffs = state_signed[:, 1:] - state_signed[:, :-1]
    if np.any(diffs < 0):
        return False
        
    return True

def make_seed():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def greedy_remove(comps):
    """
    Try to remove each comparator if the network remains valid.
    """
    n = len(comps)
    improved = True
    while improved:
        improved = False
        for idx in range(n - 1, -1, -1):
            if _time_left() < 2.0:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if verify_network(candidate):
                comps = candidate
                n = len(comps)
                improved = True
                break # Restart scan
    return comps

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Start with odd-even transposition
    comps = make_seed()
    
    # First pass: aggressive greedy removal
    comps = greedy_remove(comps)
    
    best = list(comps)
    best_len = len(best)
    
    # Local Search with Structural Moves
    # We use a combination of:
    # 1. Removal (primary goal)
    # 2. Swap (to escape local minima where removal is blocked)
    # 3. Move (to reorder comparators, potentially enabling new removals)
    # 4. "Simulated Annealing" style acceptance of same-size valid moves to explore
    
    current = list(best)
    current_len = best_len
    temperature = 10.0  # Initial "temperature" for accepting non-improving moves
    
    last_improvement = time.time()
    
    while _time_left() > 1.0:
        # Generate a candidate via mutation
        candidate = list(current)
        n = len(candidate)
        
        # Choose operation
        # Bias towards removal if we are getting stuck
        if n > 0:
            # 60% remove, 20% swap, 20% move
            op = rng.random()
            if op < 0.6:
                op = 'remove'
            elif op < 0.8:
                op = 'swap'
            else:
                op = 'move'
        else:
            op = 'remove'
            
        if op == 'remove':
            if n == 0:
                continue
            idx = rng.randrange(n)
            candidate.pop(idx)
        elif op == 'swap' and n >= 2:
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            if i1 != i2:
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        elif op == 'move' and n >= 1:
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            if i1 != i2:
                elem = candidate.pop(i1)
                candidate.insert(i2, elem)
        
        # Evaluate candidate
        is_valid = verify_network(candidate)
        
        if is_valid:
            new_len = len(candidate)
            
            if new_len < current_len:
                # Improvement
                current = candidate
                current_len = new_len
                last_improvement = time.time()
                
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
                    # Reset temperature on improvement
                    temperature = max(1.0, temperature * 0.5)
                    
                # Try greedy removal immediately after improvement
                current = greedy_remove(current)
                current_len = len(current)
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
                    
            elif new_len == current_len:
                # Same size, valid.
                # Accept with some probability to explore (Simulated Annealing style)
                # This helps escape local minima where no single removal works,
                # but a sequence of swaps/moves followed by a removal does.
                if rng.random() < 0.3:  # 30% chance to accept neutral move
                    current = candidate
                    # Run a quick greedy check to see if this new configuration allows removal
                    temp = greedy_remove(current)
                    if len(temp) < current_len:
                        current = temp
                        current_len = len(current)
                        last_improvement = time.time()
                        if current_len < best_len:
                            best = list(current)
                            best_len = current_len
                            temperature = max(1.0, temperature * 0.5)
                    else:
                        # No immediate gain, but we moved.
                        # Slight cooling
                        temperature = max(0.5, temperature * 0.95)
            else:
                # Larger size (should not happen with remove/swap/move, but just in case)
                pass
        else:
            # Invalid move
            # Cool down slightly
            temperature = max(0.5, temperature * 0.98)
            
        # If no improvement for a long time, increase temperature (restart exploration)
        if time.time() - last_improvement > 5.0:
            temperature = min(20.0, temperature * 1.2)
            # Occasionally, restart from a slightly perturbed best to escape deep local minima
            if rng.random() < 0.1:
                current = list(best)
                # Perturb best
                for _ in range(rng.randint(1, 3)):
                    if len(current) > 0:
                        idx = rng.randrange(len(current))
                        current.pop(idx)
                        if not verify_network(current):
                            current.insert(idx, best[idx])
                current_len = len(current)
                last_improvement = time.time()

    # Final greedy pass
    best = greedy_remove(best)
    
    return best

if __name__ == "__main__":
    # Quick test
    net = construct()
    print(f"Network size: {len(net)}")
    if verify_network(net):
        print("Network is valid.")
    else:
        print("Network is INVALID!")
