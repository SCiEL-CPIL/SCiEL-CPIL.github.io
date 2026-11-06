import time
import random
import numpy as np

N = 17
MAX_TIME = 55.0  # Safety margin for 60s limit

# Precompute valid sorted states for quick lookup
# A state is sorted if it is 0^a 1^b.
# Values: 0, 2^17-2^16, 2^17-2^15, ..., 2^17-1
_VALID_MASK = np.zeros(1 << N, dtype=bool)
for k in range(N + 1):
    if k == 0:
        val = 0
    else:
        val = (1 << N) - (1 << (N - k))
    _VALID_MASK[val] = True

def seed_network():
    """Odd-even transposition network (136 comparators)."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def verify_network(net):
    """
    Vectorized verification using NumPy.
    Checks if the network sorts all 2^17 boolean inputs.
    """
    if not net:
        # Empty network only sorts if N<=1, which is false here.
        return N <= 1

    # Start with all possible inputs
    states = np.arange(1 << N, dtype=np.uint32)
    
    # Apply each comparator
    for i, j in net:
        # Extract bits
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        
        # Find states where i has 1 and j has 0 (needs swap)
        mask = (bi == 1) & (bj == 0)
        
        # Apply swap: XOR with mask of bits i and j
        if np.any(mask):
            states[mask] ^= (1 << i) | (1 << j)
            
    # Check if all resulting states are in the set of valid sorted states
    return np.all(_VALID_MASK[states])

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    rng = random.Random(42)
    
    # Start with seed
    best_net = seed_network()
    best_count = len(best_net)
    
    # Phase 1: Aggressive Greedy Minimization
    # Try to remove comparators until we hit a local minimum.
    # Since verification is fast, we can do multiple passes.
    current = list(best_net)
    
    # We use a "shuffle and remove" strategy.
    # To be robust, we restart the minimization a few times from the seed 
    # or from the current best if we get stuck.
    
    def minimize(net, time_limit):
        """Greedy removal of redundant comparators."""
        current = list(net)
        progress = True
        while progress and time.time() < time_limit:
            progress = False
            # Shuffle indices to break symmetry and find different removals
            indices = list(range(len(current)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                # Try removing
                new_net = current[:idx] + current[idx+1:]
                if verify_network(new_net):
                    current = new_net
                    progress = True
                    break # Restart the pass after a successful removal
        return current

    # Initial minimization
    current = minimize(current, deadline)
    if len(current) < best_count:
        best_net = current
        best_count = len(current)
        print(f"Phase 1: Reduced to {best_count}", flush=True)

    # Phase 2: Random Restart Hill Climbing
    # Try to escape local minima by:
    # 1. Swapping two comparators
    # 2. Replacing a comparator
    # 3. Adding a random comparator (then minimizing)
    # 4. Removing a comparator (if possible, though minimize handles this)
    
    # We will run this loop until time runs out.
    # Each iteration is a "move".
    
    while time.time() < deadline:
        # Choose a move type
        move_type = rng.random()
        
        if len(current) < 2:
            break
            
        if move_type < 0.4:
            # Swap two random comparators
            i, j = rng.sample(range(len(current)), 2)
            new_net = list(current)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            
        elif move_type < 0.7:
            # Replace a random comparator with a new random one
            idx = rng.randint(0, len(current) - 1)
            new_i = rng.randint(0, N - 1)
            new_j = rng.randint(0, N - 1)
            if new_i == new_j:
                continue
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            new_net = list(current)
            new_net[idx] = (new_i, new_j)
            
        elif move_type < 0.9:
            # Add a random comparator (to allow new structures)
            new_i = rng.randint(0, N - 1)
            new_j = rng.randint(0, N - 1)
            if new_i == new_j:
                continue
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            pos = rng.randint(0, len(current))
            new_net = current[:pos] + [(new_i, new_j)] + current[pos:]
            
        else:
            # Restart from best known (if current is much worse or stuck)
            # Or just try to minimize current again with a different shuffle
            new_net = list(current)
            # Force a minimization pass
            minimized = minimize(new_net, deadline)
            if len(minimized) < len(current):
                current = minimized
                if len(current) < best_count:
                    best_net = current
                    best_count = len(current)
                    print(f"Phase 2 (minimize): Improved to {best_count}", flush=True)
            continue

        # Verify the new network
        if verify_network(new_net):
            # Try to minimize it to see if we can reduce size
            # Only minimize if we have enough time
            if time.time() < deadline - 0.1:
                minimized = minimize(new_net, deadline)
                if len(minimized) < len(current):
                    current = minimized
                    if len(current) < best_count:
                        best_net = current
                        best_count = len(current)
                        print(f"Phase 2: Improved to {best_count}", flush=True)
                elif len(minimized) == len(current):
                    # Even if same size, accept if it's different structure (helps escape)
                    # But we prefer smaller. If same size, we might get stuck.
                    # Let's accept it to change the local landscape.
                    current = minimized
            else:
                current = new_net
                if len(current) < best_count:
                    best_net = current
                    best_count = len(current)
        else:
            # If invalid, we discard the move. 
            # To escape local minima, we sometimes need to accept invalid intermediate steps?
            # No, sorting networks must be valid at every step for the "network" definition?
            # Actually, the task asks for a network that sorts. Intermediate states don't need to sort, 
            # but the final list of comparators must form a valid sorting network.
            # My `verify_network` checks the whole list.
            # So if `new_net` is invalid, we cannot use it as a base for further minimization 
            # because minimization assumes a valid start (removal of redundant comps).
            # However, we can still try to fix it? No, that's complex.
            # Standard Hill Climbing on valid solutions only.
            pass

    # Final safety check
    if not verify_network(best_net):
        best_net = seed_network()
        
    return best_net

if __name__ == "__main__":
    # Quick test
    net = construct()
    print(f"Result size: {len(net)}")
    print(f"Valid: {verify_network(net)}")
