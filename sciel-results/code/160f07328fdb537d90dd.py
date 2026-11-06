import time
import random
import numpy as np
import sys

N = 17
MAX_TIME = 85.0  # Safety margin for 90s limit

# Precompute valid sorted states for quick lookup
# A state is sorted if it is 0^a 1^b.
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
        return N <= 1

    states = np.arange(1 << N, dtype=np.uint32)
    
    for i, j in net:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        mask = (bi == 1) & (bj == 0)
        if np.any(mask):
            states[mask] ^= (1 << i) | (1 << j)
            
    return np.all(_VALID_MASK[states])

def minimize_net(net, time_limit, rng):
    """Greedy removal of redundant comparators to reduce size."""
    current = list(net)
    progress = True
    while progress and time.time() < time_limit:
        progress = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > time_limit:
                break
            new_net = current[:idx] + current[idx+1:]
            if verify_network(new_net):
                current = new_net
                progress = True
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    rng = random.Random(42)
    
    # Start with seed and minimize it
    best_net = seed_network()
    best_count = len(best_net)
    current_net = minimize_net(best_net, deadline, rng)
    if len(current_net) < best_count:
        best_net = current_net
        best_count = len(current_net)
        print(f"Initial minimization: {best_count}", flush=True)

    # Strategy: Randomized Hill Climbing with "Error Repair"
    # The directive asks to test if 'error_repair' changes the chance of improvement.
    # We implement a specific repair mechanism: when a move breaks validity, 
    # we don't just discard it. Instead, we attempt to "repair" the network by 
    # inserting a small number of random comparators or reordering to restore validity,
    # and then re-minimizing. This allows traversing the "invalid" landscape 
    # which might contain better local minima.
    
    # Additionally, we use a "deep search" phase where we try to replace 
    # random blocks of comparators with new random structures, followed by aggressive repair/minimization.
    
    while time.time() < deadline:
        move_type = rng.random()
        
        if move_type < 0.3:
            # Standard Swap
            if len(current_net) < 2: continue
            i, j = rng.sample(range(len(current_net)), 2)
            new_net = list(current_net)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            is_valid = verify_network(new_net)
            
        elif move_type < 0.6:
            # Replace
            if not current_net: continue
            idx = rng.randint(0, len(current_net) - 1)
            new_i = rng.randint(0, N - 1)
            new_j = rng.randint(0, N - 1)
            if new_i == new_j: continue
            if new_i > new_j: new_i, new_j = new_j, new_i
            new_net = list(current_net)
            new_net[idx] = (new_i, new_j)
            is_valid = verify_network(new_net)
            
        elif move_type < 0.8:
            # Add
            new_i = rng.randint(0, N - 1)
            new_j = rng.randint(0, N - 1)
            if new_i == new_j: continue
            if new_i > new_j: new_i, new_j = new_j, new_i
            pos = rng.randint(0, len(current_net))
            new_net = current_net[:pos] + [(new_i, new_j)] + current_net[pos:]
            is_valid = verify_network(new_net)
            
        else:
            # Error Repair Strategy:
            # 1. Perform a destructive move (remove 2-3 random comparators)
            # 2. This likely breaks validity.
            # 3. Attempt to repair by adding random comparators or reordering until valid.
            # 4. Minimize the result.
            if len(current_net) < 3: continue
            
            # Destructive move
            remove_count = rng.randint(1, 3)
            indices_to_remove = rng.sample(range(len(current_net)), min(remove_count, len(current_net)))
            new_net = [comp for i, comp in enumerate(current_net) if i not in indices_to_remove]
            
            # Repair Phase
            # Try to restore validity by adding random comparators
            repaired = False
            for _ in range(50): # Max repair attempts
                if verify_network(new_net):
                    repaired = True
                    break
                # Add a random comparator
                new_i = rng.randint(0, N - 1)
                new_j = rng.randint(0, N - 1)
                if new_i == new_j: continue
                if new_i > new_j: new_i, new_j = new_j, new_i
                pos = rng.randint(0, len(new_net))
                new_net = new_net[:pos] + [(new_i, new_j)] + new_net[pos:]
                
            if not repaired:
                # If repair fails, try reordering the current valid best_net as a fallback
                # Or just skip this iteration
                continue
            
            is_valid = True # It was repaired

        if is_valid:
            # Minimize the new valid network
            if time.time() < deadline - 0.5:
                minimized = minimize_net(new_net, deadline, rng)
                if len(minimized) < len(current_net):
                    current_net = minimized
                    if len(current_net) < best_count:
                        best_net = current_net
                        best_count = len(current_net)
                        print(f"Improved to {best_count} (repair/success)", flush=True)
                elif len(minimized) < len(best_net):
                    best_net = minimized
                    best_count = len(minimized)
            else:
                if len(new_net) < best_count:
                    best_net = new_net
                    best_count = len(new_net)

    # Final safety check
    if not verify_network(best_net):
        best_net = seed_network()
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Result size: {len(net)}")
    print(f"Valid: {verify_network(net)}")
