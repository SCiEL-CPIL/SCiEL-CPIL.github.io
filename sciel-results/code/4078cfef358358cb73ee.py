"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
SEED_COMPARATORS = 153
MAX_COMPARATORS = 4096
DEADLINE = 75.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _verify_bool(net):
    """Verify network sorts all 2^N Boolean inputs."""
    n = N
    sorted_vals = set()
    for k in range(n + 1):
        if k == 0:
            sorted_vals.add(0)
        else:
            sorted_vals.add(((1 << k) - 1) << (n - k))
    
    inputs = np.arange(1 << n, dtype=np.uint32)
    
    for (i, j) in net:
        bi = (inputs >> i) & 1
        bj = (inputs >> j) & 1
        mask = (bi == 1) & (bj == 0)
        if np.any(mask):
            inputs[mask] = (inputs[mask] & ~(1 << i)) | (1 << j)
    
    for v in inputs:
        if v not in sorted_vals:
            return False
    return True


def _mutate(net):
    """Apply a random mutation to the network."""
    net = list(net)
    op = random.random()
    
    if op < 0.3:
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(net))
        net.insert(pos, (i, j))
    elif op < 0.5:
        if len(net) > 60:
            pos = random.randint(0, len(net) - 1)
            net.pop(pos)
    elif op < 0.7:
        if len(net) >= 2:
            i = random.randint(0, len(net) - 1)
            j = random.randint(0, len(net) - 1)
            net[i], net[j] = net[j], net[i]
    elif op < 0.85:
        if net:
            pos = random.randint(0, len(net) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            net[pos] = (i, j)
    else:
        if len(net) >= 2:
            i = random.randint(0, len(net) - 2)
            j = random.randint(i + 1, len(net) - 1)
            net[i:j+1] = reversed(net[i:j+1])
    
    return net


def construct():
    start_time = time.time()
    random.seed(42)
    
    # Start with odd-even transposition (known correct, 153 comparators)
    best_net = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    best_len = len(best_net)
    
    if not _verify_bool(best_net):
        return best_net
    
    current_net = list(best_net)
    temp = 10.0
    cooling = 0.9995
    min_temp = 0.01
    
    while time.time() - start_time < DEADLINE - 5:
        candidate = _mutate(current_net)
        
        if len(candidate) > MAX_COMPARATORS:
            continue
        
        if _verify_bool(candidate):
            if len(candidate) < best_len:
                best_net = list(candidate)
                best_len = len(candidate)
                current_net = list(candidate)
                temp = 10.0
            else:
                delta = len(candidate) - best_len
                if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
                    current_net = list(candidate)
        
        temp = max(temp * cooling, min_temp)
        
        if time.time() - start_time > DEADLINE - 10:
            break
    
    return best_net

# EVOLVE-BLOCK-END
