import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START
# Precompute bit masks for fast verification
_ALL_MASKS = np.arange(1 << N, dtype=np.int64)
_WIRE_BITS = np.array([(_ALL_MASKS >> i) & 1 for i in range(N)], dtype=np.int8)

# Subset for fast negative verification
_SUBSET_SIZE = 2048
_SUBSET_MASKS = np.random.choice(1 << N, size=_SUBSET_SIZE, replace=False, dtype=np.int64)
_WIRE_BITS_SUBSET = np.array([(_SUBSET_MASKS >> i) & 1 for i in range(N)], dtype=np.int8)

def _verify_network_full(net):
    """Verify that a network sorts all 2^N Boolean inputs using bit-parallel ops."""
    if not net:
        return N <= 1
    wires = _WIRE_BITS.copy()
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True

def _verify_network_subset(net):
    """Verify on a subset of inputs for fast rejection."""
    if not net:
        return N <= 1
    wires = _WIRE_BITS_SUBSET.copy()
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True

def _verify_network(net):
    """Smart verification: subset first, then full."""
    if not _verify_network_subset(net):
        return False
    return _verify_network_full(net)

def _build_initial_network():
    """Build an initial sorting network using Odd-Even Transposition Sort."""
    net = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                net.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                net.append((i, i + 1))
    return net

def _random_comparator():
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    return (i, j)

def construct():
    start_time = time.time()
    deadline = 88.0  # Leave 2s margin

    # 1. Initialize with a valid network
    current_net = _build_initial_network()
    if not _verify_network_full(current_net):
        current_net = []
        for i in range(N):
            for j in range(i + 1, N):
                current_net.append((i, j))
    
    best_net = list(current_net)
    best_len = len(current_net)
    
    # Phase 1: Greedy Removal
    # Aggressively remove comparators that are not needed
    changed = True
    while changed and (time.time() - start_time < deadline):
        changed = False
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() - start_time > deadline:
                break
            candidate = current_net[:idx] + current_net[idx+1:]
            if _verify_network(candidate):
                current_net = candidate
                best_net = list(current_net)
                best_len = len(current_net)
                changed = True
                break # Restart loop to re-shuffle and check new positions
        
    # Phase 2: Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() - start_time < deadline:
        op = random.random()
        
        if op < 0.4:
            # Remove (40%)
            if not current_net:
                continue
            idx = random.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
            new_len = len(candidate)
        elif op < 0.7:
            # Swap (30%)
            if len(current_net) < 2:
                continue
            i, j = random.sample(range(len(current_net)), 2)
            candidate = list(current_net)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            new_len = len(candidate)
        elif op < 0.9:
            # Move (20%)
            if len(current_net) < 2:
                continue
            i, j = random.sample(range(len(current_net)), 2)
            candidate = list(current_net)
            comp = candidate.pop(i)
            candidate.insert(j, comp)
            new_len = len(candidate)
        else:
            # Change (10%)
            if not current_net:
                continue
            idx = random.randint(0, len(current_net) - 1)
            new_comp = _random_comparator()
            candidate = list(current_net)
            candidate[idx] = new_comp
            new_len = len(candidate)
            
        # Verify candidate
        if _verify_network(candidate):
            delta_len = new_len - len(current_net)
            
            if delta_len < 0:
                current_net = candidate
                if new_len < best_len:
                    best_net = list(current_net)
                    best_len = new_len
            else:
                if delta_len == 0:
                    if random.random() < 0.5:
                        current_net = candidate
                else:
                    prob = np.exp(-delta_len / T)
                    if random.random() < prob:
                        current_net = candidate
        else:
            pass
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    return best_net
# EVOLVE-BLOCK-END
