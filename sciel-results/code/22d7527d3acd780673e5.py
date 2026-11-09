"""Size-efficient sorting network on 18 wires via optimized local search with fast verification."""
import numpy as np
import time
import random

N = 18

def _verify_network(network):
    """
    Verify if the network sorts all 2^N Boolean inputs using the 0-1 principle.
    Uses NumPy vectorization with bitwise ops for speed.
    """
    n_comparators = len(network)
    if n_comparators == 0:
        return True
        
    comp_i = np.array([c[0] for c in network], dtype=np.int32)
    comp_j = np.array([c[1] for c in network], dtype=np.int32)
    
    total_inputs = 1 << N
    chunk_size = 1 << 16  # 65536
    arange_N = np.arange(N, dtype=np.int32)[None, :]
    
    for start in range(0, total_inputs, chunk_size):
        end = min(start + chunk_size, total_inputs)
        masks = np.arange(start, end, dtype=np.int64)
        
        # Generate inputs: shape (chunk_len, N), dtype int8
        inputs = ((masks[:, None] >> arange_N) & 1).astype(np.int8)
        
        # Apply network using bitwise ops (min=AND, max=OR for 0/1)
        for k in range(n_comparators):
            i, j = comp_i[k], comp_j[k]
            a = inputs[:, i]
            b = inputs[:, j]
            inputs[:, i] = a & b  # min
            inputs[:, j] = a | b  # max
        
        # Check sorted: each element <= next
        if np.any(np.diff(inputs, axis=1) < 0):
            return False
            
    return True

def _build_seed_network():
    """Build odd-even transposition network as seed."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _greedy_remove(network, time_budget=5.0):
    """Aggressively remove comparators with multiple passes."""
    start_time = time.time()
    current = network[:]
    
    while time.time() - start_time < time_budget:
        removed = False
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        for idx in indices:
            if idx >= len(current):
                continue
            new_net = current[:idx] + current[idx+1:]
            if _verify_network(new_net):
                current = new_net
                removed = True
                break
        
        if not removed:
            break
    
    return current

def construct():
    """Construct a sorting network on N=18 wires using optimized local search."""
    start_time = time.time()
    time_limit = 55.0
    
    # 1. Generate seed
    network = _build_seed_network()
    best_network = network[:]
    best_size = len(network)
    
    # 2. Aggressive greedy removal
    network = _greedy_remove(network, time_budget=time_limit * 0.12)
    best_network = network[:]
    best_size = len(network)
    
    # 3. Local search with simulated annealing
    T = 1.5
    current_size = len(network)
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        r = random.random()
        new_network = network[:]
        
        if r < 0.35:
            # Remove a random comparator
            if len(network) > 0:
                idx = random.randint(0, len(network) - 1)
                new_network = network[:idx] + network[idx+1:]
        elif r < 0.55:
            # Insert a random comparator
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            pos = random.randint(0, len(network))
            new_network = network[:pos] + [(i, j)] + network[pos:]
        elif r < 0.80:
            # Swap two comparators
            if len(network) > 1:
                i, j = random.sample(range(len(network)), 2)
                new_network = network[:]
                new_network[i], new_network[j] = new_network[j], new_network[i]
        else:
            # Move a comparator to a different position
            if len(network) > 1:
                i = random.randint(0, len(network) - 1)
                j = random.randint(0, len(network) - 1)
                if i != j:
                    new_network = network[:]
                    comp = new_network.pop(i)
                    new_network.insert(j, comp)
        
        new_size = len(new_network)
        
        if _verify_network(new_network):
            if new_size < current_size:
                network = new_network
                best_network = network[:]
                best_size = new_size
                current_size = new_size
                T = max(0.1, T * 0.97)
                no_improve_count = 0
            else:
                delta = new_size - current_size
                p = 0.05 if delta == 0 else np.exp(-delta / T)
                if random.random() < p:
                    network = new_network
                    current_size = new_size
                    T = max(0.1, T * 0.97)
                    no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Restart if stuck
        if no_improve_count > 150:
            if time.time() - start_time < time_limit * 0.85:
                network = _build_seed_network()
                network = _greedy_remove(network, time_budget=min(3.0, time_limit - (time.time() - start_time)))
                if len(network) < best_size:
                    best_network = network[:]
                    best_size = len(network)
                current_size = len(network)
                T = 1.0
                no_improve_count = 0
    
    return best_network

# Cache the result
_cached_network = None

_original_construct = construct

def construct():
    global _cached_network
    if _cached_network is None:
        _cached_network = _original_construct()
    return _cached_network
