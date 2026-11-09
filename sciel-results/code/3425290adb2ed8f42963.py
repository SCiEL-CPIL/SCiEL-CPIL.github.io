import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N

def _build_seed_network():
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    return network

# Pre-compute initial state
_indices = np.arange(NUM_INPUTS)
_INITIAL_STATE = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
for w in range(N):
    _INITIAL_STATE[w] = ((_indices >> w) & 1).astype(np.uint8)

def _verify_network_np(network):
    state = _INITIAL_STATE.copy()
    for i, j in network:
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj
    return np.all(state[:-1] <= state[1:])

def _greedy_remove(network, deadline):
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            new_net = network[:idx] + network[idx+1:]
            if _verify_network_np(new_net):
                network = new_net
                changed = True
                break
    return network

def _simulated_annealing(network, start_time, deadline):
    current_net = list(network)
    current_size = len(current_net)
    best_net = list(current_net)
    best_size = current_size
    
    temperature = 1.0
    cooling_rate = 0.995
    
    while time.time() < deadline and temperature > 1e-5:
        move_type = random.choice(['delete', 'insert', 'swap'])
        
        if move_type == 'delete' and len(current_net) > 0:
            idx = random.randint(0, len(current_net) - 1)
            new_net = current_net[:idx] + current_net[idx+1:]
        elif move_type == 'insert':
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(current_net))
            new_net = current_net[:pos] + [(i, j)] + current_net[pos:]
        elif move_type == 'swap' and len(current_net) > 1:
            i = random.randint(0, len(current_net) - 1)
            j = random.randint(0, len(current_net) - 1)
            if i != j:
                new_net = list(current_net)
                new_net[i], new_net[j] = new_net[j], new_net[i]
            else:
                new_net = list(current_net)
        else:
            new_net = list(current_net)
            
        new_size = len(new_net)
        is_valid = _verify_network_np(new_net)
        
        if is_valid:
            delta_cost = new_size - current_size
            if delta_cost < 0:
                current_net = new_net
                current_size = new_size
                if new_size < best_size:
                    best_net = new_net
                    best_size = new_size
            else:
                if temperature > 0:
                    prob = np.exp(-delta_cost / temperature)
                    if random.random() < prob:
                        current_net = new_net
                        current_size = new_size
        else:
            pass
            
        temperature *= cooling_rate
        
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    network = _build_seed_network()
    network = _greedy_remove(network, deadline)
    best_net = _simulated_annealing(network, start_time, deadline)
    
    return best_net
