"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

_INITIAL_MASKS_CACHE = {}

def _get_initial_masks(n_wires):
    if n_wires not in _INITIAL_MASKS_CACHE:
        total_inputs = 1 << n_wires
        masks = []
        for w in range(n_wires):
            m = 0
            for k in range(total_inputs):
                if (k >> w) & 1:
                    m |= (1 << k)
            masks.append(m)
        _INITIAL_MASKS_CACHE[n_wires] = masks
    return _INITIAL_MASKS_CACHE[n_wires]

def _verify_sorting_network(network, n_wires):
    if not network:
        return n_wires <= 1
    init_masks = _get_initial_masks(n_wires)
    masks = init_masks[:]
    for i, j in network:
        if i >= j:
            return False
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    for w in range(n_wires - 1):
        if masks[w] & ~masks[w + 1]:
            return False
    return True

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

def _greedy_remove(network, deadline):
    changed = True
    while changed:
        changed = False
        # Pre-compute states before each comparator
        init_masks = _get_initial_masks(N)
        masks = init_masks[:]
        state_before = [masks[:]]
        for i, j in network:
            mi = masks[i]
            mj = masks[j]
            masks[i] = mi & mj
            masks[j] = mi | mj
            state_before.append(masks[:])
        
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            
            # Verify removal of comparator i
            candidate_masks = state_before[i][:]
            valid = True
            for k in range(i + 1, len(network)):
                a, b = network[k]
                ma = candidate_masks[a]
                mb = candidate_masks[b]
                candidate_masks[a] = ma & mb
                candidate_masks[b] = ma | mb
            
            for w in range(N - 1):
                if candidate_masks[w] & ~candidate_masks[w + 1]:
                    valid = False
                    break
            
            if valid:
                network = network[:i] + network[i+1:]
                changed = True
                break
    
    return network

def _local_search_step(network, deadline):
    best_net = list(network)
    best_size = len(best_net)
    
    attempts = 0
    max_attempts = 200
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        candidate = list(best_net)
        
        if len(candidate) > 1 and random.random() < 0.5:
            i = random.randint(0, len(candidate) - 2)
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
        else:
            if len(candidate) > 0:
                idx = random.randint(0, len(candidate) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate[idx] = (i, j)
        
        if _verify_sorting_network(candidate, N):
            cleaned = _greedy_remove(candidate, deadline)
            if len(cleaned) < best_size:
                best_net = cleaned
                best_size = len(best_net)
    
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    network = _build_seed_network()
    best_network = list(network)
    best_size = len(network)
    
    network = _greedy_remove(network, deadline)
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    while time.time() < deadline - 2.0:
        net = _local_search_step(best_network, deadline)
        if len(net) < best_size:
            best_size = len(net)
            best_network = list(net)
            if time.time() < deadline - 2.0:
                cleaned_net = _greedy_remove(best_network, deadline)
                if len(cleaned_net) < best_size:
                    best_size = len(cleaned_net)
                    best_network = list(cleaned_net)
    
    return best_network
