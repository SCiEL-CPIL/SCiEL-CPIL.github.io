import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N
BLOCK_SIZE = 64
NUM_BLOCKS = NUM_INPUTS // BLOCK_SIZE

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # Pre-compute target states for all 2^18 inputs
    indices = np.arange(NUM_INPUTS, dtype=np.uint32).reshape(NUM_BLOCKS, BLOCK_SIZE)
    
    # Compute bits and popcounts
    bits = (indices[:, :, None] >> np.arange(N)[None, None, :]) & 1
    popcounts = bits.sum(axis=2)
    
    def pack_bits_to_u64(arr_2d):
        flat = arr_2d.flatten()
        packed = np.packbits(flat, axis=0, bitorder='little')
        packed = packed.reshape(NUM_BLOCKS, 8)
        return packed.view(np.uint64).flatten()
    
    # Compute targets for each wire
    targets = []
    for i in range(N):
        mask = (popcounts >= (N - i))
        targets.append(pack_bits_to_u64(mask))
    
    # Initialize wires (identity permutation)
    wires = []
    for i in range(N):
        wires.append(pack_bits_to_u64(bits[:, :, i]))
    
    def apply_comp(wires, i, j):
        new_wires = wires.copy()
        a = new_wires[i]
        b = new_wires[j]
        new_a = a & b
        new_b = a | b
        new_wires[i] = new_a
        new_wires[j] = new_b
        return new_wires
    
    def calc_error(wires):
        err = 0
        for i in range(N):
            diff = wires[i] ^ targets[i]
            err += np.bitwise_count(diff).sum()
        return err

    # Greedy search
    current_wires = wires.copy()
    current_error = calc_error(current_wires)
    network = []
    
    while time.time() < deadline - 20.0:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        
        for i in range(N):
            for j in range(i + 1, N):
                new_wires = apply_comp(current_wires, i, j)
                new_error = calc_error(new_wires)
                if new_error < best_error:
                    best_error = new_error
                    best_comp = (i, j)
                    best_new_wires = new_wires
                    
        if best_comp is None:
            break
            
        current_wires = best_new_wires
        current_error = best_error
        network.append(best_comp)
        
        if current_error == 0:
            break

    if current_error == 0:
        return network

    # Reduction phase: try to remove redundant comparators
    def evaluate_network(net):
        w = wires.copy()
        for i, j in net:
            w = apply_comp(w, i, j)
        return calc_error(w)
    
    best_net = network.copy()
    best_err = evaluate_network(best_net)
    
    # Try removing each comparator
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for idx in range(len(best_net)):
            if time.time() > deadline - 5.0:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            err = evaluate_network(candidate)
            if err <= best_err:
                best_net = candidate
                best_err = err
                changed = True
                break  # Restart from beginning after a removal
    
    # Local search: try swapping adjacent comparators
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for idx in range(len(best_net) - 1):
            if time.time() > deadline - 3.0:
                break
            candidate = best_net.copy()
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            err = evaluate_network(candidate)
            if err < best_err:
                best_net = candidate
                best_err = err
                improved = True
                break
    
    # Genetic algorithm phase with faster fitness
    pop_size = 30
    population = [best_net.copy() for _ in range(pop_size)]
    
    for k in range(pop_size // 2):
        random_net = []
        for _ in range(15 + np.random.randint(0, 15)):
            i = np.random.randint(0, N)
            j = np.random.randint(i + 1, N)
            random_net.append((i, j))
        population[k] = random_net
    
    best_net = best_net.copy()
    best_err = evaluate_network(best_net)
    
    while time.time() < deadline:
        fits = [evaluate_network(net) for net in population]
        
        min_idx = np.argmin(fits)
        if fits[min_idx] < best_err:
            best_err = fits[min_idx]
            best_net = population[min_idx].copy()
        
        new_population = []
        for _ in range(pop_size):
            idxs = np.random.choice(pop_size, 3, replace=False)
            winner = idxs[np.argmin([fits[i] for i in idxs])]
            new_population.append(population[winner].copy())
        
        for k in range(pop_size):
            if np.random.rand() < 0.7 and k + 1 < pop_size:
                other = population[np.random.randint(0, pop_size)]
                crossover_point = np.random.randint(0, max(len(new_population[k]), len(other), 1))
                child = new_population[k][:crossover_point] + other[crossover_point:]
                new_population[k] = child
            
            if np.random.rand() < 0.5:
                pos = np.random.randint(0, len(new_population[k]) + 1)
                i = np.random.randint(0, N)
                j = np.random.randint(i + 1, N)
                new_population[k].insert(pos, (i, j))
            
            if np.random.rand() < 0.3 and new_population[k]:
                pos = np.random.randint(0, len(new_population[k]))
                new_population[k].pop(pos)
            
            if np.random.rand() < 0.2 and len(new_population[k]) >= 2:
                i1 = np.random.randint(0, len(new_population[k]))
                i2 = np.random.randint(0, len(new_population[k]))
                new_population[k][i1], new_population[k][i2] = new_population[k][i2], new_population[k][i1]
        
        population = new_population
        population[0] = best_net.copy()
    
    if best_err == 0:
        return best_net
    elif len(best_net) < len(network):
        return best_net
    else:
        return best_net
