import time
import random
import math

N = 18
TOTAL_INPUTS = 1 << N
DEADLINE = 85.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    def make_wires():
        wires = []
        for i in range(N):
            period_width = 1 << (i + 1)
            num_periods_bits = N - i - 1
            mask = (1 << (1 << i)) - 1
            period = mask << (1 << i)
            x = period
            shift = period_width
            for _ in range(num_periods_bits):
                x |= x << shift
                shift <<= 1
            wires.append(x)
        return wires

    def is_sorted(wires):
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return False
        return True

    def verify_network(network, skip_idx=None):
        wires = make_wires()
        for idx, (i, j) in enumerate(network):
            if idx == skip_idx:
                continue
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj
        return is_sorted(wires)

    # 1. Generate Odd-Even Transposition Network
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
            
    if not verify_network(network):
        for i in range(N):
            for j in range(N - 1 - i):
                network.append((j, j + 1))
        assert verify_network(network)
        
    best_network = network[:]
    best_size = len(network)
    
    # 2. Greedy Removal
    removed = True
    while removed and (time.time() - start_time) < DEADLINE - 15:
        removed = False
        for idx in range(len(network)):
            if (time.time() - start_time) > DEADLINE - 10:
                break
            if verify_network(network, skip_idx=idx):
                network.pop(idx)
                best_network = network[:]
                best_size = len(network)
                removed = True
                break
    
    # 3. Simulated Annealing
    current_network = best_network[:]
    current_size = best_size
    T = 5.0  # Initial temperature
    cooling = 0.999
    
    while (time.time() - start_time) < DEADLINE - 5:
        if not current_network:
            break
            
        move_type = random.choice(['remove', 'add', 'swap'])
        new_network = current_network[:]
        
        if move_type == 'remove':
            if len(new_network) == 0:
                continue
            idx = random.randint(0, len(new_network) - 1)
            new_network.pop(idx)
            
        elif move_type == 'add':
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_network.append((i, j))
            
        elif move_type == 'swap':
            if len(new_network) < 2:
                continue
            i = random.randint(0, len(new_network) - 2)
            j = random.randint(i + 1, len(new_network) - 1)
            new_network[i], new_network[j] = new_network[j], new_network[i]
            
        if verify_network(new_network):
            new_size = len(new_network)
            delta = new_size - current_size
            
            if delta < 0:
                # Improvement
                current_network = new_network
                current_size = new_size
                if new_size < best_size:
                    best_network = current_network[:]
                    best_size = new_size
                T = max(T * cooling, 0.5)
            elif delta == 0:
                # Neutral
                if random.random() < 0.5:
                    current_network = new_network
                T = max(T * cooling, 0.5)
            else:
                # Worse, accept with probability
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_network = new_network
                    current_size = new_size
                T = max(T * cooling, 0.5)
        else:
            # Invalid, reject
            T = max(T * cooling, 0.5)
            
    return best_network
# EVOLVE-BLOCK-END
