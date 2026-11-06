import math
import random
import time
def construct():
    start_time = time.time()
    deadline = start_time + 58.0
    N = 18
    total_inputs = 1 << N
    
    # Precompute initial wires as Python integers
    # Wire k has bit i set if input i has 1 at position k
    init_wires = [0] * N
    for k in range(N):
        # Construct the integer for wire k
        # Pattern: 0...0 1...1
        # We can build it by chunks
        w = 0
        # Optimization:
        # The value is sum( (1<<i) for i in range(total_inputs) if (i>>k)&1 )
        # This is a geometric series of bits.
        # For k=0: 010101... (binary)
        # For k=1: 00110011...
        # We can generate this using bit tricks.
        # Let's just use a loop for simplicity first, it's only 4.7M ops.
        # Actually, we can do it in O(N^2) or less.
        # For wire k, the pattern repeats every 2^(k+1) bits.
        # In each block of 2^(k+1), the first 2^k are 0, next 2^k are 1.
        # So the block value is ( (1 << (1<<k)) - 1 ) << (1<<k) ? No.
        # The bits are at positions i.
        # Let's just use the loop, it's safe.
        for i in range(total_inputs):
            if (i >> k) & 1:
                w |= (1 << i)
        init_wires[k] = w

    def verify_network(comparators):
        # Copy the wires
        # Python ints are immutable, so we can just reference them if we don't modify?
        # No, we need to update wires.
        wires = list(init_wires)
        for i, j in comparators:
            wi = wires[i]
            wj = wires[j]
            # min to i, max to j
            wires[i] = wi & wj
            wires[j] = wi | wj
        
        # Check sortedness
        # wires[k] must be subset of wires[k+1]
        # i.e., wires[k] & ~wires[k+1] == 0
        for k in range(N - 1):
            if wires[k] & ~wires[k + 1]:
                return False
        return True

    # Initial Network: Odd-Even Transposition
    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    best_network = odd_even_transposition()
    best_size = len(best_network)
    
    # Randomized Greedy Removal with SA
    # We will run multiple restarts or a long SA with aggressive removal.
    
    # Move Set:
    # 1. Remove a random comparator
    # 2. Insert a random comparator
    # 3. Swap two comparators
    
    def get_neighbor(net, move_type):
        n = len(net)
        if n == 0:
            return None
        if move_type == 'remove':
            if n == 0: return None
            idx = random.randint(0, n - 1)
            new_net = net[:idx] + net[idx+1:]
            return new_net
        elif move_type == 'insert':
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                return None
            if i > j: i, j = j, i
            pos = random.randint(0, n)
            new_net = net[:pos] + [(i, j)] + net[pos:]
            return new_net
        elif move_type == 'swap':
            if n < 2: return None
            i, j = random.sample(range(n), 2)
            new_net = list(net)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            return new_net
        return None

    # SA Parameters
    t_start = 100.0
    t_end = 0.1
    alpha = 0.999
    
    current_net = list(best_network)
    current_energy = len(current_net)
    best_net = list(current_net)
    best_energy = current_energy
    
    t = t_start
    iterations = 0
    max_iterations = 500000 # High limit, controlled by time
    
    while time.time() < deadline:
        # Choose move
        # Bias towards removal if we have room, or if we are large
        # If stuck, bias towards insert/swap
        r = random.random()
        if r < 0.6:
            move = 'remove'
        elif r < 0.75:
            move = 'insert'
        else:
            move = 'swap'
            
        candidate = get_neighbor(current_net, move)
        if candidate is None:
            # If we can't make a move (e.g. empty net and tried remove), restart or break
            if move == 'remove' and len(current_net) == 0:
                break
            continue
        
        # Verify
        if not verify_network(candidate):
            continue # Reject invalid
        
        new_energy = len(candidate)
        delta = new_energy - current_energy
        
        if delta < 0:
            # Improvement
            current_net = candidate
            current_energy = new_energy
            if new_energy < best_energy:
                best_net = list(candidate)
                best_energy = new_energy
                # Optional: greedy removal burst on new best
                # Let's do a quick greedy pass
                changed = True
                while changed:
                    changed = False
                    n = len(best_net)
                    if n == 0: break
                    # Try to remove each
                    for i in range(n):
                        if time.time() > deadline - 5: break
                        test = best_net[:i] + best_net[i+1:]
                        if verify_network(test):
                            best_net = test
                            changed = True
                            break # Restart scan
                    if time.time() > deadline - 5: break
        else:
            # Worse or equal
            if delta == 0:
                # Accept equal to move around
                if random.random() < 0.5:
                    current_net = candidate
                    current_energy = new_energy
            else:
                # Boltzmann
                prob = math.exp(-delta / t)
                if random.random() < prob:
                    current_net = candidate
                    current_energy = new_energy
        
        t *= alpha
        if t < t_end:
            # Restart with higher temp if stuck?
            # Simple restart
            if time.time() > deadline - 10:
                break
            t = t_start * 0.5
            current_net = list(best_net)
            current_energy = best_energy
            
        iterations += 1

    return best_net
