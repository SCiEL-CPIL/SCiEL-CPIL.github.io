import time
import numpy as np
import random

N = 18
TIME_LIMIT = 55

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _bitonic_sort_network():
    """Generate a standard bitonic sorting network for N wires (N must be power of 2)."""
    # For N=18, we can't use standard bitonic directly on 18 wires easily without padding.
    # But standard bitonic for 16 wires has 60 comparators.
    # Let's use the known small network for N=16 (60 comps) and handle 18 wires?
    # Actually, standard bitonic for N=16 is 60 comparators.
    # For N=18, the known best is 77.
    # Let's just return a valid network, maybe a bitonic for 32 wires restricted? No.
    # Let's use a known compact network for N=16 and extend?
    # Better to stick to the seed or a known good start.
    # Let's return a random permutation of the seed to break symmetry.
    net = _odd_even_transposition()
    random.shuffle(net)
    return net

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states for fast verification
    # 2^18 inputs.
    # We pack 64 inputs per uint64.
    num_blocks = 2**N // 64
    
    # states[w][block] bit b is 1 if input (block*64+b) has 1 on wire w
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Fill states using vectorized approach per wire
    for w in range(N):
        for b in range(num_blocks):
            base = b * 64
            if w < 6:
                period = 1 << (w + 1)
                half = 1 << w
                mask = 0
                for bit in range(64):
                    if ((base + bit) >> w) & 1:
                        mask |= (1 << bit)
                states[w][b] = mask
            else:
                if (base >> w) & 1:
                    states[w][b] = (1 << 64) - 1
                else:
                    states[w][b] = 0

    # Pre-allocate workspace for verification
    workspace = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    def verify(comparators):
        """Verify network sorts all inputs. Uses pre-allocated workspace."""
        for w in range(N):
            workspace[w] = states[w]
        
        for i, j in comparators:
            a = workspace[i]
            b = workspace[j]
            workspace[i] = a & b
            workspace[j] = a | b
        
        for w in range(N - 1):
            if np.any(workspace[w] & ~workspace[w + 1]):
                return False
        return True

    def greedy_reduce(net):
        """Remove redundant comparators using fast verification."""
        cur_net = list(net)
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            idx = 0
            while idx < len(cur_net):
                if time.time() > deadline - 2:
                    return cur_net
                # Try removing comparator at idx
                trial = cur_net[:idx] + cur_net[idx+1:]
                if verify(trial):
                    cur_net = trial
                    changed = True
                else:
                    idx += 1
        return cur_net

    # Initial Network: Start with a known good structure.
    # Seed is 153. We can try to start with a slightly better known network?
    # For now, seed + greedy reduction.
    best_network = _odd_even_transposition()
    best_network = greedy_reduce(best_network)
    best_len = len(best_network)
    
    # Local Search
    current_network = list(best_network)
    stagnation_counter = 0
    max_stagnation = 200
    
    while time.time() < deadline - 5:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Mutation: Replace a random comparator with a new one
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                a, b = random.sample(range(N), 2)
                if a > b: a, b = b, a
                trial = list(current_network)
                trial[idx] = (a, b)
                
        elif move_type < 0.6:
            # Swap: Swap adjacent comparators
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 2)
                trial = list(current_network)
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                
        elif move_type < 0.8:
            # Insert: Add a random comparator
            idx = random.randint(0, len(current_network))
            a, b = random.sample(range(N), 2)
            if a > b: a, b = b, a
            trial = current_network[:idx] + [(a, b)] + current_network[idx:]
        else:
            # Move: Relocate a comparator to a new position
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 1)
                comp = current_network[idx]
                new_idx = random.randint(0, len(current_network) - 1)
                trial = list(current_network)
                del trial[idx]
                if new_idx > idx:
                    new_idx -= 1
                trial.insert(new_idx, comp)
        
        # Apply move and reduce
        reduced = greedy_reduce(trial)
        
        # Update best
        if len(reduced) < len(best_network):
            best_network = reduced
            best_len = len(reduced)
            current_network = reduced
            stagnation_counter = 0
        elif len(reduced) < len(current_network):
            current_network = reduced
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
            if stagnation_counter >= max_stagnation:
                # Kick: Perturb the best network to escape local minimum
                K = random.randint(2, 5)
                current_network = list(best_network)
                for _ in range(K):
                    if not current_network: break
                    idx = random.randint(0, len(current_network) - 1)
                    a, b = random.sample(range(N), 2)
                    if a > b: a, b = b, a
                    current_network[idx] = (a, b)
                
                current_network = greedy_reduce(current_network)
                stagnation_counter = 0

    return best_network
