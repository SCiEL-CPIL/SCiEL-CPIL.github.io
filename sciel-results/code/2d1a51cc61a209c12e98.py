import time
import numpy as np
import random

N = 18
TIME_LIMIT = 55  # Safety margin for 90s limit (60s eval + buffer)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

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
    # For wire w, input k has bit 1 if (k >> w) & 1
    # We can build this more efficiently:
    for w in range(N):
        # For each block of 64, the pattern of bits on wire w
        # depends on the block's base address
        for b in range(num_blocks):
            base = b * 64
            # The 64 consecutive integers base..base+63
            # Their w-th bit: if w < 6, the pattern repeats every 2^(w+1)
            # For w >= 6, the bit is constant across a block of 64
            if w < 6:
                period = 1 << (w + 1)
                half = 1 << w
                # Within 64 consecutive numbers starting at base:
                # The w-th bit toggles every 'half' numbers, on for half, off for half
                # Starting phase: (base >> w) & 1 determines if we start in the "on" or "off" half
                # Actually, let's just compute directly for small w
                # Generate the 64-bit mask
                mask = 0
                for bit in range(64):
                    if ((base + bit) >> w) & 1:
                        mask |= (1 << bit)
                states[w][b] = mask
            else:
                # w >= 6: all 64 numbers in a block have the same w-th bit
                if (base >> w) & 1:
                    states[w][b] = (1 << 64) - 1
                else:
                    states[w][b] = 0

    # Pre-allocate workspace for verification to avoid repeated copies
    workspace = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    def verify(comparators):
        """Verify network sorts all inputs. Uses pre-allocated workspace."""
        # Copy initial states into workspace
        for w in range(N):
            workspace[w] = states[w]
        
        # Apply comparators
        for i, j in comparators:
            a = workspace[i]
            b = workspace[j]
            workspace[i] = a & b
            workspace[j] = a | b
        
        # Check sorted: for each adjacent pair, no 1 on lower wire when upper is 0
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
                    # Do not increment idx
                else:
                    idx += 1
        return cur_net

    # Initial Network
    best_network = _odd_even_transposition()
    best_network = greedy_reduce(best_network)
    best_len = len(best_network)
    
    # Local Search
    current_network = list(best_network)
    stagnation_counter = 0
    max_stagnation = 300
    
    while time.time() < deadline - 5:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.35:
            # Mutation: Replace a random comparator with a new one
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                a, b = random.sample(range(N), 2)
                if a > b: a, b = b, a
                trial = list(current_network)
                trial[idx] = (a, b)
                
        elif move_type < 0.55:
            # Swap: Swap adjacent comparators
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 2)
                trial = list(current_network)
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                
        elif move_type < 0.75:
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
                # Adjust new_idx if needed
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
            # Improved current local minimum, but not global best
            current_network = reduced
            stagnation_counter = 0
        else:
            # No improvement from this move
            stagnation_counter += 1
            
            if stagnation_counter >= max_stagnation:
                # Kick: Perturb the best network to escape local minimum
                K = random.randint(2, 6)
                current_network = list(best_network)
                for _ in range(K):
                    if not current_network: break
                    idx = random.randint(0, len(current_network) - 1)
                    a, b = random.sample(range(N), 2)
                    if a > b: a, b = b, a
                    current_network[idx] = (a, b)
                
                # Reduce the perturbed network
                current_network = greedy_reduce(current_network)
                stagnation_counter = 0

    return best_network
