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
    net = _odd_even_transposition()
    random.shuffle(net)
    return net

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states for fast verification
    # 2^18 inputs.
    # We pack 64 inputs per uint64.
    num_blocks = 2**N // 64  # 4096
    
    # states[w][block] bit b is 1 if input (block*64+b) has 1 on wire w
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Vectorized fill
    for w in range(N):
        if w < 6:
            # period <= 64, every block has the same 64-bit mask
            mask = 0
            for bit in range(64):
                if (bit >> w) & 1:
                    mask |= (1 << bit)
            states[w][:] = mask
        else:
            # period > 64, bit is constant within each block
            for b in range(num_blocks):
                if (b >> (w - 6)) & 1:
                    states[w][b] = (1 << 64) - 1
                else:
                    states[w][b] = 0

    # Pre-allocate workspace for verification
    workspace = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    def apply_comparator(ws, i, j):
        """Apply compare-exchange to workspace ws in-place."""
        a = ws[i]
        b = ws[j]
        ws[i] = a & b
        ws[j] = a | b

    def verify(comparators):
        """Verify network sorts all inputs. Returns True if valid."""
        for w in range(N):
            workspace[w][:] = states[w]
        
        for i, j in comparators:
            apply_comparator(workspace, i, j)
        
        # Check sorted
        for w in range(N - 1):
            if np.any(workspace[w] & ~workspace[w + 1]):
                return False
        return True

    def greedy_reduce(net, time_margin=2.0):
        """Remove redundant comparators using fast verification."""
        cur_net = list(net)
        t_margin = deadline - time_margin
        
        # First pass: try removing from the end
        while len(cur_net) > 0 and time.time() < t_margin:
            for w in range(N):
                workspace[w][:] = states[w]
            for idx in range(len(cur_net) - 1):
                i, j = cur_net[idx]
                apply_comparator(workspace, i, j)
            valid = True
            for w in range(N - 1):
                if np.any(workspace[w] & ~workspace[w + 1]):
                    valid = False
                    break
            if valid:
                cur_net.pop()
            else:
                break
        
        # Second pass: forward sweep
        changed = True
        while changed and time.time() < t_margin:
            changed = False
            idx = 0
            while idx < len(cur_net) and time.time() < t_margin:
                trial = cur_net[:idx] + cur_net[idx+1:]
                if verify(trial):
                    cur_net = trial
                    changed = True
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
    max_stagnation = 100
    
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
                
        elif move_type < 0.7:
            # Insert: Add a random comparator
            idx = random.randint(0, len(current_network))
            a, b = random.sample(range(N), 2)
            if a > b: a, b = b, a
            trial = current_network[:idx] + [(a, b)] + current_network[idx:]
        elif move_type < 0.85:
            # Slide: Shift a comparator by 1 position
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 1)
                trial = list(current_network)
                if idx == 0:
                    trial[0], trial[1] = trial[1], trial[0]
                elif idx == len(trial) - 1:
                    trial[idx], trial[idx-1] = trial[idx-1], trial[idx]
                else:
                    if random.random() < 0.5:
                        trial[idx], trial[idx-1] = trial[idx-1], trial[idx]
                    else:
                        trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
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
        
        # Only run greedy_reduce if trial is not too long
        if len(trial) <= len(best_network) + 3:
            reduced = greedy_reduce(trial, time_margin=1.5)
        else:
            reduced = trial if verify(trial) else None
        
        if reduced is not None:
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
                    K = random.randint(2, 4)
                    current_network = list(best_network)
                    for _ in range(K):
                        if not current_network: break
                        idx = random.randint(0, len(current_network) - 1)
                        a, b = random.sample(range(N), 2)
                        if a > b: a, b = b, a
                        current_network[idx] = (a, b)
                    
                    current_network = greedy_reduce(current_network, time_margin=1.5)
                    stagnation_counter = 0
        else:
            stagnation_counter += 1
            if stagnation_counter >= max_stagnation:
                K = random.randint(2, 4)
                current_network = list(best_network)
                for _ in range(K):
                    if not current_network: break
                    idx = random.randint(0, len(current_network) - 1)
                    a, b = random.sample(range(N), 2)
                    if a > b: a, b = b, a
                    current_network[idx] = (a, b)
                current_network = greedy_reduce(current_network, time_margin=1.5)
                stagnation_counter = 0

    return best_network
