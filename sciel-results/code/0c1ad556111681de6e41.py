import numpy as np
import time
import random

N = 18
NUM_INPUTS = 2 ** N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Generate all boolean inputs: shape (2^N, N)
    # Using bitwise operations to create the matrix
    # np.arange(2**N) is 262144
    # Reshape to (262144, 1)
    # Shift right by 17, 16, ..., 0
    # & 1
    full_inputs = (np.arange(NUM_INPUTS).reshape(-1, 1) >> np.arange(N-1, -1, -1)) & 1
    
    # Generate a subset for quick validation
    # 1000 random indices
    quick_indices = np.random.choice(NUM_INPUTS, 1000, replace=False)
    quick_inputs = full_inputs[quick_indices]
    
    def simulate(network, inputs):
        state = inputs.copy()
        for i, j in network:
            a = state[:, i]
            b = state[:, j]
            state[:, i] = np.minimum(a, b)
            state[:, j] = np.maximum(a, b)
        return state

    def is_sorted(state):
        if N <= 1:
            return True
        diffs = np.diff(state, axis=1)
        return np.all(diffs >= 0)

    def is_valid_quick(network):
        if len(network) > 4096:
            return False
        out = simulate(network, quick_inputs)
        return is_sorted(out)

    def is_valid_full(network):
        if len(network) > 4096:
            return False
        out = simulate(network, full_inputs)
        return is_sorted(out)

    # Seed network: Odd-even transposition
    # 162 comparators
    net = [(i, i + 1) for phase in range(N)
           for i in range(phase % 2, N - 1, 2)]
    
    # Verify seed is valid (it should be)
    # is_valid_full(net)
    
    best_net = net
    best_size = len(net)
    current_net = net
    
    # Phase 1: Peeling (Deterministic Reduction)
    # Try to remove comparators one by one from the end
    while time.time() < deadline:
        removed = False
        # Iterate in reverse
        for idx in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            candidate = current_net[:idx] + current_net[idx+1:]
            
            # Quick check
            if is_valid_quick(candidate):
                # Full check
                if is_valid_full(candidate):
                    current_net = candidate
                    removed = True
                    break # Restart loop as indices changed
        
        if not removed:
            break
            
    # Update best if improved
    if len(current_net) < best_size:
        best_net = current_net
        best_size = len(current_net)
        
    # Phase 2: Random Search (Hill Climbing)
    # Try to find even smaller networks
    random.seed(42)
    
    while time.time() < deadline:
        # Generate a random move
        move_type = random.choice(['remove', 'add', 'swap'])
        cand = list(current_net)
        
        if move_type == 'remove':
            if len(cand) == 0:
                continue
            idx = random.randint(0, len(cand) - 1)
            cand.pop(idx)
        elif move_type == 'add':
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(cand))
            cand.insert(pos, (i, j))
        else: # swap
            if len(cand) < 2:
                continue
            i1 = random.randint(0, len(cand) - 1)
            i2 = random.randint(0, len(cand) - 1)
            if i1 != i2:
                cand[i1], cand[i2] = cand[i2], cand[i1]
        
        # Only consider if size is smaller or equal (for exploration)
        # But primarily we want to reduce size.
        # If size is larger, it's unlikely to be the global min, but might be a step?
        # For simplicity, only accept if strictly smaller or (equal and we want to explore).
        # Let's stick to strictly smaller to ensure progress towards min size.
        
        if len(cand) < best_size:
            if is_valid_quick(cand):
                if is_valid_full(cand):
                    best_net = cand
                    best_size = len(cand)
                    current_net = cand
        elif len(cand) == best_size:
            # Optional: explore same size to find different local minima
            # This is risky for time. Skip for now.
            pass

    return best_net
