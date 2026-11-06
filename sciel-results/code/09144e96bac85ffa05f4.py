import numpy as np
import time
import random

N = 18
NUM_INPUTS = 1 << N

def _precompute_inputs():
    mat = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
    for i in range(N):
        half = 1 << i
        period = 2 * half
        pattern = np.concatenate([np.zeros(half, dtype=np.uint8), np.ones(half, dtype=np.uint8)])
        mat[i] = np.tile(pattern, NUM_INPUTS // period)
    return mat

INPUTS = _precompute_inputs()

def _simulate(comparators):
    mat = INPUTS.copy()
    for (i, j) in comparators:
        mi = mat[i] & mat[j]
        mj = mat[i] | mat[j]
        mat[i] = mi
        mat[j] = mj
    return mat

def _is_valid(comparators):
    mat = _simulate(comparators)
    for i in range(N - 1):
        if np.any(mat[i] > mat[i + 1]):
            return False
    return True

def _greedy_remove(net, deadline, order='backward'):
    changed = True
    while changed and time.time() < deadline:
        changed = False
        if order == 'backward':
            indices = range(len(net) - 1, -1, -1)
        elif order == 'forward':
            indices = range(len(net))
        else:  # random
            indices = list(range(len(net)))
            random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                return net
            if idx >= len(net):
                continue
            trial = net[:idx] + net[idx+1:]
            if _is_valid(trial):
                net = trial
                changed = True
                break
    return net

def _swap_and_optimize(net, idx, deadline):
    """Swap adjacent comparators at idx, then try greedy removal."""
    if idx >= len(net) - 1:
        return net
    candidate = list(net)
    candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
    if not _is_valid(candidate):
        return net
    # Try to remove comparators from the swapped network
    candidate = _greedy_remove(candidate, deadline, order='backward')
    if len(candidate) < len(net):
        return candidate
    return net

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Initial: odd-even transposition
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    if not _is_valid(net):
        return net
    
    # First pass: greedy removal from the back
    net = _greedy_remove(net, deadline, order='backward')
    best_net = list(net)
    best_len = len(net)
    
    # Multiple passes with different removal orders
    for pass_num in range(5):
        if time.time() > deadline - 5:
            break
        order = ['forward', 'backward', 'random', 'backward', 'forward'][pass_num]
        net = _greedy_remove(net, deadline, order=order)
        if len(net) < best_len:
            best_len = len(net)
            best_net = list(net)
    
    # Try swapping adjacent comparators to enable further removals
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for idx in range(len(best_net) - 1):
            if time.time() > deadline - 5:
                break
            candidate = _swap_and_optimize(best_net, idx, deadline)
            if len(candidate) < len(best_net):
                best_net = candidate
                best_len = len(candidate)
                improved = True
                break
        if not improved:
            # Try random swaps
            rng = random.Random(42 + iteration)
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                if len(best_net) < 2:
                    break
                i = rng.randint(0, len(best_net) - 2)
                candidate = list(best_net)
                candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
                if not _is_valid(candidate):
                    continue
                candidate = _greedy_remove(candidate, deadline, order='backward')
                if len(candidate) < best_len:
                    best_net = candidate
                    best_len = len(candidate)
                    improved = True
                    break
        if not improved and time.time() > deadline - 15:
            break
    
    # Random restarts with perturbation if time remains
    if time.time() < deadline - 10:
        rng = random.Random(123)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            # Create a perturbed version of the best network
            perturbed = list(best_net)
            # Randomly swap a few comparators
            for _ in range(3):
                if len(perturbed) > 1:
                    i = rng.randint(0, len(perturbed) - 2)
                    perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
            
            # Check if perturbed is valid
            if not _is_valid(perturbed):
                continue
            
            # Try to remove comparators
            temp_network = _greedy_remove(perturbed, deadline, order='backward')
            if len(temp_network) < best_len:
                best_net = temp_network
                best_len = len(temp_network)
    
    return best_net
