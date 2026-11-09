"""Size-efficient sorting network on 16 wires.
Strategy: Start from a small known-good network (e.g., a transposition network or a small subset of Batcher),
then perform a randomized local search (add, remove, swap, invert comparators) 
using a fast vectorized zero-one verifier to find a smaller valid network.
"""
import time
import numpy as np
import random

N = 16
TOTAL_ZERO_ONE = 1 << N  # 65536

# Deadline guard: hard-kill safety margin before the 90s wall-clock limit.
T_START = time.time()
DEADLINE = T_START + 85.0  # 5s safety margin


def _verify(comp_list):
    """Return True iff the compare-exchange network sorts every 0/1 input."""
    if len(comp_list) == 0:
        return True
    comps = np.asarray(comp_list, dtype=np.int64)
    idx = np.arange(TOTAL_ZERO_ONE)
    bits = (idx[:, None] >> np.arange(N)) & 1  # (65536, 16)
    for i, j in comps:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi
    # Sorted ascending on each wire: wire k must be <= wire k+1 for all k.
    return bool(np.all(bits[:, :-1] <= bits[:, 1:]))


def _get_initial_network():
    """Get a small initial valid network. We'll use a simple odd-even transposition network
    but maybe we can start even smaller? No, let's use a known small one or just generate a valid one.
    Actually, let's use a simple valid network: the odd-even transposition network is 120 comparators.
    Can we do better? A simple merge sort network is ~119.
    Let's just use the odd-even transposition network as a seed, it's easy to generate.
    """
    # Odd-even transposition network
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def construct():
    """Build the smallest verified 16-wire sorting network found in budget."""
    random.seed(42)
    np.random.seed(42)
    
    # Start with a valid network
    network = _get_initial_network()
    if not _verify(network):
        # Fallback: should not happen, but just in case
        return network
        
    best_network = network
    best_len = len(network)
    
    # Local search
    # We will try random moves: add, remove, swap, invert
    # To make it efficient, we only keep the network if it remains valid.
    # This is a "hill climbing" approach where we only move to valid neighbors.
    # Since we start valid, and only move if valid, we stay in the valid set.
    # We want to minimize length. So we primarily try removals.
    # But removals can get stuck in local minima. 
    # So we also try swaps and additions (which increase length, but might allow future removals).
    # However, adding comparators increases length, so we must be careful.
    # A better strategy: 
    # 1. Try to remove a random comparator. If valid, keep it.
    # 2. If stuck (can't remove any), try a random swap or add to escape local minimum.
    #    But we only accept the new state if it is valid.
    #    Since add increases length, we might not want to accept it permanently unless it leads to a better solution later.
    #    But local search usually accepts the move and continues.
    #    Let's try a simpler approach: 
    #    - Always try to remove. 
    #    - If no removal works in a while, try a random "perturbation" (swap/add) and keep it if valid, 
    #      even if it's larger, hoping to find a path to a smaller network.
    #    - Keep track of the best (smallest) valid network seen so far.
    
    stuck_counter = 0
    max_stuck = 1000
    
    while time.time() < DEADLINE:
        # Try to remove a random comparator
        if len(network) > 0:
            idx = random.randint(0, len(network) - 1)
            trial = network[:idx] + network[idx+1:]
            if _verify(trial):
                network = trial
                best_len = len(network)
                best_network = network
                stuck_counter = 0
                continue
        
        # If we couldn't remove (or network is empty), try a perturbation
        # We only perturb if we are "stuck" (haven't reduced size in a while)
        if stuck_counter > max_stuck:
            # Choose a random perturbation
            move_type = random.choice(['swap', 'add', 'invert'])
            
            if move_type == 'swap' and len(network) >= 2:
                i, j = random.sample(range(len(network)), 2)
                trial = network[:]
                trial[i], trial[j] = trial[j], trial[i]
                if _verify(trial):
                    network = trial
                    stuck_counter = 0
                    # Note: length is same, so best_len/best_network not updated unless we track separately
                    # But we want to find smaller ones, so this is just to escape local min.
                    
            elif move_type == 'add':
                # Add a random comparator
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                # Check if it's already there? Not strictly necessary, but might be redundant.
                trial = network + [(i, j)]
                if _verify(trial):
                    network = trial
                    stuck_counter = 0
                    # Length increased, but we might find a path to a smaller one later.
                    
            elif move_type == 'invert':
                # Invert a random comparator (swap i and j)
                idx = random.randint(0, len(network) - 1)
                i, j = network[idx]
                trial = network[:]
                trial[idx] = (j, i)
                if _verify(trial):
                    network = trial
                    stuck_counter = 0
        
        stuck_counter += 1
        
    return [(int(i), int(j)) for (i, j) in best_network]

# Run construct to test locally (optional, but good for debugging)
if __name__ == "__main__":
    start = time.time()
    net = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s, Comparators: {len(net)}")
    print(net)
