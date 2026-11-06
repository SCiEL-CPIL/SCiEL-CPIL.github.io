"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random

def _apply_network(net, inputs):
    """Apply sorting network to a list of values. Returns sorted output."""
    vals = list(inputs)
    for (i, j) in net:
        if vals[i] > vals[j]:
            vals[i], vals[j] = vals[j], vals[i]
    return vals

def _is_sorted_network(net):
    """Verify network sorts all 2^16 boolean inputs using zero-one principle."""
    for mask in range(1 << N):
        inputs = [(mask >> k) & 1 for k in range(N)]
        result = _apply_network(net, inputs)
        # Check if result is sorted (non-decreasing)
        for k in range(N - 1):
            if result[k] > result[k + 1]:
                return False
    return True

def _greedy_prune(net, time_limit=75.0):
    """Greedily remove redundant comparators from the network."""
    net = list(net)
    deadline = time.time() + time_limit
    
    # Track which indices have been tried for removal
    # We'll do multiple passes, trying to remove each comparator
    
    removed = [False] * len(net)
    
    while time.time() < deadline:
        progress = False
        # Try removing each comparator in order
        for idx in range(len(net)):
            if time.time() > deadline:
                break
            if removed[idx]:
                continue
            # Try removing this comparator
            test_net = [net[i] for i in range(len(net)) if i != idx]
            if _is_sorted_network(test_net):
                removed[idx] = True
                net = test_net
                progress = True
        # If no progress in a full pass, try random order
        if not progress:
            # Shuffle and try again
            indices = list(range(len(net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                if removed[idx]:
                    continue
                test_net = [net[i] for i in range(len(net)) if i != idx]
                if _is_sorted_network(test_net):
                    removed[idx] = True
                    net = test_net
                    progress = True
            if not progress:
                break
    
    return net

def construct():
    # Start with odd-even transposition sorting network: 120 comparators for 16 wires.
    base_net = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Set a deadline to ensure we finish within the 90s budget
    deadline = time.time() + 80.0  # 10s safety margin
    
    # Try greedy pruning
    try:
        pruned = _greedy_prune(base_net, time_limit=75.0)
        # Verify the pruned network is correct
        if _is_sorted_network(pruned):
            return pruned
    except Exception:
        pass
    
    # Fallback: return the base network
    return base_net
# EVOLVE-BLOCK-END
