"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072
QUICK_CHECK_SIZE = 1000  # Number of random inputs for quick checks

# Precompute all Boolean inputs as a numpy array
def _get_all_bool_inputs():
    """Generate all 2^17 Boolean inputs as a numpy array of shape (131072, 17)."""
    inputs = np.zeros((NUM_BOOL_INPUTS, N), dtype=np.uint8)
    for j in range(N):
        block_size = 1 << (j + 1)
        half_block = 1 << j
        pattern = np.array([0] * half_block + [1] * half_block, dtype=np.uint8)
        num_tiles = NUM_BOOL_INPUTS // block_size
        inputs[:, j] = np.tile(pattern, num_tiles)
    return inputs

_ALL_INPUTS = _get_all_bool_inputs()

# Precompute a subset for quick checks
_quick_rng = np.random.RandomState(42)
_QUICK_INPUTS = _quick_rng.randint(0, 2, size=(QUICK_CHECK_SIZE, N)).astype(np.uint8)

def _simulate_network(network, inputs):
    """Simulate the network on given inputs. Modifies inputs in place."""
    for i, j in network:
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val

def _quick_check(network):
    """Quick check using a subset of inputs. Returns True if passes."""
    inputs = _QUICK_INPUTS.copy()
    _simulate_network(network, inputs)
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True

def _is_valid(network):
    """Full check using all Boolean inputs."""
    inputs = _ALL_INPUTS.copy()
    _simulate_network(network, inputs)
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True

def _construct_odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50  # Safety margin before 60s evaluator kill
    
    # Start with odd-even transposition
    network = _construct_odd_even_transposition()
    best_network = network[:]
    best_count = len(network)
    
    # Verify initial network
    if not _is_valid(network):
        return best_network
    
    # Phase 1: Greedy removal with quick checks
    while time.time() < deadline:
        removed_any = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            # Quick check first
            if _quick_check(candidate):
                # Full verification
                if _is_valid(candidate):
                    network = candidate
                    best_network = network[:]
                    best_count = len(network)
                    removed_any = True
                    break
        if not removed_any:
            break
    
    # Phase 2: Try moving comparators to different positions
    while time.time() < deadline:
        improved = False
        for idx in range(len(network)):
            if time.time() >= deadline:
                break
            comp = network[idx]
            for new_idx in range(len(network)):
                if new_idx == idx or new_idx == idx - 1:
                    continue
                if time.time() >= deadline:
                    break
                candidate = network[:idx] + network[idx+1:]
                insert_pos = new_idx if new_idx < idx else new_idx
                candidate = candidate[:insert_pos] + [comp] + candidate[insert_pos:]
                if len(candidate) == len(network) and candidate != network:
                    if _quick_check(candidate):
                        if _is_valid(candidate):
                            network = candidate
                            best_network = network[:]
                            best_count = len(network)
                            improved = True
                            break
            if improved:
                break
        if not improved:
            break
    
    return best_network
# EVOLVE-BLOCK-END
