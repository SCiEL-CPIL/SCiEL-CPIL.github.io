"""Size-efficient sorting network search via local mutation on 18 wires."""
import time
import random
import numpy as np

N = 18
DEADLINE = 55.0  # seconds, safety margin before 60s kill

# Precompute all 2^N bit inputs
# Shape: (2^N, N)
_IDX = np.arange(1 << N, dtype=np.uint32)
_ALL_BITS = np.zeros((1 << N, N), dtype=np.uint8)
for k in range(N):
    _ALL_BITS[:, k] = (_IDX >> k) & 1

def _apply_network(net, bits):
    """Apply compare-exchange network to a bit matrix (n_inputs, N).
    Returns the output bit matrix."""
    out = bits.copy()
    for i, j in net:
        a = out[:, i]
        b = out[:, j]
        # min goes to i, max goes to j
        # min(a,b) = a & b, max(a,b) = a | b
        out[:, i] = a & b
        out[:, j] = a | b
    return out

def _is_sorted(bits):
    """Check if every row of bits is non-decreasing (sorted ascending).
    bits is (n_inputs, N) with 0/1 values.
    A row is sorted iff it is of form 0...0 1...1."""
    # Check: no row has a 1 at position p and a 0 at position q > p
    # Specifically, no adjacent pair (1, 0)
    # bits[:, :-1] > bits[:, 1:] is True where 1 is followed by 0
    return not np.any(bits[:, :-1] > bits[:, 1:])

def _verify(net):
    """Verify network sorts all 2^N Boolean inputs. Returns True if correct."""
    out = _apply_network(net, _ALL_BITS)
    return _is_sorted(out)

def _seed_network():
    """Odd-even transposition network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Start from seed
    best_net = _seed_network()
    best_len = len(best_net)
    
    # Verify seed (should be correct)
    if not _verify(best_net):
        return best_net
    
    # Local search
    # We will try to reduce the network size.
    # Strategies:
    # 1. Remove one random comparator.
    # 2. Remove two random comparators (if 1 fails often).
    # 3. Perturb (swap) if stuck.
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # Try single removal
        if len(best_net) > 1:
            idx = rng.randint(0, len(best_net) - 1)
            new_net = best_net[:idx] + best_net[idx+1:]
            if _verify(new_net):
                best_net = new_net
                improved = True
                continue
        
        # Try double removal (if single failed or didn't improve)
        # This is more expensive (verify a shorter net) but might succeed when single fails.
        if len(best_net) > 2 and rng.random() < 0.2: # 20% chance to try double removal
            i1 = rng.randint(0, len(best_net) - 1)
            i2 = rng.randint(0, len(best_net) - 1)
            if i1 != i2:
                # Ensure i1 < i2 for easy slicing
                if i1 > i2: i1, i2 = i2, i1
                new_net = best_net[:i1] + best_net[i1+1:i2] + best_net[i2+1:]
                if _verify(new_net):
                    best_net = new_net
                    improved = True
                    continue
        
        # If no improvement, try a perturbation (swap) to escape local minima
        # This doesn't reduce size, but might allow future reductions.
        if not improved and len(best_net) > 1 and rng.random() < 0.5:
            i1 = rng.randint(0, len(best_net) - 1)
            i2 = rng.randint(0, len(best_net) - 1)
            if i1 != i2:
                new_net = list(best_net)
                new_net[i1], new_net[i2] = new_net[i2], new_net[i1]
                if _verify(new_net):
                    best_net = new_net
                    # Don't set improved=True because size didn't change, 
                    # but we changed the structure.
        
    return best_net
# EVOLVE-BLOCK-END
