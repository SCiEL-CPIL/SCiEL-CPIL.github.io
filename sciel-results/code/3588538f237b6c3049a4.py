"""Size-efficient sorting network search via local mutation on 18 wires."""
import time
import random
import numpy as np

N = 18
DEADLINE = 85.0  # seconds, safety margin before 90s kill

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
    # For each row, check that once a 1 appears, all subsequent are 1
    # i.e., no pattern 1 followed by 0
    # diff along axis 1: if any transition from 1 to 0, not sorted
    # Actually: sorted means all 0s come before all 1s.
    # Check: for each position k from 1 to N-1, if bits[:,k] == 0 then bits[:,k-1] must be 0
    # Equivalently: no row has a 1 at position p and a 0 at position q > p
    # Simplest: for each adjacent pair, if bits[:,k]==1 and bits[:,k+1]==0 -> unsorted
    for k in range(N - 1):
        if np.any((bits[:, k] == 1) & (bits[:, k + 1] == 0)):
            return False
    return True

def _verify(net):
    """Verify network sorts all 2^N Boolean inputs. Returns True if correct."""
    # Generate all 2^N inputs as a bit matrix
    n_inputs = 1 << N
    # Create indices 0..2^N-1 and extract bits
    idx = np.arange(n_inputs, dtype=np.uint32)
    bits = np.zeros((n_inputs, N), dtype=np.uint8)
    for k in range(N):
        bits[:, k] = (idx >> k) & 1
    out = _apply_network(net, bits)
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
        # Fallback: just return seed as-is
        return best_net
    
    # Local search: try mutations
    step = 0
    while time.time() - start_time < DEADLINE:
        step += 1
        # Choose mutation type
        mut_type = rng.random()
        
        new_net = list(best_net)
        
        if mut_type < 0.5:
            # Try removing a random comparator
            if len(new_net) <= 1:
                continue
            idx = rng.randint(0, len(new_net) - 1)
            removed = new_net.pop(idx)
            # Check if still correct
            if _verify(new_net):
                best_net = new_net
                best_len = len(best_net)
            else:
                new_net.insert(idx, removed)
                continue
        elif mut_type < 0.7:
            # Try inserting a random comparator at a random position
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(new_net))
            new_net.insert(pos, (i, j))
            if _verify(new_net):
                best_net = new_net
                best_len = len(best_net)
            # Don't need to revert since we only keep if better (but here it's same length)
            # Actually inserting makes it longer, so only keep if it enables future removals?
            # No - we want fewer comparators. Inserting alone doesn't help unless it allows
            # later removals. Skip insertions for now since they increase count.
            new_net = list(best_net)
        elif mut_type < 0.85:
            # Try swapping two adjacent comparators
            if len(new_net) < 2:
                continue
            idx = rng.randint(0, len(new_net) - 2)
            new_net[idx], new_net[idx + 1] = new_net[idx + 1], new_net[idx]
            if _verify(new_net):
                best_net = new_net
            else:
                new_net[idx], new_net[idx + 1] = new_net[idx + 1], new_net[idx]
                continue
        else:
            # Try moving a comparator to a different position
            if len(new_net) < 2:
                continue
            idx = rng.randint(0, len(new_net) - 1)
            new_idx = rng.randint(0, len(new_net) - 1)
            if idx == new_idx:
                continue
            comp = new_net.pop(idx)
            new_net.insert(new_idx, comp)
            if _verify(new_net):
                best_net = new_net
            else:
                # Revert
                new_net.pop(new_idx)
                new_net.insert(idx, comp)
                continue
        
        # Safety check
        if time.time() - start_time > DEADLINE - 2:
            break
    
    return best_net
# EVOLVE-BLOCK-END
