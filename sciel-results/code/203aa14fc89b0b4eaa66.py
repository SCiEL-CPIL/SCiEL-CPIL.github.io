import numpy as np
import time
import random

N = 18

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = _start_time + 85.0  # Safety margin before 90s kill

def _apply_network(net, inp):
    """Apply compare-exchange network to input array.
    inp: 1D numpy array of shape (N,)
    Returns sorted array if net is a valid sorting network.
    """
    w = inp.copy()
    for i, j in net:
        a, b = w[i], w[j]
        if a > b:
            w[i], w[j] = b, a
    return w

def _verify_network(net, sample_size=10000):
    """Verify network on random Boolean inputs. Returns True if sorted on all samples."""
    rng = np.random.RandomState(42)
    for _ in range(sample_size):
        inp = (rng.rand(N) < 0.5).astype(np.uint8)
        out = _apply_network(net, inp)
        # Check if sorted
        if np.any(np.diff(out) < 0):
            return False
    return True

def _verify_network_full(net):
    """Full verification over all 2^18 Boolean inputs."""
    w = np.zeros(N, dtype=np.uint8)
    for mask in range(1 << N):
        w[0] = mask & 1
        for k in range(1, N):
            w[k] = (mask >> k) & 1
        out = _apply_network(net, w)
        if np.any(np.diff(out) < 0):
            return False
    return True

def _count_comparators(net):
    return len(net)

def _make_odd_even(N_wires):
    """Standard odd-even transposition network."""
    comps = []
    for phase in range(N_wires):
        for i in range(phase % 2, N_wires - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_remove(net, idx):
    """Try removing comparator at index idx; return new net if still valid."""
    new_net = net[:idx] + net[idx+1:]
    if _verify_network(new_net, sample_size=5000):
        return new_net
    return None

def _try_swap(net, i, j):
    """Try swapping comparators at positions i and j."""
    new_net = list(net)
    new_net[i], new_net[j] = new_net[j], new_net[i]
    if _verify_network(new_net, sample_size=5000):
        return new_net
    return None

def _try_move(net, src, dst):
    """Try moving comparator from src to dst position."""
    if src == dst:
        return None
    comp = net[src]
    new_net = list(net)
    del new_net[src]
    new_net.insert(dst, comp)
    if _verify_network(new_net, sample_size=5000):
        return new_net
    return None

def _local_search(net, max_iters=2000):
    """Supercharged local search: try remove, swap, move operations."""
    best = list(net)
    rng = random.Random(12345)
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        n = len(best)
        # Try remove
        if n > 1:
            idx = rng.randint(0, n - 1)
            candidate = _try_remove(best, idx)
            if candidate is not None:
                best = candidate
                continue
        # Try swap
        if n > 2:
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            candidate = _try_swap(best, i, j)
            if candidate is not None:
                best = candidate
                continue
        # Try move
        if n > 2:
            src = rng.randint(0, n - 1)
            dst = rng.randint(0, n)
            candidate = _try_move(best, src, dst)
            if candidate is not None:
                best = candidate
                continue
    return best

def _simulated_annealing(net, max_iters=3000):
    """SA with remove/swap/move moves."""
    best = list(net)
    current = list(net)
    T = 1.0
    cooling = 0.999
    rng = random.Random(99999)
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        n = len(current)
        move_type = rng.random()
        if move_type < 0.4 and n > 1:
            idx = rng.randint(0, n - 1)
            candidate = _try_remove(current, idx)
        elif move_type < 0.7 and n > 2:
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            candidate = _try_swap(current, i, j)
        else:
            if n > 2:
                src = rng.randint(0, n - 1)
                dst = rng.randint(0, n)
                candidate = _try_move(current, src, dst)
            else:
                candidate = None
        
        if candidate is not None:
            new_len = len(candidate)
            delta = new_len - len(current)
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-6)):
                current = candidate
                if new_len < len(best):
                    best = candidate
        T *= cooling
    return best

def _add_random_comparator(net, rng):
    """Add a random comparator to potentially enable further removals."""
    i = rng.randint(0, N - 2)
    j = rng.randint(i + 1, N - 1)
    new_net = list(net) + [(i, j)]
    if _verify_network(new_net, sample_size=2000):
        return new_net
    return None

def _kick_search(net, max_kicks=20):
    """Kick out of local minima by adding then removing comparators."""
    best = list(net)
    rng = random.Random(777)
    for k in range(max_kicks):
        if time.time() > _DEADLINE:
            break
        kicked = _add_random_comparator(best, rng)
        if kicked is not None:
            # Now do local search from the kicked state
            result = _local_search(kicked, max_iters=200)
            if len(result) < len(best):
                best = result
    return best

def construct():
    global _start_time, _DEADLINE
    _start_time = time.time()
    _DEADLINE = _start_time + 85.0
    
    # Start from odd-even transposition network
    net = _make_odd_even(N)
    best = list(net)
    
    # Phase 1: Aggressive local search to remove redundant comparators
    for iteration in range(10):
        if time.time() > _DEADLINE - 5:
            break
        result = _local_search(best, max_iters=500)
        if len(result) < len(best):
            best = result
        else:
            # Try kick search to escape local minimum
            result = _kick_search(best, max_kicks=5)
            if len(result) < len(best):
                best = result
            else:
                break
    
    # Phase 2: Simulated annealing for global refinement
    if time.time() < _DEADLINE - 10:
        result = _simulated_annealing(best, max_iters=2000)
        if len(result) < len(best):
            best = result
    
    # Phase 3: Final local search polish
    if time.time() < _DEADLINE - 3:
        result = _local_search(best, max_iters=500)
        if len(result) < len(best):
            best = result
    
    # Final full verification (only if we have time)
    if time.time() < _DEADLINE - 1:
        if not _verify_network_full(best):
            # Fallback: verify the original seed
            if _verify_network_full(_make_odd_even(N)):
                best = _make_odd_even(N)
    
    return best

# EVOLVE-BLOCK-END
