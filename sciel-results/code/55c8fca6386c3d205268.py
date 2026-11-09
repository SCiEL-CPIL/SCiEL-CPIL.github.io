import time
import random
import numpy as np

N = 16
DEADLINE = time.time() + 80  # safety margin before 90s kill

# Bitmask helper: apply one comparator (i,j) to a 16-bit state
def apply_bitmask(state, i, j):
    if (state >> i) & 1 > (state >> j) & 1:
        state ^= (1 << i) | (1 << j)
    return state

# Precompute apply for speed on arrays
def apply_network_to_bitmask(state, network):
    for i, j in network:
        if (state >> i) & 1 > (state >> j) & 1:
            state ^= (1 << i) | (1 << j)
    return state

def is_sorted_bitmask(state):
    # Sorted ascending: all 1s before 0s => state is of form (1<<k)-1 for some k, 
    # i.e., bits 0..k-1 are 1, k..15 are 0. Actually min at i=0, so sorted means
    # state has form: lower bits 0, upper bits 1? No: wire 0 gets min.
    # After sorting ascending, wire 0 <= wire 1 <= ... <= wire 15.
    # In bitmask (bit i = wire i), sorted ascending means there's a cutoff:
    # bits 0..k-1 are 0, bits k..15 are 1. So state = ((1<<16)-1) ^ ((1<<k)-1)
    # i.e. state & (state+1) == 0 is NOT the condition. Let's just check.
    # A sorted ascending bitmask has the form: 0b000...0111...1 (zeros then ones).
    # That means (state + 1) has only the low bits set that were 0... 
    # Simplest: state must equal sum of 2^j for j in [k,15] for some k.
    # Equivalent: state | (state >> 1) ... no.
    # Check: if state is sorted, then for all i<j, if bit i=1 then bit j=1.
    # That's equivalent to: state & (state >> 1) == state? No.
    # If bit i is 1, then all higher bits must be 1. So state is a "suffix of ones".
    # state & ((state-1)) ... 
    # A suffix-of-ones number: state = 2^16-1 - (2^k - 1) = 2^16 - 2^k, or 2^16-1 (k=0).
    # So state+1 must be a power of 2 (or 2^16). i.e., (state+1) & state == 0.
    return (state + 1) & state == 0

# Fast full verification using numpy: represent all 65536 states
def verify_full(network):
    # Build a lookup of comparator actions is complex; do vectorized step.
    states = np.arange(65536, dtype=np.int64)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap = bi > bj
        # where swap, flip both bits
        mask = swap
        states = np.where(mask, states ^ ((1 << i) | (1 << j)), states)
    # check sorted: (states+1) & states == 0
    return bool(np.all(((states + 1) & states) == 0))

# Sampled verification (faster, probabilistic)
def verify_sample(network, n_samples=20000, rng=None):
    if rng is None:
        rng = random
    for _ in range(n_samples):
        state = rng.randrange(65536)
        s = state
        for i, j in network:
            if (s >> i) & 1 > (s >> j) & 1:
                s ^= (1 << i) | (1 << j)
        if (s + 1) & s != 0:
            return False
    return True

# Build an initial odd-even transposition network (120 comparators)
def initial_network():
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    start = time.time()
    net = initial_network()
    # Confirm initial is correct (it is known to be)
    best = list(net)
    best_len = len(best)
    
    rng = random.Random(12345)
    
    # Greedy removal: try to remove each comparator if network still sorts.
    # Iterate passes until no more removals or deadline.
    # Use sampled verification for speed, full verify at the end on best.
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        # shuffle order to avoid bias
        order = list(range(len(best)))
        rng.shuffle(order)
        for idx in order:
            if time.time() > DEADLINE - 5:
                break
            candidate = best[:idx] + best[idx+1:]
            # quick sample check
            if verify_sample(candidate, n_samples=15000, rng=rng):
                best = candidate
                best_len = len(best)
                improved = True
    
    # Final full verification of best
    if verify_full(best):
        return best
    else:
        # fallback to known-good initial
        return initial_network()
