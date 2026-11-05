"""Size-efficient sorting networks on 18 wires.
Uses bit-parallel verification over all 2^18 Boolean inputs to enable
greedy removal of redundant comparators from a seed network.
"""
import time
import numpy as np

N = 18
TOTAL_BOOL = 1 << N  # 262144

# Deadline guard: return best result well before the 90s wall-clock limit.
DEADLINE = 80.0  # seconds (10s safety margin)

# EVOLVE-BLOCK-START
def construct():
    """Construct a sorting network for N=18 wires with as few comparators as possible."""
    t0 = time.time()
    deadline = t0 + DEADLINE

    # --- Seed network: odd-even transposition (153 comparators for N=18) ---
    seed = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    # --- Bit-parallel verification helpers ---
    # Represent all 2^N boolean inputs as an array of shape (N, 2^N) of uint8
    # where arr[w, x] is the bit on wire w for input x.
    # Actually, for speed we pack: each input is a 18-bit integer 0..2^18-1.
    # We can process all inputs at once using numpy vectorization.

    def make_initial_state():
        """Create array of shape (N, 2^N) where state[w, x] = bit w of x."""
        # x ranges from 0 to 2^N - 1
        xs = np.arange(TOTAL_BOOL, dtype=np.uint32)
        state = np.zeros((N, TOTAL_BOOL), dtype=np.uint8)
        for w in range(N):
            state[w] = ((xs >> w) & 1).astype(np.uint8)
        return state

    def apply_comparator(state, i, j):
        """Apply compare-exchange on wires i, j: min goes to i, max goes to j."""
        a = state[i].copy()
        b = state[j].copy()
        # min(a,b) -> wire i, max(a,b) -> wire j
        state[i] = np.minimum(a, b)
        state[j] = np.maximum(a, b)
        return state

    def is_sorted(state):
        """Check if state is sorted: for all x, state[0,x] >= state[1,x] >= ... >= state[N-1,x].
        Actually sorting networks sort in non-decreasing order: wire 0 has min, wire N-1 has max.
        So we need state[0,x] <= state[1,x] <= ... <= state[N-1,x] for all x.
        """
        for w in range(N - 1):
            if np.any(state[w] > state[w + 1]):
                return False
        return True

    def verify_network(comparators):
        """Verify that the given list of comparators sorts all boolean inputs."""
        state = make_initial_state()
        for (i, j) in comparators:
            apply_comparator(state, i, j)
        return is_sorted(state)

    # --- Greedy removal: try to remove each comparator if the network still sorts ---
    # Start with the seed network and try to remove comparators one by one.
    # For efficiency, we process in reverse order (removing from the end first
    # may be faster since later comparators are more likely redundant).
    
    net = list(seed)
    
    # First verify the seed is correct (sanity check)
    # This takes some time but is necessary.
    
    # To speed up greedy removal, we can use a faster verification approach.
    # Key insight: we only need to check if the network sorts ALL boolean inputs.
    # With bit-parallel, each comparator application is O(N * 2^N) which is ~4.7M ops.
    # For 153 comparators, that's ~720M ops per full verification.
    # In numpy with uint8, this should be fast enough.
    
    # Let's estimate: numpy can do ~100M+ simple ops per second on modern hardware.
    # 720M ops might take ~1-2 seconds per verification.
    # We have 80s, so we can do maybe 40-80 verifications.
    # That's not enough to try all 153 removals individually.
    
    # Better approach: use a smarter greedy strategy.
    # 1. Try removing comparators in a specific order.
    # 2. Use early termination in verification.
    
    # Actually, let's use a different strategy:
    # - Start with the seed
    # - Try to remove each comparator (in some order) and check if still valid
    # - If valid, keep it removed; if not, keep it
    # - Repeat until no more can be removed
    
    # For speed, let's use a more efficient verification:
    # Instead of checking all 2^N inputs, we can use the zero-principle:
    # A comparator network sorts all inputs iff it sorts all 0-1 inputs.
    # That's what we're doing. But we can speed up by:
    # 1. Using uint8 and numpy vectorization (already doing)
    # 2. Early exit in is_sorted
    
    # Let's implement a faster verification that processes comparators and
    # checks sorting at the end.
    
    def fast_verify(comparators):
        """Fast verification using bit-parallel numpy operations."""
        # Create initial state: all 2^N boolean inputs
        # state shape: (N, 2^N), uint8
        # We can create this efficiently:
        xs = np.arange(TOTAL_BOOL, dtype=np.uint32)
        state = np.empty((N, TOTAL_BOOL), dtype=np.uint8)
        for w in range(N):
            state[w] = ((xs >> w) & 1).astype(np.uint8)
        
        for (i, j) in comparators:
            # Compare-exchange: min to i, max to j
            a = state[i]
            b = state[j]
            state[i] = np.minimum(a, b)
            state[j] = np.maximum(a, b)
        
        # Check if sorted: state[w] <= state[w+1] for all w
        for w in range(N - 1):
            if np.any(state[w] > state[w + 1]):
                return False
        return True
    
    # Try greedy removal
    # Order: try removing from the end first (later comparators are often redundant)
    # Then try from the beginning, etc.
    
    removed = []
    i = len(net) - 1
    while i >= 0 and time.time() < deadline - 5.0:  # 5s margin for final verification
        # Try removing net[i]
        trial = net[:i] + net[i+1:]
        if fast_verify(trial):
            removed.append(i)
            net = trial
            # Don't decrement i since indices shifted; try same position again
            # Actually, after removal, the element at position i is now the old i+1
            # Let's just continue with i unchanged to try removing it too
        else:
            i -= 1
    
    # Second pass: try removing from the beginning
    i = 0
    while i < len(net) and time.time() < deadline - 5.0:
        trial = net[:i] + net[i+1:]
        if fast_verify(trial):
            net = trial
            # Don't increment i, try same position again
        else:
            i += 1
    
    # Third pass: another round from the end
    i = len(net) - 1
    while i >= 0 and time.time() < deadline - 5.0:
        trial = net[:i] + net[i+1:]
        if fast_verify(trial):
            net = trial
        else:
            i -= 1
    
    # Fourth pass: another round from the beginning
    i = 0
    while i < len(net) and time.time() < deadline - 5.0:
        trial = net[:i] + net[i+1:]
        if fast_verify(trial):
            net = trial
        else:
            i += 1
    
    # Final verification to make sure our network is valid
    if time.time() < deadline - 2.0:
        if not fast_verify(net):
            # Fallback to seed if something went wrong
            return seed
    
    return net
# EVOLVE-BLOCK-END
