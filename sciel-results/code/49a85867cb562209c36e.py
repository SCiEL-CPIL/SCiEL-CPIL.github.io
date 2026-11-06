"""Size-efficient sorting network for 16 wires using greedy + local search."""
import time
import random
import numpy as np

N = 16
N_BOOL = 1 << N  # 65536

# Precompute bit positions for speed
def _popcount(x):
    return bin(x).count('1')

def _apply_compare(state, i, j):
    """Apply compare-exchange (i,j) to a 16-bit state. min to i, max to j."""
    bi = (state >> i) & 1
    bj = (state >> j) & 1
    if bi > bj:
        # swap: set i to 0, j to 1
        state &= ~(1 << i)
        state |= (1 << j)
    return state

def _verify_fast(comparators):
    """Verify network sorts all 2^16 inputs using zero-one principle.
    Returns True if valid."""
    # For each input, simulate
    for inp in range(N_BOOL):
        state = inp
        for (i, j) in comparators:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi > bj:
                state &= ~(1 << i)
                state |= (1 << j)
        if state != (N_BOOL - 1):
            return False
    return True

def _verify_fast_numpy(comparators):
    """Verify using numpy vectorization over batches of inputs."""
    n_cmp = len(comparators)
    # Process in batches to keep memory reasonable
    batch_size = 4096
    all_pass = True
    for start in range(0, N_BOOL, batch_size):
        end = min(start + batch_size, N_BOOL)
        states = np.arange(start, end, dtype=np.uint32)
        for (i, j) in comparators:
            bi = (states >> i) & 1
            bj = (states >> j) & 1
            # where bi > bj: swap
            mask = bi > bj
            if np.any(mask):
                states = states & ~(1 << i)
                states = states | ((1 << j) & mask.astype(np.uint32))
        # All should equal 0xFFFF
        if not np.all(states == (N_BOOL - 1)):
            all_pass = False
            break
    return all_pass

def _depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    # Simple layering: assign each comparator to earliest possible layer
    # where both wires are free
    wire_last_layer = [-1] * N
    depth = 0
    for (i, j) in comparators:
        layer = max(wire_last_layer[i], wire_last_layer[j]) + 1
        wire_last_layer[i] = layer
        wire_last_layer[j] = layer
        depth = max(depth, layer)
    return depth

def _greedy_build():
    """Build a sorting network greedily: add comparators that fix the most inputs."""
    # Start with empty network
    comparators = []
    # Track which inputs are still unsorted
    unsorted = set(range(N_BOOL))
    
    if not unsorted:
        return comparators
    
    # Precompute: for each possible comparator (i,j), which inputs does it help?
    # We'll use a heuristic: try all pairs, pick the one that sorts the most remaining inputs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    max_iter = 200  # safety
    iteration = 0
    while unsorted and iteration < max_iter:
        iteration += 1
        best_pair = None
        best_fixed = 0
        # Sample a subset of pairs for speed
        random.shuffle(all_pairs)
        sample = all_pairs[:200]
        for (i, j) in sample:
            # Count how many unsorted inputs this comparator would fix
            fixed = 0
            for inp in unsorted:
                state = inp
                # Apply existing comparators
                for (ci, cj) in comparators:
                    bi = (state >> ci) & 1
                    bj = (state >> cj) & 1
                    if bi > bj:
                        state &= ~(1 << ci)
                        state |= (1 << cj)
                # Apply new comparator
                bi = (state >> i) & 1
                bj = (state >> j) & 1
                if bi > bj:
                    state &= ~(1 << i)
                    state |= (1 << j)
                if state == (N_BOOL - 1):
                    fixed += 1
            if fixed > best_fixed:
                best_fixed = fixed
                best_pair = (i, j)
        if best_pair is None or best_fixed == 0:
            # Try adding a random pair
            best_pair = random.choice(all_pairs)
        comparators.append(best_pair)
        # Recompute unsorted
        new_unsorted = set()
        for inp in unsorted:
            state = inp
            for (ci, cj) in comparators:
                bi = (state >> ci) & 1
                bj = (state >> cj) & 1
                if bi > bj:
                    state &= ~(1 << ci)
                    state |= (1 << cj)
            if state != (N_BOOL - 1):
                new_unsorted.add(inp)
        unsorted = new_unsorted
    
    return comparators

def _local_search(comparators, time_budget):
    """Try to remove or swap comparators to reduce count while maintaining validity."""
    start_time = time.time()
    comparators = list(comparators)
    
    # Try removing each comparator
    changed = True
    while changed and (time.time() - start_time) < time_budget * 0.5:
        changed = False
        n = len(comparators)
        for idx in range(n - 1, -1, -1):
            if time.time() - start_time > time_budget * 0.5:
                break
            # Try removing comparator at idx
            trial = comparators[:idx] + comparators[idx+1:]
            if _verify_fast(trial):
                comparators = trial
                changed = True
                break
    
    # Try swapping adjacent comparators
    changed = True
    while changed and (time.time() - start_time) < time_budget:
        changed = False
        n = len(comparators)
        for idx in range(n - 1):
            if time.time() - start_time > time_budget:
                break
            trial = comparators[:]
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _verify_fast(trial):
                comparators = trial
                changed = True
                break
    
    return comparators

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # 80s safety margin (90s kill, 10s buffer)
    
    # Start with odd-even transposition (known valid, 120 comparators)
    best = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    # Verify it's valid (should be)
    if not _verify_fast(best):
        # Shouldn't happen, but fallback
        return best
    
    # Try local search to reduce comparator count
    time_left = deadline - time.time()
    if time_left > 5:
        best = _local_search(best, min(time_left - 5, 60))
    
    # Try greedy construction as alternative
    time_left = deadline - time.time()
    if time_left > 10:
        random.seed(42)
        greedy = _greedy_build()
        if greedy and _verify_fast(greedy) and len(greedy) < len(best):
            best = greedy
            # Local search on greedy result
            time_left = deadline - time.time()
            if time_left > 5:
                best = _local_search(best, min(time_left - 5, 30))
    
    # Final local search pass with remaining time
    time_left = deadline - time.time()
    if time_left > 3:
        best = _local_search(best, max(1, time_left - 3))
    
    return best
# EVOLVE-BLOCK-END
