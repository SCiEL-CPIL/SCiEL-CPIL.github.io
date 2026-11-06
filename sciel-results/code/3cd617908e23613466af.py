"""Sorting network search for N=17 wires using local optimization."""
import time
import random
import itertools

N = 17
NUM_ZERO_ONE_INPUTS = 1 << N  # 131072

# Precompute all zero-one inputs as bitmasks for fast checking
# We represent an input as a tuple of 0s and 1s, or use bitmask operations

def _apply_comparator(state, i, j):
    """Apply compare-exchange: min to i, max to j. state is a list of 0/1."""
    si = state[i]
    sj = state[j]
    if si > sj:
        state[i] = sj
        state[j] = si

def _verify_network(comparators):
    """Verify network sorts all 2^N zero-one inputs. Returns True if valid."""
    # For speed, we can check all 2^N inputs
    # But 131072 * len(comparators) might be slow in pure Python
    # Let's use a smarter approach: track the set of possible sorted states
    # Actually, the zero-one principle says we just need to check all 2^N inputs.
    
    # Optimization: if the number of 1s is k, the sorted output is (0,...,0,1,...,1) with k ones at the end
    # We can group inputs by number of 1s.
    
    # Even better: use the fact that for a network to be correct, for every subset S of wires (input mask),
    # after applying the network, the output must have all 1s in the highest positions.
    
    # Let's just iterate. With ~70 comparators and 131072 inputs, that's ~9M operations.
    # In Python this might take a few seconds. Let's try to be efficient.
    
    # Alternative: use the "signal" approach.
    # For each input bitmask, simulate the network.
    
    # Precompute for each comparator (i,j) the bit operations
    # Actually, let's just do it directly but efficiently.
    
    for mask in range(1 << N):
        # Create state from mask
        state = [(mask >> i) & 1 for i in range(N)]
        
        # Apply all comparators
        for i, j in comparators:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        
        # Check if sorted: state should be non-decreasing
        # For zero-one, this means all 0s come before all 1s
        # i.e., once we see a 1, all subsequent must be 1
        seen_one = False
        for s in state:
            if seen_one and s == 0:
                return False
            if s == 1:
                seen_one = True
    
    return True

def _verify_network_fast(comparators):
    """Faster verification using bitmask operations."""
    # For each input mask, we track which wires have 1s.
    # A compare-exchange (i,j) on the set of 1-wires:
    # If i has 1 and j has 0, swap: remove i, add j
    # If i has 0 and j has 1, no change (min=0 goes to i, max=1 stays at j)
    # If both 1 or both 0, no change
    
    # So the state is just a bitmask of which wires have 1s.
    # Compare (i,j): if bit i is 1 and bit j is 0, then new mask = mask with bit i cleared and bit j set.
    
    for mask in range(1 << N):
        m = mask
        for i, j in comparators:
            # If i has 1 and j has 0
            if (m >> i) & 1 and not ((m >> j) & 1):
                m = (m & ~(1 << i)) | (1 << j)
        # Check if sorted: all 1s are in the highest positions
        # The number of 1s in m should equal the number of 1s in mask
        # And m should be of the form: lower bits 0, upper bits 1
        k = bin(m).count('1')
        # The sorted form with k ones is: bits N-k through N-1 are 1, rest 0
        expected = ((1 << k) - 1) << (N - k) if k > 0 else 0
        if m != expected:
            return False
    return True

def _depth_of_network(comparators):
    """Compute the parallel depth of the network."""
    if not comparators:
        return 0
    # depth[i] = the latest depth at which wire i is used
    depth = [0] * N
    for i, j in comparators:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s safety margin
    
    # Start with odd-even transposition network
    def make_oe_network():
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    current = make_oe_network()
    current_count = len(current)
    
    # Verify initial
    if not _verify_network_fast(current):
        raise ValueError("Initial network is invalid")
    
    best = current[:]
    best_count = current_count
    best_depth = _depth_of_network(best)
    
    # Local search: try to remove comparators
    random.seed(42)
    
    def try_remove(comps):
        """Try to remove each comparator and check validity."""
        n = len(comps)
        # Try removing each one
        removals = list(range(n))
        random.shuffle(removals)
        
        for idx in removals:
            if time.time() > deadline:
                break
            new_comps = comps[:idx] + comps[idx+1:]
            if _verify_network_fast(new_comps):
                return new_comps
        return None
    
    def try_swap(comps):
        """Try swapping two adjacent comparators."""
        n = len(comps)
        if n < 2:
            return None
        # Try all adjacent swaps (limited for time)
        swaps = list(range(n - 1))
        random.shuffle(swaps)
        
        for idx in swaps:
            if time.time() > deadline:
                break
            new_comps = comps[:]
            new_comps[idx], new_comps[idx+1] = new_comps[idx+1], new_comps[idx]
            if _verify_network_fast(new_comps):
                return new_comps
        return None
    
    def try_replace(comps):
        """Try replacing a comparator with a different one."""
        n = len(comps)
        if n == 0:
            return None
        
        # Pick a random comparator to replace
        idx = random.randint(0, n - 1)
        old_i, old_j = comps[idx]
        
        # Generate random alternative
        attempts = 0
        while attempts < 50:
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i < new_j and (new_i, new_j) != (old_i, old_j):
                new_comps = comps[:]
                new_comps[idx] = (new_i, new_j)
                if _verify_network_fast(new_comps):
                    return new_comps
            attempts += 1
        
        return None
    
    # Iterative local search
    iterations = 0
    stuck_count = 0
    
    while time.time() < deadline - 5:  # Leave 5s buffer
        iterations += 1
        improved = False
        
        # Try removals first (most impactful)
        if len(current) > 63:  # Don't go below proven lower bound
            new_comps = try_remove(current)
            if new_comps is not None:
                current = new_comps
                current_count = len(current)
                improved = True
                stuck_count = 0
                if current_count < best_count:
                    best = current[:]
                    best_count = current_count
                    best_depth = _depth_of_network(best)
        
        if not improved:
            # Try swaps
            new_comps = try_swap(current)
            if new_comps is not None:
                current = new_comps
                improved = True
                stuck_count = 0
        
        if not improved:
            # Try replacements
            new_comps = try_replace(current)
            if new_comps is not None:
                current = new_comps
                improved = True
                stuck_count = 0
        
        if not improved:
            stuck_count += 1
            # If stuck for a while, try a more aggressive move
            if stuck_count > 10:
                # Try removing multiple comparators at once
                if len(current) > 65:
                    n = len(current)
                    # Try removing 2 random comparators
                    for _ in range(20):
                        if time.time() > deadline - 5:
                            break
                        idx1, idx2 = random.sample(range(n), 2)
                        idx1, idx2 = min(idx1, idx2), max(idx1, idx2)
                        new_comps = current[:idx1] + current[idx1+1:idx2] + current[idx2+1:]
                        if _verify_network_fast(new_comps):
                            current = new_comps
                            current_count = len(current)
                            stuck_count = 0
                            if current_count < best_count:
                                best = current[:]
                                best_count = current_count
                                best_depth = _depth_of_network(best)
                            break
        
        # Periodically restart from best
        if iterations % 50 == 0 and len(current) > best_count + 2:
            current = best[:]
            current_count = best_count
            stuck_count = 0
    
    # Final cleanup: try to remove as many as possible
    if time.time() < deadline - 2:
        while len(current) > 63:
            new_comps = try_remove(current)
            if new_comps is None:
                break
            current = new_comps
            current_count = len(current)
            if current_count < best_count:
                best = current[:]
                best_count = current_count
    
    return best
# EVOLVE-BLOCK-END
