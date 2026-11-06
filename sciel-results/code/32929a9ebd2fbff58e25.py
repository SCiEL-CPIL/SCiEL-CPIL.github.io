"""Size-efficient sorting networks on 18 wires."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill

    # Precompute all 2^N boolean inputs as a bit-packed matrix for speed
    # Each input is N bits; we store them in a uint64 array (N=18 fits easily)
    total_inputs = 1 << N
    # inputs[i] is a uint64 where bit k corresponds to wire k
    inputs = np.arange(total_inputs, dtype=np.uint64)
    # We only care about the low N bits, which arange already gives us

    # Precompute sorted check: for a uint64 value v, it's sorted if
    # all 1s are in the low positions (i.e., v == (1<<popcount)-1)
    # Actually for our network, sorted means wires 0..k-1 are 0 and k..N-1 are 1
    # So sorted output for k ones is: bits k..N-1 set, bits 0..k-1 clear
    # That is: v == ((1 << (N - k)) - 1) << k  where k = number of zeros
    # Equivalently: v == (~((1 << k) - 1)) & ((1 << N) - 1)
    # Simpler: sorted iff (v & ((1 << N) - 1)) has the form 000...0111...1
    # i.e., v & (v + 1) == 0 when masked, but that's not quite right either.
    # Let's just check: for each possible count c of ones (0..N),
    # the sorted value is ((1 << (N - c)) - 1) << c
    # We can precompute a lookup: for each possible 18-bit value, is it sorted?
    # 2^18 = 262144 entries, each 1 byte -> 256 KB, fine.
    sorted_lookup = np.zeros(1 << N, dtype=np.bool_)
    for c in range(N + 1):  # c = number of zeros
        val = ((1 << (N - c)) - 1) << c
        if val < (1 << N):
            sorted_lookup[val] = True

    def apply_comparator(vals, i, j):
        """Apply compare-exchange on wires i,j to array of uint64 values.
        min goes to wire i, max to wire j.
        """
        mask_i = (1 << i)
        mask_j = (1 << j)
        # Extract bits
        bit_i = (vals >> i) & 1
        bit_j = (vals >> j) & 1
        # min = bit_i & bit_j, max = bit_i | bit_j
        min_bit = bit_i & bit_j
        max_bit = bit_i | bit_j
        # Clear both bits
        vals = vals & ~(mask_i | mask_j)
        # Set new values
        vals = vals | (min_bit << i) | (max_bit << j)
        return vals

    def verify_full(net):
        """Verify network against all inputs. Returns True if all sorted."""
        state = inputs.copy()
        for i, j in net:
            state = apply_comparator(state, i, j)
        # Check all are sorted
        return np.all(sorted_lookup[state & ((1 << N) - 1)])

    def verify_subset(net, subset_indices):
        """Verify against a subset of inputs."""
        state = inputs[subset_indices].copy()
        for i, j in net:
            state = apply_comparator(state, i, j)
        return np.all(sorted_lookup[state & ((1 << N) - 1)])

    def find_violations(net, subset_indices):
        """Return indices (within subset) of inputs that are NOT sorted after applying net."""
        state = inputs[subset_indices].copy()
        for i, j in net:
            state = apply_comparator(state, i, j)
        sorted_mask = sorted_lookup[state & ((1 << N) - 1)]
        return np.where(~sorted_mask)[0]

    def _odd_even_network():
        """Standard odd-even transposition network."""
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    # Start with odd-even transposition (153 comparators)
    best_network = _odd_even_network()
    best_count = len(best_network)

    if not verify_full(best_network):
        return best_network

    # Use a random subset for fast checking during search
    rng_init = np.random.RandomState(42)
    subset_size = min(4096, total_inputs)
    subset_indices = rng_init.choice(total_inputs, size=subset_size, replace=False)

    def verify_fast(net):
        """Fast verification on subset."""
        return verify_subset(net, subset_indices)

    def find_violations_fast(net):
        """Find violations on subset."""
        return find_violations(net, subset_indices)

    def systematic_removal(net, time_limit):
        """Try removing each comparator in order. Return best result."""
        best = list(net)
        best_count = len(net)
        t0 = time.time()
        
        for idx in range(len(net)):
            if time.time() > time_limit:
                break
            
            candidate = list(net)
            candidate.pop(idx)
            
            if verify_fast(candidate):
                if verify_full(candidate):
                    best = candidate
                    best_count = len(candidate)
                    # Continue trying to remove more from the new best
                    net = list(best)
                    idx = 0  # restart from beginning
                    continue
            # If fast check fails, skip to next idx
        
        return best, best_count

    def error_guided_repair(net, violations_subset, time_limit):
        """Given a network and its violations on subset, try to repair by
        inserting comparators that fix the violations.
        """
        if len(violations_subset) == 0:
            return net, len(net)
        
        t0 = time.time()
        current = list(net)
        current_count = len(current)
        
        # Get the violating inputs
        violating_inputs = inputs[subset_indices[violations_subset]]
        
        # For each violating input, determine which wire pairs are "out of order"
        # i.e., wire i has 1 and wire j has 0 with i < j
        # A comparator (i, j) with i < j would fix this if applied
        # But we need to find comparators that, when inserted, fix the issue
        
        # Strategy: for each violating input, find all (i, j) pairs where
        # wire i = 1 and wire j = 0, i < j. These are the "inversions".
        # A comparator (i, j) fixes one inversion but may create others.
        # We'll try adding comparators that fix the most inversions.
        
        # Count how many violating inputs have each (i, j) inversion
        inversion_count = np.zeros((N, N), dtype=np.int32)
        for val in violating_inputs:
            for i in range(N):
                if (val >> i) & 1:
                    for j in range(i + 1, N):
                        if not ((val >> j) & 1):
                            inversion_count[i, j] += 1
        
        # Get top candidate comparators
        flat_counts = inversion_count.flatten()
        # Only consider i < j
        candidates = []
        for i in range(N):
            for j in range(i + 1, N):
                if inversion_count[i, j] > 0:
                    candidates.append((inversion_count[i, j], i, j))
        
        candidates.sort(reverse=True)
        
        # Try inserting top candidates at various positions
        tried = set()
        for count, i, j in candidates[:20]:
            if time.time() > time_limit:
                break
            
            for pos in range(len(current) + 1):
                if (i, j, pos) in tried:
                    continue
                tried.add((i, j, pos))
                
                candidate = list(current)
                candidate.insert(pos, (i, j))
                
                if verify_fast(candidate):
                    if verify_full(candidate):
                        return candidate, len(candidate)
        
        return current, current_count

    def local_search_remove_repair(start_net, time_budget):
        """Local search: alternate between systematic removal and error-guided repair."""
        current = list(start_net)
        current_count = len(current)
        best_local = list(current)
        best_local_count = current_count
        
        t0 = time.time()
        
        while time.time() < t0 + time_budget:
            # Phase 1: systematic removal
            improved = False
            for _ in range(5):  # up to 5 removal passes
                if time.time() > t0 + time_budget:
                    break
                new_net, new_count = systematic_removal(current, time.time() + 2.0)
                if new_count < current_count:
                    current = new_net
                    current_count = new_count
                    improved = True
                    if current_count < best_local_count:
                        best_local = list(current)
                        best_local_count = current_count
                else:
                    break
            
            if time.time() > t0 + time_budget:
                break
            
            # Phase 2: if stuck, try error-guided repair to escape
            if not improved:
                violations = find_violations_fast(current)
                if len(violations) > 0:
                    # Try to repair by adding a comparator
                    repaired, rep_count = error_guided_repair(
                        current, violations, time.time() + 3.0)
                    if rep_count <= current_count + 2:  # allow small growth
                        # Check if this enables further removals
                        new_net, new_count = systematic_removal(
                            repaired, time.time() + 3.0)
                        if new_count < best_local_count:
                            current = new_net
                            current_count = new_count
                            best_local = list(current)
                            best_local_count = current_count
                        elif new_count < current_count:
                            current = new_net
                            current_count = new_count
        
        return best_local, best_local_count

    # Multi-restart search
    global_seed = 0
    while time.time() < deadline:
        time_left = deadline - time.time()
        if time_left < 3.0:
            break
        
        time_budget = min(8.0, time_left - 2.0)
        
        # Start from best known
        start_net = best_network
        
        cand_net, cand_count = local_search_remove_repair(start_net, time_budget)
        
        if cand_count < best_count:
            best_network = cand_net
            best_count = cand_count
        
        global_seed += 1

    return best_network

# EVOLVE-BLOCK-END
