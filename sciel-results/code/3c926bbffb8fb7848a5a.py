"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Generate all 2^18 Boolean inputs as a (262144, 18) uint8 array
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1
    
    # State: each wire's value for each input
    state = all_inputs.copy()
    
    def apply_comparator(s, i, j):
        """Apply compare-exchange on wires i and j. Returns new state."""
        new_s = s.copy()
        new_s[:, i] = np.minimum(s[:, i], s[:, j])
        new_s[:, j] = np.maximum(s[:, i], s[:, j])
        return new_s
    
    def is_sorted(s):
        """Check if all inputs are sorted (non-decreasing across wires)."""
        for i in range(N - 1):
            if np.any((s[:, i] == 1) & (s[:, i+1] == 0)):
                return False
        return True
    
    # Greedy search: repeatedly pick the comparator that most reduces total inversions
    network = []
    current_state = state.copy()
    
    # Precompute all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Precompute prefix and suffix sums for fast penalty calculation
    # For each input, prefix_ones[k] = number of 1s in wires 0..k-1
    # For each input, suffix_zeros[k] = number of 0s in wires k+1..N-1
    
    def compute_prefix_suffix(s):
        """Compute prefix count of 1s and suffix count of 0s for fast penalty lookup."""
        # prefix_ones[i] for input x = sum of s[x, 0:i]
        # suffix_zeros[j] for input x = sum of (1 - s[x, j+1:N])
        
        # Use cumulative sums along axis=1
        # prefix_ones: shape (num_inputs, N+1), prefix_ones[x, k] = sum(s[x, :k])
        prefix_ones = np.zeros((s.shape[0], N + 1), dtype=np.int32)
        for k in range(N):
            prefix_ones[:, k+1] = prefix_ones[:, k] + s[:, k]
        
        # suffix_zeros: shape (num_inputs, N+1), suffix_zeros[x, k] = sum(1-s[x, k:])
        suffix_zeros = np.zeros((s.shape[0], N + 1), dtype=np.int32)
        for k in range(N-1, -1, -1):
            suffix_zeros[:, k] = suffix_zeros[:, k+1] + (1 - s[:, k])
        
        return prefix_ones, suffix_zeros
    
    while True:
        if time.time() > DEADLINE:
            break
        
        if is_sorted(current_state):
            break
        
        # Compute prefix/suffix for fast penalty calculation
        prefix_ones, suffix_zeros = compute_prefix_suffix(current_state)
        
        # Find the comparator that maximally reduces total inversions
        best_gain = -1
        best_comp = None
        
        for (i, j) in all_comparators:
            # Only consider inputs where old_i=1 and old_j=0
            mask = (current_state[:, i] == 1) & (current_state[:, j] == 0)
            n_affected = np.sum(mask)
            if n_affected == 0:
                continue
            
            # Base gain: fix (i,j) pair + fix all intermediate pairs
            base_gain = 1 + (j - i - 1)  # per affected input
            
            # Penalty left: for k < i, count of old_k=1 among affected inputs
            # prefix_ones[mask, i] gives sum of 1s in wires 0..i-1 for each affected input
            penalty_left = np.sum(prefix_ones[mask, i])
            
            # Penalty right: for k > j, count of old_k=0 among affected inputs
            # suffix_zeros[mask, j+1] gives sum of 0s in wires j+1..N-1 for each affected input
            penalty_right = np.sum(suffix_zeros[mask, j+1])
            
            total_gain = n_affected * base_gain - penalty_left - penalty_right
            
            if total_gain > best_gain:
                best_gain = total_gain
                best_comp = (i, j)
        
        if best_comp is None or best_gain <= 0:
            # No progress possible with greedy; break
            break
        
        # Apply the best comparator
        i, j = best_comp
        current_state = apply_comparator(current_state, i, j)
        network.append((i, j))
    
    # If we ran out of time or got stuck, return what we have
    # But we need a valid network! Let's check if current state is sorted.
    # If not, we should return the seed network to be safe.
    
    if not is_sorted(current_state):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return network
# EVOLVE-BLOCK-END
