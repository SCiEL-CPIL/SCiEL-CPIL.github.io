import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55  # Safety margin before 60s kill

# Precompute full matrix of all 2^N boolean inputs
all_masks = np.arange(1 << N, dtype=np.int32)
W0_full = np.zeros((1 << N, N), dtype=np.int8)
for i in range(N):
    W0_full[:, i] = (all_masks >> i) & 1

# Precompute subset matrix for fast rejection during search
# We pick a diverse set of inputs, focusing on the "hard" middle weights
# and some random ones.
# Weight N/2 = 9 is the hardest.
# Let's pick all inputs with weight 9? That's C(18,9) = 48620. Too many for "fast" check if we want speed.
# Let's pick 2048 random inputs and 1024 inputs with weight 9.
random.seed(42)
subset_indices = np.random.choice(1 << N, 2048, replace=False)
# Add some structured ones
mid_indices = [i for i in range(1 << N) if bin(i).count('1') == 9][:1024]
subset_indices = np.concatenate([subset_indices, np.array(mid_indices)])
W0_subset = W0_full[subset_indices]

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net, use_subset=False):
    """
    Verify that the comparator network sorts all boolean inputs.
    use_subset=True checks only W0_subset (faster, for search).
    use_subset=False checks W0_full (slow, for final validation).
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    if use_subset:
        W = W0_subset.copy()
    else:
        W = W0_full.copy()
        
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        # min = AND, max = OR for bits
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    # Check if sorted: W[:, k] <= W[:, k+1] for all k
    # Equivalent to: no W[:, k] > W[:, k+1]
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
        
    return True

def _batcher_merge(a, b):
    """
    Generate Batcher's merge network for two sorted sequences of length a and b.
    Returns a list of comparators (i, j) assuming input 0..a-1 is seq A and a..a+b-1 is seq B.
    """
    if a == 0:
        return []
    if b == 0:
        return []
    if a == 1 and b == 1:
        return [(0, 1)]
    
    # Odd-even merge
    # Compare elements at even positions in A with even positions in B (offset)
    # and odd with odd.
    # Standard Batcher Odd-Merge:
    # 1. Recursively merge the even-indexed elements of the two sequences.
    # 2. Recursively merge the odd-indexed elements of the two sequences.
    # 3. Compare the results to interleave them.
    
    # This implementation is complex to map directly to global indices without tracking.
    # It's easier to generate the full network structure recursively.
    pass

def _batcher_sort(n):
    """
    Generate Batcher's Odd-Merge Sort network for n wires.
    Returns a list of (i, j) pairs.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    
    # Recursively sort the two halves
    net1 = _batcher_sort(n1)
    net2 = _batcher_sort(n2)
    
    # Shift net2 indices by n1
    net2 = [(i + n1, j + n1) for i, j in net2]
    
    # Merge the two sorted halves
    # Batcher's merge of two sorted lists of size n1 and n2
    # We need a merge function that works on global indices.
    # Let's implement a specific merge for the sorted halves.
    # The merge network for two sorted sequences of length m and k (m<=k)
    # consists of a sequence of comparators.
    
    # For simplicity and correctness, we can use a known small merge network generator
    # or just use a generic merge that might not be optimal but is correct.
    # However, Batcher's merge is specific.
    
    # Let's implement the odd-even merge step explicitly for the two halves.
    # This is tricky to do in-place on global indices without a recursive structure that returns the full network.
    
    # Alternative: Use a simpler, robust baseline.
    # Since implementing Batcher's Odd-Merge correctly from scratch in a short time is error-prone,
    # and we have time, let's stick to a very strong heuristic search starting from a decent seed.
    # A "Bitonic" sort is easier to implement?
    # Or just use the Odd-Even Transposition and improve it heavily?
    # The current program starts with OET (153).
    # Let's try to implement a simple recursive merge that is correct, even if not minimal.
    
    # Actually, let's just use the OET seed but optimize the search much better.
    # The main gain comes from the search algorithm and verification speed.
    return None # Placeholder

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Greedy removal
    # This is very effective for reducing the size of OET.
    changed = True
    while changed:
        if time.time() > deadline - 10:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10:
                return best_net
            
            new_net = net[:idx] + net[idx+1:]
            
            # Fast check first
            if _is_sorted_network(new_net, use_subset=True):
                # Full check to be sure
                if _is_sorted_network(new_net, use_subset=False):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 10:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _is_sorted_network(new_net, use_subset=True):
                    if _is_sorted_network(new_net, use_subset=False):
                        net = new_net
                        best_net = net[:]
                        best_count = len(net)
                        changed = True
                        break
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 5:
        T = 1.0
        min_T = 0.05
        cooling = 0.999
        current_count = len(net)
        
        while time.time() < deadline - 2:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 1.0
                
            # Generate move
            r = random.random()
            if r < 0.3:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.6:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            elif r < 0.8:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            else:
                # Swap adjacent comparators
                if len(net) < 2:
                    continue
                idx = random.randint(0, len(net) - 2)
                new_net = net[:]
                new_net[idx], new_net[idx+1] = new_net[idx+1], new_net[idx]
            
            # Fast check
            if not _is_sorted_network(new_net, use_subset=True):
                continue
            
            # Full check
            if not _is_sorted_network(new_net, use_subset=False):
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    
            T *= cooling
    
    # Final validation
    if not _is_sorted_network(best_net, use_subset=False):
        best_net = _odd_even_transposition()
        
    return best_net
