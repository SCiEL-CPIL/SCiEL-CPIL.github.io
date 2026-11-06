import time
import random
import numpy as np
from itertools import combinations

N = 17
MAX_TIME = 58.0  # Safety margin for 60s limit

# Precompute valid sorted states for quick lookup
# A state is sorted if it is 0^a 1^b.
# Values: 0, 2^17-2^16, 2^17-2^15, ..., 2^17-1
_VALID_MASK = np.zeros(1 << N, dtype=bool)
for k in range(N + 1):
    if k == 0:
        val = 0
    else:
        val = (1 << N) - (1 << (N - k))
    _VALID_MASK[val] = True

def seed_network():
    """Odd-even transposition network (136 comparators)."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def verify_network(net):
    """
    Vectorized verification using NumPy.
    Checks if the network sorts all 2^17 boolean inputs.
    """
    if not net:
        return N <= 1

    states = np.arange(1 << N, dtype=np.uint32)
    
    for i, j in net:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        mask = (bi == 1) & (bj == 0)
        if np.any(mask):
            states[mask] ^= (1 << i) | (1 << j)
            
    return np.all(_VALID_MASK[states])

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Start with a known valid network (Seed)
    best_net = seed_network()
    best_count = len(best_net)
    
    # Strategy: Layer-by-Layer Backtracking (CSP/SAT-like)
    # We model the network as a sequence of layers.
    # Each layer contains a set of disjoint comparators.
    # We search for the minimal configuration of layers that sorts the input.
    
    # To make this tractable, we restrict the search space:
    # 1. We assume a fixed number of layers (or search over it).
    # 2. For each layer, we select a matching (set of disjoint edges) from a candidate set.
    # 3. We use backtracking: if a partial sequence of layers fails to sort
    #    a subset of "hard" inputs (or all inputs), we backtrack.
    
    # Optimization: Instead of checking all 2^17 inputs at every step,
    # we can check a random subset of "witness" inputs that are likely to fail
    # early. However, for correctness, we must eventually verify all.
    # A better approach: Use the "Zero-One Principle" on the *current state* of the wires.
    
    # Let's implement a "Greedy Layer Construction" with Backtracking.
    # 1. Start with empty layers.
    # 2. At each layer index L, try to add a set of comparators that reduces
    #    the "disorder" of the inputs.
    # 3. We define "disorder" as the number of inversions in the current state
    #    of the 17 wires for a sample of inputs.
    
    # To speed up, we precompute the effect of a layer on a small set of witness inputs.
    # Witness set: All inputs with exactly k ones for k in [1..16]? No, too many.
    # Let's use a random sample of 1024 inputs for heuristic guidance, 
    # and full verification only when a candidate network is complete.
    
    witness_states = np.random.choice(1 << N, size=1024, replace=False, dtype=np.uint32)
    
    def apply_layer(states, comparators):
        """Apply a list of disjoint comparators to a set of states."""
        for i, j in comparators:
            bi = (states >> i) & 1
            bj = (states >> j) & 1
            mask = (bi == 1) & (bj == 0)
            if np.any(mask):
                states[mask] ^= (1 << i) | (1 << j)
        return states

    def count_inversions(states):
        """Count total inversions in a set of states."""
        inv = 0
        # For each state, count inversions
        # Vectorized: for each pair (i, j), i < j, check if bit i is 1 and bit j is 0
        # Sum over all states
        # This is O(N^2 * num_states)
        # N=17, num_states=1024 -> 289 * 1024 ~ 300k ops. Fast.
        for i in range(N):
            for j in range(i + 1, N):
                bi = (states >> i) & 1
                bj = (states >> j) & 1
                inv += np.sum((bi == 1) & (bj == 0))
        return inv

    def search_layers(max_layers=20):
        """
        Backtracking search for a sequence of layers.
        Each layer is a set of disjoint comparators.
        """
        current_layers = []
        current_witness_states = witness_states.copy()
        
        # We want to minimize the number of comparators.
        # We will try to build a network layer by layer.
        # At each step, we choose a layer that minimizes the inversion count
        # of the witness states, subject to the constraint that the layer
        # is a matching (disjoint edges).
        
        # To avoid getting stuck in local minima, we use a limited beam search
        # or restarts.
        
        best_found = None
        
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Randomize the witness states slightly or the search order
            # to escape local minima
            curr_states = witness_states.copy()
            layers = []
            
            # Try to build a network of up to max_layers layers
            # We stop when curr_states are all sorted (on the witness set)
            # Then we verify on the full set.
            
            for layer_idx in range(max_layers):
                # Find the best matching to apply
                # We evaluate all possible matchings? No, too many.
                # We use a greedy heuristic: pick the comparator that reduces
                # inversions the most, then find a matching around it.
                
                # Simpler: Randomly generate a few matchings and pick the best.
                best_matching = []
                best_inv = count_inversions(curr_states)
                
                # Try 100 random matchings
                for _ in range(100):
                    # Generate a random matching
                    wires = list(range(N))
                    random.shuffle(wires)
                    matching = []
                    used = set()
                    for k in range(0, N - 1, 2):
                        if wires[k] in used or wires[k+1] in used:
                            continue
                        i, j = wires[k], wires[k+1]
                        if i > j: i, j = j, i
                        matching.append((i, j))
                        used.add(i)
                        used.add(j)
                    
                    if not matching:
                        continue
                        
                    new_states = apply_layer(curr_states, matching)
                    new_inv = count_inversions(new_states)
                    
                    if new_inv < best_inv:
                        best_inv = new_inv
                        best_matching = matching
                        if new_inv == 0:
                            break
                
                if not best_matching:
                    # No improvement found, break
                    break
                    
                layers.append(best_matching)
                curr_states = apply_layer(curr_states, best_matching)
                
                # Check if witness states are sorted
                if np.all(_VALID_MASK[curr_states]):
                    # Candidate found! Verify on full set
                    full_net = []
                    for layer in layers:
                        full_net.extend(layer)
                    
                    # Optional: Remove redundant comparators via greedy removal
                    # But first, check if it's valid
                    if verify_network(full_net):
                        if best_found is None or len(full_net) < len(best_found):
                            best_found = full_net
                    break
            
            if best_found is not None:
                break
                
        return best_found

    # Run the layer-based search
    found_net = search_layers()
    
    if found_net is not None and len(found_net) < best_count:
        best_net = found_net
        best_count = len(best_net)
    
    # If the layer-based search didn't beat the seed, try a quick local search
    # on the best found so far (or seed) to squeeze out a few more.
    # This is a fallback to ensure we don't lose to the seed if the new strategy fails.
    
    def local_search(net, time_limit=5.0):
        start = time.time()
        current = list(net)
        best = list(current)
        best_len = len(best)
        
        while time.time() - start < time_limit:
            improved = False
            # Try removing each comparator
            for i in range(len(current)):
                if time.time() - start > time_limit:
                    break
                candidate = current[:i] + current[i+1:]
                if verify_network(candidate):
                    current = candidate
                    improved = True
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(best)
                    break # Restart loop with new current
            
            if not improved:
                break
                
        return best

    # Apply local search to the best candidate
    if len(best_net) > 63: # Lower bound
        optimized_net = local_search(best_net, time_limit=10.0)
        if len(optimized_net) < len(best_net):
            best_net = optimized_net
            best_count = len(best_net)

    # Final safety check
    if not verify_network(best_net):
        best_net = seed_network()
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Result size: {len(net)}")
    print(f"Valid: {verify_network(net)}")
