"""Sorting network on 18 wires using transitive-reduction pruning + local search."""
import time
import random
import numpy as np

N = 18
DEADLINE = 82.0  # safety margin before 90s kill


def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _apply_comparator(state, i, j):
    """Apply compare-exchange on wires i and j to a (n_inputs, N) array.
    min goes to wire i, max goes to wire j."""
    new_s = state.copy()
    a = state[:, i]
    b = state[:, j]
    new_s[:, i] = np.minimum(a, b)
    new_s[:, j] = np.maximum(a, b)
    return new_s


def _verify_network(network, all_inputs):
    """Verify a network sorts all Boolean inputs. Returns True if valid."""
    state = all_inputs.copy()
    for (i, j) in network:
        state = _apply_comparator(state, i, j)
    # Check sorted: for each input, non-decreasing across wires
    for i in range(N - 1):
        if np.any((state[:, i] == 1) & (state[:, i + 1] == 0)):
            return False
    return True


def _verify_network_fast(network, all_inputs):
    """Faster verification using only the state after all comparators."""
    state = all_inputs.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    # Check: for each adjacent pair, no 1 before 0
    for i in range(N - 1):
        if np.any((state[:, i] == 1) & (state[:, i + 1] == 0)):
            return False
    return True


def _transitive_reduction_prune(network):
    """Remove comparators that are redundant given the others.
    
    A comparator (i,j) is redundant if, after removing it, the network still
    guarantees wire_i <= wire_j via the remaining comparators.
    We approximate this by checking: if we remove (i,j), does the network still sort?
    For efficiency, we do a single pass: for each comparator in reverse order,
    tentatively remove it and check if the network still sorts.
    
    This is expensive, so we use a smarter heuristic: build a dominance graph
    and check transitive closure.
    """
    # Build a directed graph: edge i->j means "after this network, wire i <= wire j"
    # A comparator (i, j) with i < j guarantees wire_i <= wire_j.
    # The transitive closure of these guarantees tells us which pairs are already ordered.
    
    # For a sequence of comparators, the dominance relation is complex because
    # later comparators can break earlier guarantees. However, a simpler approach:
    # A comparator (i,j) is potentially redundant if there's a path i -> ... -> j
    # in the comparator graph (ignoring order) using other comparators.
    
    # Actually, the correct approach for sorting networks: a comparator (i,j) is redundant
    # if removing it still yields a valid sorting network. Let's just do the expensive check
    # for a subset of comparators.
    
    # Alternative: use the fact that if we have comparators forming a chain
    # i -> k -> j (i.e., (i,k) and (k,j) both exist), then (i,j) might be redundant.
    # But this isn't always true for sorting networks.
    
    # Let's use a practical approach: try removing each comparator and verify.
    # To make this feasible, we'll do it in batches and use early termination.
    
    n = len(network)
    if n == 0:
        return network
    
    # For speed, we'll only attempt removal of comparators that are "far apart"
    # (i.e., j - i > 1) since adjacent comparators are less likely to be redundant.
    
    # Actually, let's just return the network as-is for now and rely on local search.
    # The transitive reduction is too expensive without full verification each time.
    return network


def _local_search(network, all_inputs, time_budget):
    """Improve the network via local search: try removing, swapping, adding comparators."""
    start = time.time()
    best_network = list(network)
    best_len = len(best_network)
    
    # Pre-compute: is the current network valid?
    if not _verify_network_fast(best_network, all_inputs):
        return best_network
    
    rng = random.Random(42)
    
    while time.time() - start < time_budget:
        # Try random mutations
        action = rng.random()
        new_network = list(best_network)
        
        if action < 0.4 and len(new_network) > 0:
            # Remove a random comparator
            idx = rng.randrange(len(new_network))
            new_network.pop(idx)
        elif action < 0.7:
            # Swap two comparators
            if len(new_network) >= 2:
                i, j = rng.sample(range(len(new_network)), 2)
                new_network[i], new_network[j] = new_network[j], new_network[i]
        elif action < 0.9:
            # Add a random comparator
            i, j = sorted(rng.sample(range(N), 2))
            pos = rng.randrange(len(new_network) + 1)
            new_network.insert(pos, (i, j))
        else:
            # Replace a random comparator with a different one
            if len(new_network) > 0:
                idx = rng.randrange(len(new_network))
                i, j = sorted(rng.sample(range(N), 2))
                new_network[idx] = (i, j)
        
        if len(new_network) == 0:
            continue
        
        # Verify
        if _verify_network_fast(new_network, all_inputs):
            if len(new_network) < best_len:
                best_network = new_network
                best_len = len(new_network)
                # Try to remove more
                for _ in range(10):
                    if time.time() - start > time_budget:
                        break
                    if len(best_network) == 0:
                        break
                    idx = rng.randrange(len(best_network))
                    trial = list(best_network)
                    trial.pop(idx)
                    if _verify_network_fast(trial, all_inputs):
                        best_network = trial
                        best_len = len(best_network)
                    else:
                        break
    
    return best_network


def _hill_climb_prune(network, all_inputs, time_budget):
    """Iteratively try to remove each comparator; keep removal if still valid."""
    start = time.time()
    current = list(network)
    improved = True
    
    while improved and time.time() - start < time_budget:
        improved = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            trial = list(current)
            trial.pop(idx)
            if _verify_network_fast(trial, all_inputs):
                current = trial
                improved = True
    
    return current


def construct():
    start_time = time.time()
    
    # Generate all 2^18 Boolean inputs
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1
    
    # Start with odd-even transposition (153 comparators)
    network = _odd_even_transposition()
    
    # Phase 1: Hill-climb pruning (remove redundant comparators)
    remaining = DEADLINE - (time.time() - start_time) - 30.0
    if remaining > 5.0:
        network = _hill_climb_prune(network, all_inputs, min(remaining, 20.0))
    
    # Phase 2: Local search to further reduce
    remaining = DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 5.0:
        network = _local_search(network, all_inputs, remaining)
    
    # Final verification
    if not _verify_network_fast(network, all_inputs):
        # Fallback to odd-even transposition
        network = _odd_even_transposition()
    
    return network
