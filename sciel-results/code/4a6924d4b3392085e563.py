"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START

# Precompute bit masks for fast verification
_ALL_MASKS = np.arange(1 << N, dtype=np.int64)
_WIRE_BITS = np.array([(_ALL_MASKS >> i) & 1 for i in range(N)], dtype=np.int8)  # shape (N, 2^N)

def _verify_network_bitparallel(net):
    """Verify that a network sorts all 2^N Boolean inputs using bit-parallel ops."""
    if not net:
        return N <= 1
    wires = _WIRE_BITS.copy()  # shape (N, 2^N)
    for i, j in net:
        if i < 0 or j >= N or i >= j:
            return False
        wi = wires[i]
        wj = wires[j]
        # min goes to i, max goes to j
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    # Check if all wires are sorted: wire[k] <= wire[k+1] for all k
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True

def _depth(net):
    """Compute parallel depth of a network."""
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _odd_even_transposition_network(n):
    """Odd-even transposition sorting network (correct but large)."""
    net = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                net.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                net.append((i, i + 1))
    return net

def _count_inversions(wires):
    """Count inversions in the current wire assignment (bit-parallel)."""
    total = 0
    for i in range(N):
        for j in range(i + 1, N):
            if np.any(wires[i] > wires[j]):
                total += 1
    return total

def _build_bitonic_network_18():
    """
    Build a bitonic-style sorting network for 18 wires.
    Use a merge-based approach: split into two halves, sort each, then merge.
    For non-power-of-2, we handle the split carefully.
    """
    def _sort_n(n, offset=0, ascending=True):
        """Sort n wires starting at offset, recursively."""
        if n <= 1:
            return []
        if n == 2:
            if ascending:
                return [(offset, offset + 1)]
            else:
                return [(offset + 1, offset)]  # This is invalid since i<j required
                # Actually, to sort descending with i<j, we swap: put max at i, min at j
                # So (i, j) with i<j: min goes to i, max to j => ascending
                # For descending, we need max at i, min at j => but that's just swapping the role
                # Actually, a comparator (i,j) always puts min at i and max at j.
                # To get descending order, we'd need the opposite.
                # So for descending sort, we can just do ascending sort and then reverse the wire order?
                # No, the wires are fixed. We need the values to be in descending order on the wires.
                # So wire[0] >= wire[1] >= ... >= wire[n-1]
                # A comparator (i,j) with i<j puts min at i, max at j.
                # So if we want descending, we need the larger value at the smaller index.
                # This means we need a comparator that puts max at i and min at j, which is the opposite.
                # But the task says: "Each compare-exchange writes min to wire i and max to j."
                # So we can only do ascending comparators.
                # To sort descending, we can sort ascending and then the wires are in ascending order,
                # which is the opposite of what we want.
                # 
                # Wait, let me re-read the task: "min to wire i and max to j".
                # So if i=0, j=1, and values are [3, 1], then after comparator: wire[0]=1, wire[1]=3.
                # So the comparator always sorts the pair in ascending order (smaller at lower index).
                # 
                # So to get a globally descending sort, we'd need the values to be in descending order
                # on the wires, i.e., wire[0] >= wire[1] >= ... >= wire[N-1].
                # But the comparators only sort in ascending order.
                # 
                # So bitonic sort for descending isn't straightforward with this comparator definition.
                # Let me just use a simpler approach: build a correct sorting network using a known method.
                # 
                # Let me use the approach from the Batcher odd-even merge sort, which works for any n.
                pass
        
        # For now, let's just use a simpler method
        return _odd_even_transposition_network(n)
    
    # Let me just use odd-even transposition for now, and rely on local search to improve
    return _odd_even_transposition_network(N)

def _build_initial_network():
    """Build an initial sorting network."""
    return _odd_even_transposition_network(N)

def _local_search(net, max_time=50.0, seed=42):
    """
    Local search with Remove, Replace, and Move operations.
    Uses bit-parallel verification.
    """
    start_time = time.time()
    best_net = list(net)
    best_score = len(best_net)
    rng = random.Random(seed)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Precompute a set for O(1) lookup
    comp_set = set(all_comparators)
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Phase 1: Try removing comparators (most impactful)
        indices = list(range(len(best_net)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() - start_time >= max_time:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network_bitparallel(candidate):
                best_net = candidate
                best_score = len(best_net)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try replacing a comparator with a different one
        if time.time() - start_time >= max_time:
            break
        
        for _ in range(30):
            if time.time() - start_time >= max_time:
                break
            idx = rng.randint(0, len(best_net) - 1)
            old_comp = best_net[idx]
            new_comp = rng.choice(all_comparators)
            if new_comp == old_comp:
                continue
            candidate = list(best_net)
            candidate[idx] = new_comp
            if _verify_network_bitparallel(candidate):
                best_net = candidate
                best_score = len(best_net)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 3: Try moving a comparator to a different position
        if time.time() - start_time >= max_time:
            break
        
        for _ in range(20):
            if time.time() - start_time >= max_time:
                break
            idx = rng.randint(0, len(best_net) - 1)
            comp = best_net[idx]
            new_idx = rng.randint(0, len(best_net) - 1)
            if new_idx == idx:
                continue
            candidate = list(best_net)
            del candidate[idx]
            if new_idx > idx:
                new_idx -= 1
            candidate.insert(new_idx, comp)
            if _verify_network_bitparallel(candidate):
                best_net = candidate
                best_score = len(best_net)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 4: Aggressive perturbation - remove a random subset
        if time.time() - start_time >= max_time:
            break
        
        n_remove = min(2, len(best_net) // 5)
        if n_remove > 0:
            remove_indices = sorted(rng.sample(range(len(best_net)), n_remove), reverse=True)
            candidate = list(best_net)
            for ri in remove_indices:
                del candidate[ri]
            if _verify_network_bitparallel(candidate):
                best_net = candidate
                best_score = len(best_net)
                improved = True
        
        if improved:
            continue
        
        # Phase 5: Random restart with perturbation
        if time.time() - start_time >= max_time:
            break
        
        n_replace = rng.randint(3, 8)
        if n_replace > len(best_net):
            n_replace = len(best_net)
        replace_indices = rng.sample(range(len(best_net)), n_replace)
        candidate = list(best_net)
        for ri in replace_indices:
            candidate[ri] = rng.choice(all_comparators)
        if _verify_network_bitparallel(candidate):
            best_net = candidate
            best_score = len(best_net)
    
    return best_net

def _simulated_annealing(net, max_time=50.0, seed=42):
    """
    Simulated annealing that allows accepting worse solutions temporarily.
    Uses a custom acceptance criterion based on network size.
    """
    start_time = time.time()
    current_net = list(net)
    current_score = len(current_net)
    best_net = list(current_net)
    best_score = current_score
    
    rng = random.Random(seed)
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    T = 100.0  # Initial temperature
    cooling_rate = 0.999
    
    while time.time() - start_time < max_time:
        # Generate a neighbor
        move_type = rng.random()
        candidate = list(current_net)
        
        if move_type < 0.4:
            # Remove a comparator
            if len(candidate) > 0:
                idx = rng.randint(0, len(candidate) - 1)
                del candidate[idx]
        elif move_type < 0.7:
            # Replace a comparator
            if len(candidate) > 0:
                idx = rng.randint(0, len(candidate) - 1)
                candidate[idx] = rng.choice(all_comparators)
        elif move_type < 0.85:
            # Move a comparator
            if len(candidate) > 1:
                idx = rng.randint(0, len(candidate) - 1)
                comp = candidate[idx]
                del candidate[idx]
                new_idx = rng.randint(0, len(candidate))
                candidate.insert(new_idx, comp)
        else:
            # Add a random comparator (to escape local minima)
            if len(candidate) < 200:
                candidate.append(rng.choice(all_comparators))
        
        # Verify
        if not _verify_network_bitparallel(candidate):
            # Reject invalid moves
            # But for annealing, we might want to accept some invalid moves temporarily?
            # No, we must maintain validity. So reject.
            # Instead, let's just reject and try again
            continue
        
        candidate_score = len(candidate)
        delta = candidate_score - current_score
        
        # Acceptance criterion
        if delta <= 0:
            # Improvement or equal: accept
            current_net = candidate
            current_score = candidate_score
            if current_score < best_score:
                best_net = candidate
                best_score = current_score
        else:
            # Worse: accept with probability exp(-delta/T)
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_net = candidate
                    current_score = candidate_score
        
        T *= cooling_rate
        if T < 0.01:
            # Reheat
            T = 50.0
    
    return best_net

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin before 90s kill
    
    # Start with odd-even transposition (known correct)
    net = _build_initial_network()
    
    # Verify it's correct
    if not _verify_network_bitparallel(net):
        raise ValueError("Initial network failed verification")
    
    # Phase 1: Local search
    if time.time() - start_time < deadline - 10.0:
        remaining = deadline - time.time() - 10.0
        net = _local_search(net, max_time=min(remaining * 0.6, 40.0), seed=42)
    
    # Phase 2: Simulated annealing from the local search result
    if time.time() - start_time < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        net_sa = _simulated_annealing(net, max_time=min(remaining * 0.8, 30.0), seed=123)
        if len(net_sa) < len(net):
            net = net_sa
    
    # Phase 3: Final local search polish
    if time.time() - start_time < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        net = _local_search(net, max_time=min(remaining * 0.9, 15.0), seed=777)
    
    # Final verification
    if not _verify_network_bitparallel(net):
        # Fallback to original if something went wrong
        net = _build_initial_network()
    
    return net

# EVOLVE-BLOCK-END
