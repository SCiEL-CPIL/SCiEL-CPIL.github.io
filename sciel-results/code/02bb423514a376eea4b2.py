"""Size-efficient sorting networks on 17 wires - search-based optimizer."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 80  # safety margin under 90s wall-clock

def _apply_comparators(wires, comps):
    """Apply a list of comparators to a wire vector. Returns new vector."""
    w = wires.copy()
    for i, j in comps:
        if w[i] > w[j]:
            w[i], w[j] = w[j], w[i]
    return w

def _verify_network(comps, n_wires=N):
    """Verify a network sorts all 2^n_wires Boolean inputs (zero-one principle)."""
    n = len(comps)
    if n == 0:
        # Only valid if n_wires <= 1
        return n_wires <= 1
    # Precompute comparator indices
    ci = np.array([c[0] for c in comps], dtype=np.int32)
    cj = np.array([c[1] for c in comps], dtype=np.int32)
    
    total = 1 << n_wires
    for mask in range(total):
        # Build input vector: bit k is 1 if mask has bit k set
        w = np.array([(mask >> k) & 1 for k in range(n_wires)], dtype=np.int8)
        # Apply comparators
        for idx in range(n):
            i, j = ci[idx], cj[idx]
            if w[i] > w[j]:
                w[i], w[j] = w[j], w[i]
        # Check sorted: all 1s should be at the end
        # Count ones
        ones = w.sum()
        # The last 'ones' positions should all be 1
        if ones > 0:
            if w[-ones:].min() != 1:
                return False
    return True

def _verify_network_fast(comps, n_wires=N):
    """Faster verification using vectorized operations."""
    n = len(comps)
    if n == 0:
        return n_wires <= 1
    
    ci = np.array([c[0] for c in comps], dtype=np.int32)
    cj = np.array([c[1] for c in comps], dtype=np.int32)
    
    total = 1 << n_wires
    # Process in batches for speed
    batch_size = min(1024, total)
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        masks = np.arange(start, end, dtype=np.int32)
        # Build input matrix: (batch, n_wires)
        # bit k of mask -> wire k
        bits = np.arange(n_wires, dtype=np.int32)
        w = ((masks[:, None] >> bits[None, :]) & 1).astype(np.int8)
        
        # Apply comparators
        for idx in range(n):
            i, j = ci[idx], cj[idx]
            wi = w[:, i].copy()
            wj = w[:, j].copy()
            # swap where wi > wj
            mask_swap = wi > wj
            w[mask_swap, i] = wj[mask_swap]
            w[mask_swap, j] = wi[mask_swap]
        
        # Check sorted
        ones = w.sum(axis=1)
        # For each row, check that the last 'ones' elements are all 1
        # Equivalent: w should be non-decreasing
        diffs = np.diff(w, axis=1)
        if np.any(diffs < 0):
            return False
    return True

def _depth_of_network(comps):
    """Compute parallel depth of a network."""
    if not comps:
        return 0
    # Depth = length of longest chain of dependent comparators
    # Two comparators are dependent if they share a wire
    # We can compute this via topological sorting on a DAG
    # Simpler: use the fact that depth = number of "levels" where comparators can run in parallel
    
    # Build dependency: comparator a depends on b if they share a wire and b comes before a
    # Actually, for depth, we just need to find the longest path in the dependency graph
    # where edge b->a means a must come after b because they share a wire
    
    # More efficient: assign each comparator a "level"
    # level[a] = max(level[b] for b < a that shares a wire with a) + 1
    wire_to_last_level = [0] * N
    max_level = 0
    for i, j in comps:
        lv = max(wire_to_last_level[i], wire_to_last_level[j]) + 1
        wire_to_last_level[i] = lv
        wire_to_last_level[j] = lv
        if lv > max_level:
            max_level = lv
    return max_level

def _remove_redundant(comps, n_wires=N):
    """Try to remove redundant comparators from a verified network."""
    if not comps:
        return comps
    result = list(comps)
    n = len(result)
    # Try removing each comparator and check if still valid
    # This is O(n * verification_cost) which can be expensive
    # Use a heuristic: only try removing if the network is "large"
    if n > 200:
        # Do a few random removals
        for attempt in range(min(50, n // 2)):
            idx = random.randint(0, len(result) - 1)
            candidate = result[:idx] + result[idx+1:]
            if _verify_network_fast(candidate, n_wires):
                result = candidate
    else:
        # Try each one
        for idx in range(len(result)):
            candidate = result[:idx] + result[idx+1:]
            if _verify_network_fast(candidate, n_wires):
                result = candidate
                break  # restart after removal
    return result

def _generate_initial_network(n_wires=N):
    """Generate an initial sorting network (odd-even transposition)."""
    comps = []
    for phase in range(n_wires):
        for i in range(phase % 2, n_wires - 1, 2):
            comps.append((i, i + 1))
    return comps

def _swap_comparators(comps, pos1, pos2):
    """Swap two comparators in the list."""
    result = list(comps)
    result[pos1], result[pos2] = result[pos2], result[pos1]
    return result

def _insert_comparator(comps, i, j, pos):
    """Insert comparator (i,j) at position pos."""
    result = list(comps)
    result.insert(pos, (i, j))
    return result

def _remove_comparator(comps, pos):
    """Remove comparator at position pos."""
    result = list(comps)
    result.pop(pos)
    return result

def _move_comparator(comps, pos1, pos2):
    """Move comparator from pos1 to pos2."""
    if pos1 == pos2:
        return list(comps)
    result = list(comps)
    comp = result.pop(pos1)
    result.insert(pos2, comp)
    return result

def construct():
    """Find a small sorting network for N=17 wires using simulated annealing."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    random.seed(42)
    
    # Start with odd-even transposition
    best_comps = _generate_initial_network(N)
    best_cost = len(best_comps)
    
    # Verify initial
    if not _verify_network_fast(best_comps, N):
        # Fallback: should not happen
        return best_comps
    
    # Try to remove redundant from initial
    best_comps = _remove_redundant(best_comps, N)
    best_cost = len(best_comps)
    
    current_comps = list(best_comps)
    current_cost = best_cost
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    # Precompute valid comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    iteration = 0
    while time.time() < deadline - 5:  # 5s safety margin
        iteration += 1
        
        # Choose a move
        move_type = random.random()
        n = len(current_comps)
        
        if move_type < 0.3 and n > 10:
            # Remove a comparator
            pos = random.randint(0, n - 1)
            candidate = _remove_comparator(current_comps, pos)
        elif move_type < 0.5 and n < MAX_COMPARATORS:
            # Insert a random comparator
            i, j = random.choice(all_pairs)
            pos = random.randint(0, n)
            candidate = _insert_comparator(current_comps, i, j, pos)
        elif move_type < 0.7:
            # Swap two comparators
            if n < 2:
                continue
            pos1 = random.randint(0, n - 1)
            pos2 = random.randint(0, n - 1)
            candidate = _swap_comparators(current_comps, pos1, pos2)
        elif move_type < 0.85:
            # Move a comparator
            if n < 2:
                continue
            pos1 = random.randint(0, n - 1)
            pos2 = random.randint(0, n - 1)
            candidate = _move_comparator(current_comps, pos1, pos2)
        else:
            # Modify a comparator's endpoints
            if n == 0:
                continue
            pos = random.randint(0, n - 1)
            i, j = current_comps[pos]
            # Randomly change one endpoint
            if random.random() < 0.5:
                new_i = random.randint(0, N - 1)
                if new_i >= j:
                    new_i = random.randint(0, j - 1) if j > 0 else 0
                if new_i < 0:
                    new_i = 0
                candidate = list(current_comps)
                candidate[pos] = (new_i, j)
            else:
                new_j = random.randint(0, N - 1)
                if new_j <= i:
                    new_j = random.randint(i + 1, N - 1) if i < N - 1 else N - 1
                if new_j >= N:
                    new_j = N - 1
                candidate = list(current_comps)
                candidate[pos] = (i, new_j)
        
        if not candidate or len(candidate) > MAX_COMPARATORS:
            continue
        
        # Check validity
        if not _verify_network_fast(candidate, N):
            # Invalid: accept with probability based on temperature
            # Cost penalty for invalidity
            cand_cost = len(candidate)
            delta = cand_cost - current_cost
            # Also add a penalty for being invalid
            penalty = 100
            delta_total = delta + penalty
            if delta_total < 0 or random.random() < np.exp(-delta_total / T):
                current_comps = candidate
                current_cost = cand_cost
            T *= cooling
            if T < T_min:
                T = T_min
            continue
        
        # Valid network
        cand_cost = len(candidate)
        delta = cand_cost - current_cost
        
        if delta <= 0:
            # Better or equal: always accept
            current_comps = candidate
            current_cost = cand_cost
            if cand_cost < best_cost:
                best_comps = list(candidate)
                best_cost = cand_cost
                # Try to remove redundant
                best_comps = _remove_redundant(best_comps, N)
                best_cost = len(best_comps)
                current_comps = list(best_comps)
                current_cost = best_cost
        else:
            # Worse: accept with probability
            if random.random() < np.exp(-delta / T):
                current_comps = candidate
                current_cost = cand_cost
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Occasionally restart if stuck
        if iteration % 5000 == 0 and T <= T_min * 1.01:
            T = 50.0
    
    # Final cleanup: remove redundant from best
    best_comps = _remove_redundant(best_comps, N)
    
    # Ensure we return a valid network
    if not _verify_network_fast(best_comps, N):
        # Fallback to initial
        best_comps = _generate_initial_network(N)
    
    return best_comps
