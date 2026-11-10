# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) with integer entries (int64) such that:
      - second dimension is 11
      - k >= 2
      - no point is the zero vector
      - min squared pairwise distance >= max squared norm
    
    Strategy:
    Use weight-4 vectors (norm^2=4) with constraint dot <= 2.
    Pool size: C(11,4)*2^4 = 5280.
    
    Structural change: Use Python int bitsets for conflict tracking and
    implement iterated local search (ILS) with greedy initialization
    and single-move perturbations.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    d = 11
    
    def generate_pool(w):
        pool = []
        for idx in itertools.combinations(range(d), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool.append(v)
        return np.array(pool)
    
    def build_conflict_bitsets(pool, max_dot):
        """Build conflict masks as Python ints (bit j set if j conflicts with i)."""
        n = len(pool)
        D = pool @ pool.T
        conflict_mask = D > max_dot
        # Convert to Python int bitsets
        bitsets = []
        for i in range(n):
            # Create a Python int where bit j is 1 if conflict_mask[i, j] is True
            mask = conflict_mask[i]
            # Build int from boolean array
            bits = 0
            for j in range(n):
                if mask[j]:
                    bits |= (1 << j)
            bitsets.append(bits)
        return bitsets
    
    def greedy_with_bitset(pool, conflict_bitsets, rng):
        """Greedy selection using Python int bitsets."""
        n = len(pool)
        all_bits = (1 << n) - 1
        available = all_bits
        selected = []
        
        # Create a shuffled order
        order = list(range(n))
        rng.shuffle(order)
        
        for idx in order:
            if available & (1 << idx):
                # Select idx
                selected.append(idx)
                # Remove idx and all its conflicts from available
                available &= ~(conflict_bitsets[idx] | (1 << idx))
        
        return selected
    
    def local_search_improve(pool, conflict_bitsets, selected, rng, time_limit):
        """Try to improve the selected set by adding one vector at a time."""
        n = len(pool)
        selected_set = set(selected)
        # Build available bitset
        available = 0
        for i in range(n):
            if i not in selected_set:
                # Check if i conflicts with any selected vector
                conflicts = False
                for s in selected_set:
                    if conflict_bitsets[i] & (1 << s):
                        conflicts = True
                        break
                if not conflicts:
                    available |= (1 << i)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add any available vector
            if available:
                # Pick a random available vector
                bits = available
                idx = (bits & -bits).bit_length() - 1  # lowest set bit
                selected.append(idx)
                selected_set.add(idx)
                # Update available: remove idx and its conflicts
                available &= ~(conflict_bitsets[idx] | (1 << idx))
                improved = True
        
        return selected
    
    def perturb_and_repair(pool, conflict_bitsets, selected, rng, time_limit):
        """Remove one random selected vector, then try to greedily add back."""
        n = len(pool)
        if len(selected) == 0:
            return selected
        
        # Remove a random vector from selected
        remove_idx = rng.choice(len(selected))
        removed = selected.pop(remove_idx)
        
        # Rebuild available
        selected_set = set(selected)
        available = 0
        for i in range(n):
            if i not in selected_set:
                conflicts = False
                for s in selected_set:
                    if conflict_bitsets[i] & (1 << s):
                        conflicts = True
                        break
                if not conflicts:
                    available |= (1 << i)
        
        # Greedily add vectors from available
        order = list(range(n))
        rng.shuffle(order)
        for idx in order:
            if time.time() > time_limit:
                break
            if available & (1 << idx):
                selected.append(idx)
                selected_set.add(idx)
                available &= ~(conflict_bitsets[idx] | (1 << idx))
        
        return selected
    
    def ils_search(pool, max_dot, time_limit, n_restarts=200):
        """Iterated Local Search."""
        n = len(pool)
        if n == 0:
            return np.empty((0, d), dtype=np.int64)
        
        conflict_bitsets = build_conflict_bitsets(pool, max_dot)
        rng = np.random.RandomState(42)
        
        best_selected = []
        
        for restart in range(n_restarts):
            if time.time() > time_limit:
                break
            
            # Greedy initialization
            selected = greedy_with_bitset(pool, conflict_bitsets, rng)
            
            # Local search: try to add more
            selected = local_search_improve(pool, conflict_bitsets, selected, rng, time_limit)
            
            # Iterated perturbation
            for _ in range(50):
                if time.time() > time_limit:
                    break
                perturbed = perturb_and_repair(pool, conflict_bitsets, list(selected), rng, time_limit)
                if len(perturbed) > len(selected):
                    selected = perturbed
                    # Try to improve further
                    selected = local_search_improve(pool, conflict_bitsets, selected, rng, time_limit)
            
            if len(selected) > len(best_selected):
                best_selected = selected
        
        if len(best_selected) == 0:
            return np.empty((0, d), dtype=np.int64)
        
        return pool[best_selected]

    # Weight 2 fallback: all weight-2 vectors
    points2 = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points2.append(v)
    arr2 = np.array(points2, dtype=np.int64)
    k2 = len(arr2)  # 220

    # Try weight 4 with ILS
    pool4 = generate_pool(4)
    result4 = ils_search(pool4, 2, deadline, n_restarts=200)
    k4 = len(result4)

    # Select the best result
    if k4 >= k2:
        arr = result4
    else:
        arr = arr2

    arr = np.around(arr).astype(np.int64)
    return arr

# EVOLVE-BLOCK-END
