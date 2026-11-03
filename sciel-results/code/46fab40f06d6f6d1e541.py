import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115  # safety margin before 120s kill
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Precompute: for each pair of subsets, whether they are compatible
    # Compatible if |S ∩ T| <= 2 (i.e., they don't share 3 or 4 elements)
    # Two 4-sets are incompatible if |S ∩ T| >= 3
    
    # Build compatibility matrix using vectorized approach
    # Represent each subset as a bitmask
    masks = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(subsets):
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks[i] = m
    
    # Compute pairwise intersection sizes
    # For each pair (i, j), count bits in (masks[i] & masks[j])
    # Use broadcasting for efficiency
    # masks shape: (n,)
    # We need (n, n) matrix of popcount(masks[i] & masks[j])
    
    # Expand to (n, 1) and (1, n) for broadcasting
    m1 = masks[:, np.newaxis]  # (n, 1)
    m2 = masks[np.newaxis, :]  # (1, n)
    
    # Bitwise AND: (n, n)
    and_matrix = m1 & m2
    
    # Count bits: use numpy's bit_count or manual
    # For int64, we can use a lookup or loop over bits
    # Since D=11, we can do it efficiently
    
    # Method: use the fact that we only need to check if popcount >= 3
    # popcount(x) >= 3 iff x has at least 3 bits set
    # We can check: x & (x-1) & (x-2) != 0 doesn't work simply
    # Let's just compute popcount for values 0 to 2^11-1 = 2047
    
    popcount_table = np.zeros(2048, dtype=np.int32)
    for i in range(2048):
        popcount_table[i] = bin(i).count('1')
    
    # and_matrix values are 0 to 2^11-1 = 2047
    popcount_matrix = popcount_table[and_matrix]  # (n, n)
    
    # Compatible if popcount <= 2
    compatible = (popcount_matrix <= 2)  # (n, n) boolean matrix
    # Note: compatible[i][i] is True (popcount of mask[i] is 4 > 2), so diagonal is False
    # Actually popcount of a 4-set mask is 4, so compatible[i][i] = (4 <= 2) = False. Good.
    
    # We want a maximum independent set in the conflict graph
    # Conflict graph: edge between i and j if NOT compatible[i][j]
    # Independent set: no two selected vertices are adjacent in conflict graph
    # Equivalently: for all pairs i,j in selected, compatible[i][j] is True
    
    # Use simulated annealing on the 330-bit configuration
    # State: binary vector x of length n, x[i] = 1 if subset i is selected
    # Energy: number of incompatible pairs among selected subsets
    # We want to minimize energy while maximizing sum(x)
    # Or: maximize sum(x) - penalty * energy
    
    # Better: use a greedy + local search hybrid
    # Start with a good greedy solution, then do local search (flip, swap)
    
    # Greedy initialization: pick subsets in order of fewest conflicts
    # For each subset, count how many other subsets it conflicts with
    conflict_count = np.sum(~compatible, axis=1)  # number of conflicts for each subset
    
    # Greedy: repeatedly pick the subset with fewest conflicts among remaining
    remaining = np.ones(n, dtype=bool)
    selected = np.zeros(n, dtype=bool)
    
    while remaining.any():
        # Among remaining, pick the one with fewest conflicts with other remaining
        # conflicts with remaining = sum of ~compatible[i, j] for j in remaining
        conflicts_with_remaining = np.sum(np.where(remaining[None, :], ~compatible, 0), axis=1)
        # Only consider remaining
        candidates = np.where(remaining)[0]
        if len(candidates) == 0:
            break
        # Pick the one with minimum conflicts
        min_idx = candidates[np.argmin(conflicts_with_remaining[candidates])]
        selected[min_idx] = True
        remaining[min_idx] = False
        # Remove all that conflict with min_idx from remaining
        remaining[~compatible[min_idx, :]] = False
    
    best_selected = selected.copy()
    best_score = np.sum(best_selected)
    
    # Local search: try flipping individual bits and swaps
    def get_score(x):
        """Compute score: number of selected subsets with no conflicts among them."""
        sel = np.where(x)[0]
        if len(sel) == 0:
            return 0
        # Check if all pairs are compatible
        if len(sel) == 1:
            return 1
        # Check compatibility among selected
        sub_compat = compatible[np.ix_(sel, sel)]
        # All off-diagonal entries should be True
        n_sel = len(sel)
        if n_sel > 1:
            # Check upper triangle
            iu = np.triu_indices(n_sel, k=1)
            if not np.all(sub_compat[iu]):
                return -1  # invalid
        return n_sel
    
    def is_valid(x):
        """Check if selection is valid (no conflicts)."""
        sel = np.where(x)[0]
        if len(sel) <= 1:
            return True
        sub_compat = compatible[np.ix_(sel, sel)]
        n_sel = len(sel)
        iu = np.triu_indices(n_sel, k=1)
        return np.all(sub_compat[iu])
    
    # Simulated annealing local search
    current = best_selected.copy()
    current_score = best_score
    T = 10.0  # initial temperature
    cooling = 0.999
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Try random moves
        move_type = rng.choice(['flip_in', 'flip_out', 'swap'])
        
        if move_type == 'flip_in':
            # Try adding a random unselected subset
            unselected = np.where(~current)[0]
            if len(unselected) == 0:
                continue
            idx = rng.choice(unselected)
            new_x = current.copy()
            new_x[idx] = True
            if is_valid(new_x):
                new_score = np.sum(new_x)
                if new_score >= current_score:
                    current = new_x
                    current_score = new_score
                    if current_score > best_score:
                        best_selected = current.copy()
                        best_score = current_score
                # Always accept improving or equal
            else:
                # Invalid, reject
                pass
        elif move_type == 'flip_out':
            # Try removing a random selected subset
            selected_idx = np.where(current)[0]
            if len(selected_idx) == 0:
                continue
            idx = rng.choice(selected_idx)
            new_x = current.copy()
            new_x[idx] = False
            new_score = np.sum(new_x)
            # Removing always keeps validity
            delta = new_score - current_score
            if delta >= 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                current = new_x
                current_score = new_score
        elif move_type == 'swap':
            # Try swapping: remove one selected, add one unselected
            selected_idx = np.where(current)[0]
            unselected = np.where(~current)[0]
            if len(selected_idx) == 0 or len(unselected) == 0:
                continue
            out_idx = rng.choice(selected_idx)
            in_idx = rng.choice(unselected)
            new_x = current.copy()
            new_x[out_idx] = False
            new_x[in_idx] = True
            if is_valid(new_x):
                new_score = np.sum(new_x)
                delta = new_score - current_score
                if delta >= 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                    current = new_x
                    current_score = new_score
                    if current_score > best_score:
                        best_selected = current.copy()
                        best_score = current_score
        
        T *= cooling
    
    # Use the best found
    final_selected = best_selected
    sel_indices = np.where(final_selected)[0]
    
    # Build vectors: for each selected subset, generate all 2^4 = 16 sign combinations
    vectors = []
    for idx in sel_indices:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    if len(vectors) == 0:
        # Fallback: return a single vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    return np.array(vectors, dtype=np.int64)
