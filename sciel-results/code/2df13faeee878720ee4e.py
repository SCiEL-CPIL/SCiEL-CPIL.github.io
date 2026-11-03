import itertools
import random
import time
import numpy as np
import scipy.sparse as sp

# EVOLVE-BLOCK-START

def _build_incidence(v, k, t, deadline):
    """Build sparse incidence matrix A (n_k x n_t) and helper structures."""
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)

    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)

    row_indices = []
    col_indices = []
    data = []

    for i, block in enumerate(k_subs):
        block_set = set(block)
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
        if time.time() > deadline:
            break

    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)

    # Precompute column indices per block for fast updates
    block_cols = []
    for i in range(n_k):
        s = A.indptr[i]
        e = A.indptr[i + 1]
        block_cols.append(A.indices[s:e].copy())

    return k_subs, t_subs, A, block_cols, n_k, n_t


def _greedy_cover(A, block_cols, n_k, n_t, deadline, rng, max_iter=50000):
    """Vectorized greedy set cover using sparse matrix-vector product for gains."""
    u = np.ones(n_t, dtype=np.int8)
    selected = []

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if u.sum() == 0:
            break

        # Gain for each block = number of currently uncovered t-subsets it covers
        gains = A @ u
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]

        if best_gain == 0:
            break

        selected.append(best_idx)
        cols = block_cols[best_idx]
        u[cols] = 0

    return selected, u


def _remove_redundant(selected, block_cols, n_t, deadline):
    """Iteratively remove blocks that don't uniquely cover any t-subset."""
    if not selected:
        return selected

    # cover_count[c] = number of selected blocks covering t-subset c
    cover_count = np.zeros(n_t, dtype=np.int32)
    for idx in selected:
        cover_count[block_cols[idx]] += 1

    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        to_remove = []
        for i in range(len(selected)):
            cols = block_cols[selected[i]]
            is_essential = False
            for c in cols:
                if cover_count[c] == 1:
                    is_essential = True
                    break
            if not is_essential:
                to_remove.append(i)

        if to_remove:
            changed = True
            for i in sorted(to_remove, reverse=True):
                cols = block_cols[selected[i]]
                cover_count[cols] -= 1
                selected.pop(i)

    return selected

def _local_improve_2for1(selected, block_cols, A, n_k, n_t, deadline, rng, max_rounds=10):
    """
    Local search with 2-for-1 moves:
    Try to remove two blocks A and B and add one block C if C covers all t-subsets
    that would become uncovered by removing A and B.
    This reduces the block count by 1.
    """
    if len(selected) < 2:
        return selected

    best = list(selected)
    best_size = len(best)
    
    # Precompute cover counts
    cover_count = np.zeros(n_t, dtype=np.int32)
    for idx in best:
        cover_count[block_cols[idx]] += 1

    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        n_best = len(best)
        if n_best < 2:
            break

        # To speed up, we can randomize the order of pairs
        pairs = list(itertools.combinations(range(n_best), 2))
        rng.shuffle(pairs)

        for i, j in pairs:
            if time.time() > deadline:
                break
            
            idx_i = best[i]
            idx_j = best[j]
            cols_i = block_cols[idx_i]
            cols_j = block_cols[idx_j]
            
            # Determine which t-subsets would become uncovered if we remove i and j
            # A t-subset c is uncovered if:
            # 1. c in cols_i and c not in cols_j and cover_count[c] == 1
            # 2. c in cols_j and c not in cols_i and cover_count[c] == 1
            # 3. c in cols_i and c in cols_j and cover_count[c] <= 2
            
            # Efficiently find these sets using numpy
            # We need a set of indices of t-subsets that are "critically covered" by i or j.
            
            # Create a temporary array for cover counts? No, we can just iterate.
            # But iterating over all t-subsets is slow.
            # Instead, iterate over the union of cols_i and cols_j.
            
            # To avoid double counting in intersection, we can use sets.
            # But sets are slow in Python.
            # Let's use numpy boolean masks.
            
            # We need to check if there exists a block C such that it covers all these critical t-subsets.
            
            # Critical set U
            # We can approximate by just taking the union and checking cover counts.
            # If c is in union, and cover_count[c] <= 2, it might be critical.
            # Specifically:
            # If c in i \ j, critical if count==1
            # If c in j \ i, critical if count==1
            # If c in i \cap j, critical if count<=2
            
            # Let's build the list of critical t-subsets
            critical = []
            # To speed up, we can check if the number of critical subsets is small.
            
            # Optimization: If the intersection of i and j is large, or if they are similar, 
            # the union of unique coverages might be small.
            
            # Let's just iterate over the union of cols_i and cols_j.
            # Since cols are sorted arrays, we can merge them.
            # But for simplicity and speed in Python, let's use sets for the critical check.
            # If it's too slow, we'll see.
            
            # Actually, a faster way:
            # We want to find a block C that contains U.
            # We can check this by: For each candidate C, check if U is a subset of block_cols[C].
            # To make this fast, we can precompute a "signature" or just use set intersection.
            
            # Let's try to find U first.
            # We can use numpy to compute:
            # mask_i = np.zeros(n_t, dtype=bool); mask_i[cols_i] = True
            # This is O(n_t) per pair. Too slow.
            
            # Alternative:
            # Only consider pairs where the blocks are "close" or have high overlap.
            # Or, just try a limited number of random pairs.
            
            # Let's try a heuristic: Only try pairs if the blocks share at least k-2 elements?
            # No, they don't have to share much.
            
            # Let's stick to a simpler local search first: 1-for-1 and 1-for-0 (removal).
            # The 2-for-1 is powerful but complex.
            
            # Let's implement a robust 1-for-1 swap and removal.
            pass

    return best

# The 2-for-1 logic above was getting complicated. Let's use a simpler but effective structural change:
# **Simulated Annealing** on the set of blocks.
# State: Set of blocks.
# Move: 
#   1. Remove a random block.
#   2. Add a random block.
#   3. Swap a random block with another.
# Objective: Minimize (Number of Blocks * 1000 + Number of Uncovered t-subsets).
# This allows temporary increases in uncovered subsets to escape local minima.

def _simulated_annealing(selected, block_cols, A, n_k, n_t, deadline, rng, max_iter=5000):
    """
    Simulated Annealing to minimize block count while covering all t-subsets.
    Objective: cost = N_blocks * 1000 + N_uncovered
    """
    if not selected:
        return selected

    current = list(selected)
    current_uncovered = np.zeros(n_t, dtype=np.int8)
    for idx in current:
        current_uncovered[block_cols[idx]] += 1
    current_uncovered_count = np.sum(current_uncovered == 0)
    current_cost = len(current) * 1000 + current_uncovered_count

    best = list(current)
    best_cost = current_cost
    best_uncovered_count = current_uncovered_count

    T = 50.0
    cooling_rate = 0.999

    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        move_type = rng.random()
        
        if len(current) == 0:
            # Add a random block
            new_idx = rng.randint(0, n_k - 1)
            new_current = [new_idx]
            # Update uncovered
            new_uncovered = np.zeros(n_t, dtype=np.int8)
            new_uncovered[block_cols[new_idx]] += 1
            new_uncovered_count = np.sum(new_uncovered == 0)
            new_cost = len(new_current) * 1000 + new_uncovered_count
            
        elif move_type < 0.3:
            # Remove a random block
            if len(current) <= 1:
                continue
            rem_idx_pos = rng.randint(0, len(current) - 1)
            rem_idx = current[rem_idx_pos]
            
            new_current = current[:rem_idx_pos] + current[rem_idx_pos+1:]
            
            # Update uncovered
            new_uncovered = current_uncovered.copy()
            new_uncovered[block_cols[rem_idx]] -= 1
            new_uncovered_count = np.sum(new_uncovered == 0)
            new_cost = len(new_current) * 1000 + new_uncovered_count
            
        elif move_type < 0.6:
            # Add a random block
            add_idx = rng.randint(0, n_k - 1)
            if add_idx in current:
                continue
            new_current = current + [add_idx]
            
            new_uncovered = current_uncovered.copy()
            new_uncovered[block_cols[add_idx]] += 1
            new_uncovered_count = np.sum(new_uncovered == 0)
            new_cost = len(new_current) * 1000 + new_uncovered_count
            
        else:
            # Swap a random block with another
            if len(current) < 2:
                continue
            swap_pos = rng.randint(0, len(current) - 1)
            old_idx = current[swap_pos]
            new_idx = rng.randint(0, n_k - 1)
            if new_idx == old_idx or new_idx in current:
                continue
                
            new_current = current.copy()
            new_current[swap_pos] = new_idx
            
            new_uncovered = current_uncovered.copy()
            new_uncovered[block_cols[old_idx]] -= 1
            new_uncovered[block_cols[new_idx]] += 1
            new_uncovered_count = np.sum(new_uncovered == 0)
            new_cost = len(new_current) * 1000 + new_uncovered_count

        # Acceptance criterion
        delta_cost = new_cost - current_cost
        if delta_cost < 0:
            # Accept
            current = new_current
            current_uncovered = new_uncovered
            current_uncovered_count = new_uncovered_count
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best = current
                best_uncovered_count = current_uncovered_count
                if best_uncovered_count == 0 and best_cost < 300 * 1000:
                    # Good enough?
                    pass
        else:
            # Metropolis criterion
            if T > 0:
                prob = np.exp(-delta_cost / T)
                if rng.random() < prob:
                    current = new_current
                    current_uncovered = new_uncovered
                    current_uncovered_count = new_uncovered_count
                    current_cost = new_cost

        T *= cooling_rate

    return best

def construct(v, k, t):
    """Construct a covering design C(v,k,t) using SA with greedy initialization."""
    start_time = time.time()
    deadline = start_time + 130.0

    rng = random.Random(42)

    # Build incidence structure
    k_subs, t_subs, A, block_cols, n_k, n_t = _build_incidence(v, k, t, deadline)

    best_blocks = None
    best_count = float('inf')

    # Multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break

        # Greedy initialization
        selected, u = _greedy_cover(A, block_cols, n_k, n_t, deadline, rng)

        if not selected:
            continue

        if u.sum() == 0:
            # Remove redundant
            selected = _remove_redundant(selected, block_cols, n_t, deadline)
            
            # Simulated Annealing
            selected = _simulated_annealing(selected, block_cols, A, n_k, n_t, deadline, rng, max_iter=2000)
            
            # Final redundant removal
            selected = _remove_redundant(selected, block_cols, n_t, deadline)

            if len(selected) < best_count:
                best_count = len(selected)
                best_blocks = selected

    if best_blocks is None:
        selected, u = _greedy_cover(A, block_cols, n_k, n_t, deadline, rng)
        if u.sum() == 0:
            selected = _remove_redundant(selected, block_cols, n_t, deadline)
            best_blocks = selected
        else:
            best_blocks = selected

    result = [list(k_subs[idx]) for idx in best_blocks]
    return result

# EVOLVE-BLOCK-END
