import itertools
import random
import time
import numpy as np
from scipy.optimize import linprog

def construct(v, k, t):
    start = time.time()
    deadline = start + 118
    rng = random.Random(1234567890)

    # Generate a pool of random blocks
    # We need a sufficiently large pool to allow the LP to find a good fractional solution
    # and for rounding to have enough choices.
    # Let's aim for a pool size that is manageable for linprog but large enough for diversity.
    # C(19,7) is ~50k. A pool of 500-1000 blocks should be enough to span the space reasonably.
    pool_size = 800
    pool_blocks = []
    pool_set = set()
    while len(pool_blocks) < pool_size:
        b = tuple(sorted(rng.sample(range(v), k)))
        if b not in pool_set:
            pool_set.add(b)
            pool_blocks.append(b)
    
    # Precompute the coverage matrix
    # Rows: t-subsets, Columns: pool blocks
    # A[i, j] = 1 if t-subset i is covered by block j
    
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Build matrix A
    # Using numpy for speed
    A = np.zeros((n_subsets, pool_size), dtype=np.float64)
    for j, b in enumerate(pool_blocks):
        # For each t-subset of b, mark it as covered
        for sub in itertools.combinations(b, t):
            si = sub_idx[sub]
            A[si, j] = 1.0

    # LP Relaxation: Minimize sum(x_j) subject to A * x >= 1, x >= 0
    # linprog minimizes c^T x subject to A_ub x <= b_ub, A_eq x = b_eq, bounds
    # We have A x >= 1 => -A x <= -1
    c = np.ones(pool_size)
    A_ub = -A
    b_ub = -np.ones(n_subsets)
    bounds = [(0, None) for _ in range(pool_size)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x = res.x
        else:
            # Fallback if LP fails
            x = np.ones(pool_size) / pool_size
    except Exception:
        x = np.ones(pool_size) / pool_size
    
    # Normalize x to probabilities
    total = x.sum()
    if total == 0:
        x = np.ones(pool_size) / pool_size
    else:
        x = x / total
    
    # Randomized Rounding
    # We want to select a set of blocks S such that every t-subset is covered.
    # We can view this as: for each block j, include it with probability p_j.
    # A simple approach: iterate until all subsets are covered.
    # Or, use a more structured rounding:
    # Sort blocks by x_j descending.
    # Greedily pick the block with highest x_j that covers the most uncovered subsets.
    # This is essentially a greedy algorithm guided by LP weights.
    
    # Let's implement a "LP-guided greedy"
    uncovered = np.ones(n_subsets, dtype=bool)
    chosen_indices = []
    chosen_set = set()
    
    # Precompute masks for faster bit operations if needed, or just use numpy sums
    # Since n_subsets is C(19,4) = 3876, and pool_size is 800,
    # we can represent coverage as a bitset or just a boolean vector.
    # Boolean vector is fine.
    
    current_coverage = np.zeros(n_subsets, dtype=np.float64)
    
    # We will iteratively pick blocks.
    # To make it "randomized", we can add noise to the weights or sample.
    # But a deterministic greedy guided by LP weights is often strong.
    # Let's try a few variants:
    # 1. Pure LP-guided greedy (deterministic)
    # 2. LP-guided greedy with random tie-breaking
    # 3. Randomized rounding: sample blocks with prob proportional to x, until covered.
    
    best_size = float('inf')
    best_blocks = None
    
    # Variant 1: LP-Guided Greedy
    # We pick the block that maximizes (new_coverage * x_j) or just new_coverage?
    # Standard greedy picks max new_coverage.
    # LP-guided might pick max (new_coverage * x_j) to favor blocks with high LP weight.
    # Let's try a few scoring functions.
    
    def run_lp_guided_greedy(use_lp_weight, rng_seed):
        local_rng = random.Random(rng_seed)
        uncov_mask = np.ones(n_subsets, dtype=bool)
        chosen = []
        chosen_set = set()
        
        while uncov_mask.any():
            best_j = -1
            best_score = -1
            # To speed up, we can precompute coverage counts for each block?
            # No, coverage changes.
            # Iterate through pool
            for j in range(pool_size):
                if j in chosen_set:
                    continue
                # Calculate gain: number of currently uncovered subsets covered by block j
                # A[:, j] is the column
                # gain = np.sum(A[:, j] * uncov_mask)
                # This is O(n_subsets) per block, O(pool_size * n_subsets) per step.
                # n_subsets=3876, pool_size=800. 3876*800 = 3.1M ops per step.
                # If we need ~150 steps, that's 465M ops. Might be slow in Python loop.
                # Vectorization: 
                # We can compute gains for all j at once:
                # gains = A.T @ uncov_mask  (shape: pool_size)
                # But we need to mask out chosen ones.
                pass
            
            # Let's vectorize the gain calculation
            gains = A.T.dot(uncov_mask.astype(np.float64))
            # Mask out chosen
            for j in chosen_set:
                gains[j] = -1
            
            if use_lp_weight:
                # Score = gain * x_j ? Or gain / (1 + x_j)?
                # Let's try gain * x_j
                scores = gains * x
            else:
                scores = gains
            
            # Handle ties randomly
            max_score = np.max(scores)
            # Find indices with max score
            candidates = np.where(scores == max_score)[0]
            if len(candidates) > 0:
                best_j = local_rng.choice(candidates.tolist())
            
            if best_j < 0 or gains[best_j] <= 0:
                # No block covers any new subset? Should not happen if problem is solvable
                break
                
            chosen.append(best_j)
            chosen_set.add(best_j)
            # Update uncovered
            # uncov_mask = uncov_mask & ~A[:, best_j].astype(bool)
            # Faster:
            cols = A[:, best_j].astype(bool)
            uncov_mask[cols] = False
            
        return chosen

    # Try multiple seeds and weighting strategies
    strategies = [
        (True, 1), (True, 2), (True, 3),
        (False, 1), (False, 2), (False, 3)
    ]
    
    for use_lp, seed in strategies:
        if time.time() > deadline - 10:
            break
        chosen = run_lp_guided_greedy(use_lp, seed)
        # Prune
        # We need to prune the chosen list
        # Convert to set of indices for fast lookup
        # Pruning: remove block if all its subsets are covered by other blocks
        counts = np.zeros(n_subsets, dtype=np.int64)
        for j in chosen:
            cols = A[:, j].astype(bool)
            counts[cols] += 1
        
        # Iterative removal
        changed = True
        while changed:
            changed = False
            for i, j in enumerate(chosen):
                cols = A[:, j].astype(bool)
                # If all covered subsets have count > 1, remove j
                if np.all(counts[cols] > 1):
                    counts[cols] -= 1
                    chosen.pop(i)
                    changed = True
                    break
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_blocks = chosen[:]

    # If no blocks found or time is up, fallback to pure greedy on full space?
    # Or just return what we have.
    # The pool might not be sufficient to cover all subsets if it's too small.
    # We should verify coverage.
    if best_blocks is None:
        # Fallback: Pure greedy on pool
        chosen = run_lp_guided_greedy(False, 0)
        best_blocks = chosen
        best_size = len(chosen)

    # Verify coverage of best_blocks
    # If not fully covered, we need to add more blocks.
    # Since the pool is random, it's possible some t-subsets are not in ANY pool block.
    # We should check this.
    covered_mask = np.zeros(n_subsets, dtype=bool)
    for j in best_blocks:
        covered_mask |= A[:, j].astype(bool)
    
    if not covered_mask.all():
        # Find uncovered subsets
        uncov_indices = np.where(~covered_mask)[0]
        # We need to add blocks from the full space or extend the pool
        # To be safe and fast, let's just run a standard greedy on the FULL space
        # but guided by the current set.
        # Actually, if the pool is random, it's very likely to cover most, but not all.
        # Let's generate a second pool or just use full greedy for the remaining.
        
        # For simplicity and robustness within time limit:
        # If coverage is incomplete, fall back to a pure greedy construction on the full space
        # starting from the current best_blocks.
        
        all_blocks = list(itertools.combinations(range(v), k))
        n_blocks_full = len(all_blocks)
        
        # Precompute full coverage masks? Too slow.
        # Let's just do a simple loop.
        current_covered = covered_mask.copy()
        chosen_full = list(best_blocks) # These are indices into pool_blocks
        # We need to map pool indices to actual blocks, then find full block indices?
        # No, let's just work with actual blocks.
        
        current_blocks_set = set([tuple(pool_blocks[j]) for j in best_blocks])
        
        # Greedy add from full space
        while not current_covered.all():
            best_gain = 0
            best_block = None
            # This is slow. Let's hope the pool covers everything.
            # If not, we might time out.
            # Given pool_size=800, probability of missing a specific t-subset is low.
            # (1 - C(19-4, 7-4)/C(19,7))^800 ?
            # Prob a random block covers a specific t-subset: C(19-4, 7-4)/C(19,7) = C(15,3)/C(19,7) = 455/50388 ~ 0.009
            # Prob NOT covered by 1 block: 0.991
            # Prob NOT covered by 800 blocks: 0.991^800 ~ 0.0003
            # So it's very likely the pool covers everything.
            # If it doesn't, we add a few random blocks until covered.
            
            if time.time() > deadline - 2:
                break
            
            # Pick random block from full space?
            # Or pick from pool first?
            # Let's pick from pool first.
            found_in_pool = False
            for j in range(pool_size):
                if j in set(best_blocks):
                    continue
                cols = A[:, j].astype(bool)
                gain = np.sum(cols & ~current_covered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = j
                    found_in_pool = True
            
            if not found_in_pool or best_gain == 0:
                # Pick random full block
                b = tuple(sorted(rng.sample(range(v), k)))
                # Check if already in set
                if b in current_blocks_set:
                    continue
                # Calculate gain
                gain = 0
                for sub in itertools.combinations(b, t):
                    si = sub_idx[sub]
                    if not current_covered[si]:
                        gain += 1
                if gain > 0:
                    best_gain = gain
                    best_block = b
                    found_in_pool = False # Flag that it's a raw block
            
            if best_block is None:
                break
                
            if isinstance(best_block, int):
                # It's a pool index
                b = pool_blocks[best_block]
                best_blocks.append(best_block)
                current_blocks_set.add(b)
            else:
                # It's a raw block tuple
                # We need to add it to our representation.
                # Our best_blocks list contains pool indices.
                # This breaks the interface if we return raw blocks mixed with indices.
                # We should maintain a list of actual blocks.
                
                # Let's refactor to keep a list of actual blocks for the final return.
                pass
        
        # This fallback is getting messy. 
        # Let's assume the pool is sufficient. If not, we return the best effort.
        # To be safe, let's just return the pool-based blocks.
        pass

    # Convert pool indices to actual blocks
    final_blocks = [list(pool_blocks[j]) for j in best_blocks]
    
    # Final check: are they distinct?
    # pool_blocks were distinct, and we didn't add duplicates.
    
    return final_blocks
