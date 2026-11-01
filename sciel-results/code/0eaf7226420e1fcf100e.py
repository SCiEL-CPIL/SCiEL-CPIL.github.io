"""Direct block-level local search for covering design C(24,7,3)."""
import itertools
import random
import time

import numpy as np

V = 24
K = 7
T = 3
ALL_TSETS = np.array(list(itertools.combinations(range(V), T)))
NT = len(ALL_TSETS)  # 2024
TSET_IDX = {ts: i for i, ts in enumerate(ALL_TSETS)}


def _block_tset_indices(block):
    """Return array of t-set indices covered by a block."""
    return np.array([TSET_IDX[ts] for ts in itertools.combinations(block, T)], dtype=np.int32)


def _fast_greedy(rng, time_limit):
    """Fast greedy set-cover initialization."""
    deadline = time.time() + time_limit
    # Precompute all 7-subsets' coverage as a lookup for speed
    # We'll just do standard greedy
    uncovered = np.ones(NT, dtype=bool)
    blocks = []
    # Precompute: for each point, list of t-sets containing it (for candidate scoring)
    # Actually just do greedy over random candidates
    
    while uncovered.any() and time.time() < deadline:
        # Find an uncovered t-set to anchor
        unc_idx = np.where(uncovered)[0]
        anchor = ALL_TSETS[rng.randint(0, len(unc_idx) - 1)]
        anchor_set = set(anchor)
        
        # Build block around anchor greedily
        block = list(anchor)
        for _ in range(K - T):
            candidates = [x for x in range(V) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                # Count how many uncovered t-sets would be covered by adding x
                # A t-set {a,b,x} is covered if {a,b} subset of current block
                gain = 0
                for a, b in itertools.combinations(block, 2):
                    ts = tuple(sorted((a, b, x)))
                    idx = TSET_IDX[ts]
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < K:
            # Fill randomly
            remaining = [x for x in range(V) if x not in block]
            block.extend(rng.sample(remaining, K - len(block)))
        
        block.sort()
        blocks.append(tuple(block))
        # Update uncovered
        for ts in itertools.combinations(block, T):
            uncovered[TSET_IDX[ts]] = False
    
    return [list(b) for b in blocks]


def _build_coverage_matrix(blocks):
    """Build (n_blocks, NT) boolean matrix: cov[i,j]=True if block i covers t-set j."""
    n = len(blocks)
    cov = np.zeros((n, NT), dtype=bool)
    for i, b in enumerate(blocks):
        idxs = _block_tset_indices(b)
        cov[i, idxs] = True
    return cov


def _is_valid(cov):
    """Check if every t-set is covered by at least one block."""
    return cov.any(axis=0).all()


def _try_remove(cov, i):
    """Try removing block i. Returns (new_cov, success)."""
    if not cov[i].any():
        return None, False
    # Check: for each t-set covered by block i, is it covered by another block?
    # cov_without_i = cov with row i zeroed
    # A t-set is still covered if cov[:,j].sum() >= 2 (before removal) or
    # equivalently, for all j where cov[i,j], cov[j].sum() >= 2
    col_sums = cov.sum(axis=0)
    # For t-sets covered by block i, need col_sums[j] >= 2
    if not (col_sums[cov[i]] >= 2).all():
        return None, False
    new_cov = cov.copy()
    new_cov[i, :] = False
    return new_cov, True


def _try_swap(cov, remove_i, add_block):
    """Try swapping: remove block remove_i, add add_block.
    Returns (new_cov, success, n_redundant) where n_redundant is number of
    t-sets that become covered by >=2 blocks (redundancy created)."""
    add_cov = np.zeros(NT, dtype=bool)
    idxs = _block_tset_indices(add_block)
    add_cov[idxs] = True
    
    # New coverage = (cov without row remove_i) | add_cov
    cov_wo = cov.copy()
    cov_wo[remove_i, :] = False
    new_cov = cov_wo | add_cov
    
    # Must still be valid
    if not new_cov.any(axis=0).all():
        return None, False, 0
    
    # Count redundancy: t-sets covered by >=2 blocks in new_cov
    new_col_sums = new_cov.sum(axis=0)
    n_redundant = int((new_col_sums >= 2).sum())
    
    # Old redundancy for comparison
    old_col_sums = cov.sum(axis=0)
    old_redundant = int((old_col_sums >= 2).sum())
    
    # Net gain in redundancy (higher is better for future removals)
    gain = n_redundant - old_redundant
    return new_cov, True, gain


def _local_search(blocks, rng, time_limit):
    """Direct block-level local search with removals and swaps."""
    deadline = time.time() + time_limit
    cov = _build_coverage_matrix(blocks)
    n = len(blocks)
    
    if not _is_valid(cov):
        return blocks
    
    best_n = n
    best_blocks = [list(b) for b in blocks]
    best_cov = cov.copy()
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        n = len(blocks)
        if n <= 69:  # Lower bound
            break
        
        improved = False
        
        # --- Phase 1: Try all single removals ---
        removable = []
        for i in range(n):
            if time.time() > deadline:
                break
            col_sums = cov.sum(axis=0)
            if (col_sums[cov[i]] >= 2).all():
                removable.append(i)
        
        if removable:
            # Remove the one that creates the most redundancy (least critical)
            # Actually just remove one at random among removable
            i = rng.choice(removable)
            cov[i, :] = False
            blocks.pop(i)
            n -= 1
            improved = True
            no_improve_count = 0
            if n < best_n:
                best_n = n
                best_blocks = [list(b) for b in blocks]
                best_cov = cov.copy()
            continue
        
        # --- Phase 2: Try swaps (remove one, add one) ---
        # Pick a block to remove
        i = rng.randint(0, n - 1)
        
        # Find t-sets that would be uncovered if we remove block i
        col_sums = cov.sum(axis=0)
        critical = cov[i] & (col_sums == 1)  # t-sets only covered by block i
        n_critical = int(critical.sum())
        
        if n_critical == 0:
            # Shouldn't happen (we checked removals above), but handle
            cov[i, :] = False
            blocks.pop(i)
            n -= 1
            improved = True
            no_improve_count = 0
            if n < best_n:
                best_n = n
                best_blocks = [list(b) for b in blocks]
                best_cov = cov.copy()
            continue
        
        # Generate candidate blocks that cover all critical t-sets
        # A block covers a critical t-set {a,b,c} iff it contains a, b, c
        # So candidate must contain the union of all critical t-sets' elements
        critical_set = set()
        for j in np.where(critical)[0]:
            critical_set.update(ALL_TSETS[j])
        
        # If critical_set has more than K elements, no single block can cover all
        if len(critical_set) > K:
            no_improve_count += 1
            continue
        
        # Must contain critical_set, plus (K - len(critical_set)) more from remaining
        must_have = list(critical_set)
        optional = [x for x in range(V) if x not in critical_set]
        need_more = K - len(must_have)
        
        if need_more < 0:
            no_improve_count += 1
            continue
        
        # Try a few random completions
        best_swap_gain = -1
        best_new_cov = None
        best_add_block = None
        
        n_tries = 30
        for _ in range(n_tries):
            if time.time() > deadline:
                break
            extra = rng.sample(optional, need_more)
            cand = sorted(must_have + extra)
            cand_set = set(cand)
            
            # Verify it covers all critical (it should by construction)
            # Build its coverage
            add_cov = np.zeros(NT, dtype=bool)
            for ts in itertools.combinations(cand, T):
                add_cov[TSET_IDX[ts]] = True
            
            # New coverage
            cov_wo = cov.copy()
            cov_wo[i, :] = False
            new_cov = cov_wo | add_cov
            
            if not new_cov.any(axis=0).all():
                continue
            
            # Score: redundancy created
            new_col_sums = new_cov.sum(axis=0)
            n_redundant = int((new_col_sums >= 2).sum())
            old_col_sums = cov.sum(axis=0)
            old_redundant = int((old_col_sums >= 2).sum())
            gain = n_redundant - old_redundant
            
            if gain > best_swap_gain:
                best_swap_gain = gain
                best_new_cov = new_cov
                best_add_block = cand
        
        if best_new_cov is not None:
            cov = best_new_cov
            blocks[i] = best_add_block
            improved = True
            no_improve_count = 0
            # n stays the same, but redundancy may have increased
            # Check if now a removal is possible
            if len(blocks) < best_n:
                best_n = len(blocks)
                best_blocks = [list(b) for b in blocks]
                best_cov = cov.copy()
        else:
            no_improve_count += 1
        
        if not improved:
            no_improve_count += 1
    
    # Final: if best is better, return it
    if best_n < len(blocks):
        return best_blocks
    return blocks


def _perturbation(blocks, rng):
    """Perturb: randomly replace 2-3 blocks to escape local optima."""
    n = len(blocks)
    if n < 5:
        return blocks
    blocks = [list(b) for b in blocks]
    n_replace = rng.randint(2, 3)
    for _ in range(n_replace):
        i = rng.randint(0, n - 1)
        blocks[i] = sorted(rng.sample(range(V), K))
    return blocks


def construct(v, k, t):
    start_time = time.time()
    total_budget = 145.0  # Safety margin under 150s
    
    # Phase 1: Fast greedy initialization
    rng = random.Random(42)
    blocks = _fast_greedy(rng, 15.0)
    
    if not blocks:
        # Emergency: return a trivial covering (all 7-subsets containing {0,1,2})
        # This is too many; instead just do a quick greedy
        blocks = _fast_greedy(random.Random(1), 10.0)
    
    # Verify
    cov = _build_coverage_matrix(blocks)
    if not _is_valid(cov):
        # Fix: add blocks for uncovered t-sets
        uncovered = np.where(~cov.any(axis=0))[0]
        for idx in uncovered:
            ts = ALL_TSETS[idx]
            # Add a block containing this t-set
            remaining = [x for x in range(v) if x not in ts]
            block = sorted(list(ts) + rng.sample(remaining, K - T))
            blocks.append(block)
        cov = _build_coverage_matrix(blocks)
    
    best_blocks = [list(b) for b in blocks]
    best_n = len(blocks)
    
    # Phase 2: Local search with multiple restarts
    elapsed = time.time() - start_time
    remaining = total_budget - 10.0 - elapsed
    n_restarts = max(1, int(remaining // 20))
    per_restart = remaining / n_restarts
    
    for r in range(n_restarts):
        if time.time() > total_budget - 10.0:
            break
        
        if r > 0:
            # Perturb from best
            blocks = _perturbation(best_blocks, rng)
            # Ensure validity
            cov = _build_coverage_matrix(blocks)
            if not _is_valid(cov):
                uncovered = np.where(~cov.any(axis=0))[0]
                for idx in uncovered:
                    ts = ALL_TSETS[idx]
                    remaining_pts = [x for x in range(v) if x not in ts]
                    block = sorted(list(ts) + rng.sample(remaining_pts, K - T))
                    blocks.append(block)
        
        # Local search
        result = _local_search(blocks, rng, per_restart)
        
        if result:
            cov_r = _build_coverage_matrix(result)
            if _is_valid(cov_r) and len(result) < best_n:
                best_n = len(result)
                best_blocks = [list(b) for b in result]
    
    # Phase 3: Final polish on best
    elapsed = time.time() - start_time
    remaining = total_budget - 5.0 - elapsed
    if remaining > 3.0:
        result = _local_search(best_blocks, rng, remaining)
        if result:
            cov_r = _build_coverage_matrix(result)
            if _is_valid(cov_r) and len(result) < best_n:
                best_blocks = [list(b) for b in result]
                best_n = len(result)
    
    # Final safety check
    if time.time() > total_budget - 3.0:
        if not best_blocks:
            best_blocks = _fast_greedy(random.Random(7), 3.0)
        return best_blocks
    
    # Verify final result
    if best_blocks:
        cov_final = _build_coverage_matrix(best_blocks)
        if _is_valid(cov_final):
            return best_blocks
    
    # Fallback
    fallback = _fast_greedy(random.Random(99), 5.0)
    return fallback
