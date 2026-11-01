"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Greedy construction with Lagrangian relaxation (dual-weighted gain) +
aggressive local search with Counter-based coverage tracking and smart swap candidates.
"""
import time
import itertools
import random
from collections import Counter

_DEADLINE = 115.0  # seconds, safety margin under 150s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets as ints, and for every k-subset its coverage bitmap."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    # coverage: list of sets of t-indices for each k-subset
    coverage = []
    for ks in k_subsets:
        cov = set()
        for ts in itertools.combinations(ks, t):
            cov.add(t_index[ts])
        coverage.append(cov)
    return all_t, t_index, k_subsets, coverage


def _greedy_construct(v, k, t, rng, use_lagrangian=True):
    """Greedy set cover with optional Lagrangian dual-weighted gain."""
    all_t, t_index, k_subsets, coverage = _precompute(v, k, t)
    n_t = len(all_t)
    uncovered = set(range(n_t))
    blocks = []
    
    # Dual weights for Lagrangian relaxation
    # w[i] represents the penalty for t-subset i being uncovered
    # Initially uniform, updated via subgradient descent
    if use_lagrangian:
        w = [1.0] * n_t
        step_size = 0.1
    else:
        w = None
    
    iter_count = 0
    while uncovered and _elapsed() < _DEADLINE - 25:
        best_i = -1
        best_gain = 0
        
        if use_lagrangian:
            # Weighted gain: sum of dual weights for newly covered t-subsets
            for i, cov in enumerate(coverage):
                g = 0.0
                for idx in cov:
                    if idx in uncovered:
                        g += w[idx]
                if g > best_gain:
                    best_gain = g
                    best_i = i
        else:
            for i, cov in enumerate(coverage):
                g = len(cov & uncovered)
                if g > best_gain:
                    best_gain = g
                    best_i = i
        
        if best_i < 0:
            break
        
        blocks.append(best_i)
        uncovered -= coverage[best_i]
        
        # Update dual weights via subgradient descent
        if use_lagrangian:
            # Subgradient: for each t-subset, if uncovered, increase weight; else decrease
            for idx in range(n_t):
                if idx in uncovered:
                    w[idx] += step_size
                else:
                    w[idx] = max(0.1, w[idx] - step_size * 0.5)
            # Decay step size
            step_size *= 0.98
        
        iter_count += 1
    
    return blocks, all_t, t_index, k_subsets, coverage


def _local_search(blocks, coverage, n_t, k_subsets, rng, time_budget):
    """Removal + swap local search with Counter-based coverage tracking."""
    blocks = list(blocks)
    
    # Build coverage counter
    cov_count = Counter()
    for b in blocks:
        for idx in coverage[b]:
            cov_count[idx] += 1
    
    def is_valid():
        return all(cov_count[idx] > 0 for idx in range(n_t))
    
    def is_redundant(b):
        for idx in coverage[b]:
            if cov_count[idx] == 1:
                return False
        return True
    
    def remove_block(b):
        blocks.remove(b)
        for idx in coverage[b]:
            cov_count[idx] -= 1
    
    def add_block(b):
        blocks.append(b)
        for idx in coverage[b]:
            cov_count[idx] += 1
    
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            if is_redundant(b):
                remove_block(b)
                improved = True
                continue
            i += 1
        
        # --- Swap pass: try to replace blocks with better ones ---
        for _ in range(50):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            old_b = blocks[i]
            
            # Find critical t-subsets for old_b
            critical = []
            for idx in coverage[old_b]:
                if cov_count[idx] == 1:
                    critical.append(idx)
            
            if not critical:
                # old_b is redundant, should have been removed
                remove_block(old_b)
                improved = True
                continue
            
            # Find candidates that cover all critical t-subsets
            candidates = None
            for idx in critical:
                # Find all blocks covering this t-subset
                covering_blocks = [j for j in range(len(k_subsets)) if idx in coverage[j]]
                if candidates is None:
                    candidates = set(covering_blocks)
                else:
                    candidates.intersection_update(covering_blocks)
            
            if candidates:
                candidates -= set(blocks)
                if candidates:
                    # Pick a random candidate
                    new_b = rng.choice(list(candidates))
                    # Tentatively swap
                    blocks[i] = new_b
                    # Update coverage
                    for idx in coverage[old_b]:
                        cov_count[idx] -= 1
                    for idx in coverage[new_b]:
                        cov_count[idx] += 1
                    
                    # Check if valid and if removal is now possible
                    if is_valid():
                        # Try to remove redundant blocks
                        removed_any = False
                        for j in range(len(blocks)):
                            if is_redundant(blocks[j]):
                                remove_block(blocks[j])
                                removed_any = True
                        if removed_any:
                            improved = True
                        else:
                            # Keep the swap even if no removal (might help later)
                            pass
                    else:
                        # Revert
                        blocks[i] = old_b
                        for idx in coverage[new_b]:
                            cov_count[idx] -= 1
                        for idx in coverage[old_b]:
                            cov_count[idx] += 1
        
        # --- Add pass: if not covered, add best block ---
        if not is_valid():
            missing = [idx for idx in range(n_t) if cov_count[idx] == 0]
            if missing:
                best_i = -1
                best_g = 0
                for idx, c in enumerate(coverage):
                    g = len(c & set(missing))
                    if g > best_g:
                        best_g = g
                        best_i = idx
                if best_i >= 0:
                    add_block(best_i)
    
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)
    
    all_t, t_index, k_subsets, coverage = _precompute(v, k, t)
    n_t = len(all_t)
    
    # Try both standard greedy and Lagrangian-weighted greedy
    blocks_standard, _, _, _, _ = _greedy_construct(v, k, t, rng, use_lagrangian=False)
    blocks_lag, _, _, _, _ = _greedy_construct(v, k, t, rng, use_lagrangian=True)
    
    # Pick the better initial solution
    if len(blocks_lag) < len(blocks_standard):
        blocks = blocks_lag
    else:
        blocks = blocks_standard
    
    # Local search
    time_budget = _DEADLINE - 5
    blocks = _local_search(blocks, coverage, n_t, k_subsets, rng, time_budget)
    
    # Safety: ensure valid; if not, do a quick repair
    cov = set()
    for b in blocks:
        cov |= coverage[b]
    if len(cov) < n_t:
        while len(cov) < n_t and _elapsed() < _DEADLINE - 2:
            missing = set(range(n_t)) - cov
            best_i = -1
            best_g = 0
            for idx, c in enumerate(coverage):
                g = len(c & missing)
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i < 0:
                break
            blocks.append(best_i)
            cov |= coverage[best_i]
    
    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in blocks]
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
