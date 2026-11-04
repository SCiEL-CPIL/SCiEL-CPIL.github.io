"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    max_U = max(U)
    if max_U == 0:
        return 0.0
    denom = math.log(2 * max_U + 1)
    if denom <= 0:
        return 0.0
    # Compute U+U and U-U sizes
    # For efficiency, use sets
    sumset = set()
    diffset = set()
    # Use numpy for faster computation if possible
    arr = np.array(U)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    if sumset_size == 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _score_set_fast(U):
    """Faster scoring using bitsets or numpy, with early termination."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    max_U = max(U)
    if max_U == 0:
        return 0.0
    denom = math.log(2 * max_U + 1)
    if denom <= 0:
        return 0.0
    
    # Use numpy for vectorized sum/diff computation
    arr = np.array(U)
    n = len(arr)
    
    # For small sets, direct computation is fine
    # For larger sets, we need to be careful about memory
    if n > 5000:
        # Sample-based estimate for very large sets
        sample_size = min(5000, n)
        idx = np.random.choice(n, sample_size, replace=False)
        sample = arr[idx]
        sums = sample[:, None] + sample[None, :]
        diffs = sample[:, None] - sample[None, :]
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
    else:
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _geometric_seed(base, power, n_terms):
    """Generate a geometric-like seed set."""
    U = [0]
    val = 1
    for i in range(n_terms - 1):
        U.append(val)
        val *= base
        if val > 10**9:
            break
    return U


def _ldpc_inspired_seed(n_bits, rate, seed=42):
    """
    Generate a set inspired by LDPC code structure.
    We select codewords from a sparse parity-check structure.
    The idea: use a random sparse generator matrix to create
    a set of integers with good difference/sum properties.
    """
    rng = random.Random(seed)
    
    # Create a sparse "generator" structure
    # Each codeword is a sum of selected basis elements
    n_basis = n_bits
    sparsity = max(2, int(rate * n_bits))
    
    # Generate random sparse vectors
    basis_elements = [1 << i for i in range(min(n_bits, 60))]
    
    U = [0]
    seen = {0}
    
    for _ in range(n_bits * 2):
        # Pick a random sparse combination
        k = rng.randint(1, sparsity)
        idxs = rng.sample(range(len(basis_elements)), min(k, len(basis_elements)))
        val = 0
        for i in idxs:
            val ^= basis_elements[i]  # XOR to mimic linear code over F2
        
        if val not in seen and val > 0:
            U.append(val)
            seen.add(val)
    
    # Sort for consistency
    U.sort()
    return U


def _random_diverse_seed(n, max_val, seed=42):
    """Generate a random set with diverse spacing."""
    rng = random.Random(seed)
    U = [0]
    remaining = n - 1
    current_max = max_val
    
    # Use a mixture of geometric and random steps
    for i in range(remaining):
        if rng.random() < 0.3:
            # Geometric jump
            jump = rng.randint(1, max(1, current_max // 3))
        else:
            # Random jump
            jump = rng.randint(1, max(1, current_max // max(1, i + 1)))
        
        U.append(U[-1] + jump)
        if U[-1] >= max_val:
            break
    
    return U


def _local_search(U, max_iter=200, time_budget=30.0, seed=42):
    """
    Local search: try adding/removing elements to improve the score.
    """
    rng = random.Random(seed)
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    U.sort()
    
    current_score = _score_set_fast(U)
    best_score = current_score
    best_U = U[:]
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Try modifications
        candidates = []
        
        # Try adding a new element
        max_U = max(U) if U else 0
        for _ in range(5):
            new_val = rng.randint(1, max(1, int(max_U * 1.5)))
            if new_val not in U:
                candidates.append(('add', new_val))
        
        # Try removing an element (not 0)
        removable = [x for x in U if x != 0]
        if removable:
            for _ in range(3):
                candidates.append(('remove', rng.choice(removable)))
        
        # Try replacing an element
        removable = [x for x in U if x != 0]
        if removable:
            for _ in range(3):
                old = rng.choice(removable)
                new_val = rng.randint(1, max(1, int(max_U * 1.5)))
                if new_val not in U:
                    candidates.append(('replace', old, new_val))
        
        improved = False
        for cand in candidates:
            if time.time() - start_time > time_budget:
                break
            
            if cand[0] == 'add':
                new_U = U + [cand[1]]
            elif cand[0] == 'remove':
                new_U = [x for x in U if x != cand[1]]
            elif cand[0] == 'replace':
                new_U = [x for x in U if x != cand[1]] + [cand[2]]
            
            if not new_U or 0 not in new_U:
                continue
            
            new_score = _score_set_fast(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U[:]
                current_score = new_score
                U = new_U
                improved = True
                break
        
        if not improved:
            # Try a larger perturbation
            if U and len(U) > 2:
                # Remove a random element and add two new ones
                removable = [x for x in U if x != 0]
                if removable:
                    old = rng.choice(removable)
                    new_U = [x for x in U if x != old]
                    max_U = max(new_U) if new_U else 0
                    for _ in range(2):
                        nv = rng.randint(1, max(1, int(max_U * 1.3)))
                        if nv not in new_U:
                            new_U.append(nv)
                    if 0 in new_U:
                        new_score = _score_set_fast(new_U)
                        if new_score > best_score + 1e-12:
                            best_score = new_score
                            best_U = new_U[:]
                            current_score = new_score
                            U = new_U
                            improved = True
    
    return best_U, best_score


def _hill_climbing(U, max_iter=100, time_budget=20.0, seed=42):
    """
    More aggressive hill climbing with structured moves.
    """
    rng = random.Random(seed)
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    U.sort()
    
    current_score = _score_set_fast(U)
    best_score = current_score
    best_U = U[:]
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Structured move: adjust one element by a small delta
        if len(U) > 1:
            idx = rng.randint(1, len(U) - 1)  # Skip 0
            old_val = U[idx]
            
            # Try several deltas
            for delta in [1, -1, 2, -2, 3, -3, 5, -5, 10, -10]:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if new_val in U:
                    continue
                
                new_U = U[:]
                new_U[idx] = new_val
                new_U.sort()
                
                new_score = _score_set_fast(new_U)
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_U = new_U[:]
                    current_score = new_score
                    U = new_U
                    break
        
        # Occasionally try adding or removing
        if rng.random() < 0.2:
            max_U = max(U) if U else 0
            new_val = rng.randint(1, max(1, int(max_U * 1.2)))
            if new_val not in U:
                new_U = U + [new_val]
                new_score = _score_set_fast(new_U)
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_U = new_U[:]
                    current_score = new_score
                    U = new_U
        elif rng.random() < 0.1 and len(U) > 2:
            removable = [x for x in U if x != 0]
            if removable:
                old = rng.choice(removable)
                new_U = [x for x in U if x != old]
                new_score = _score_set_fast(new_U)
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_U = new_U[:]
                    current_score = new_score
                    U = new_U
    
    return best_U, best_score


def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound.
    
    Uses a combination of:
    1. LDPC-inspired seed generation
    2. Geometric seeds
    3. Local search and hill climbing
    4. Deadline guard for 150s wall-clock limit
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Phase 1: Generate diverse seeds
    seeds = []
    
    # Geometric seeds
    for base in [2, 3, 4, 5]:
        for n_terms in [5, 8, 12, 15]:
            seed = _geometric_seed(base, 1, n_terms)
            if 0 in seed:
                seeds.append(seed)
    
    # LDPC-inspired seeds
    for n_bits in [10, 15, 20, 25]:
        for rate in [0.3, 0.5, 0.7]:
            for seed_val in [42, 123, 456]:
                seed = _ldpc_inspired_seed(n_bits, rate, seed_val)
                if 0 in seed and len(seed) > 1:
                    seeds.append(seed)
    
    # Random diverse seeds
    for n in [10, 20, 30, 50]:
        for max_val in [1000, 10000, 100000]:
            for seed_val in [42, 123, 456]:
                seed = _random_diverse_seed(n, max_val, seed_val)
                if 0 in seed and len(seed) > 1:
                    seeds.append(seed)
    
    # Evaluate all seeds
    for seed in seeds:
        if time.time() - start_time > deadline - 30:
            break
        score = _score_set_fast(seed)
        if score > best_score:
            best_score = score
            best_U = seed[:]
    
    # Phase 2: Local search on best seed
    if time.time() - start_time < deadline - 20:
        improved_U, improved_score = _local_search(
            best_U, max_iter=150, time_budget=min(20.0, deadline - (time.time() - start_time) - 10), seed=42
        )
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 3: Hill climbing on best result
    if time.time() - start_time < deadline - 15:
        improved_U, improved_score = _hill_climbing(
            best_U, max_iter=100, time_budget=min(15.0, deadline - (time.time() - start_time) - 5), seed=123
        )
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 4: Try a few more random restarts if time permits
    if time.time() - start_time < deadline - 10:
        for restart_seed in [789, 1011, 2023]:
            if time.time() - start_time > deadline - 5:
                break
            seed = _random_diverse_seed(20, 10000, restart_seed)
            if 0 in seed:
                score = _score_set_fast(seed)
                if score > best_score:
                    best_score = score
                    best_U = seed[:]
                
                if time.time() - start_time < deadline - 5:
                    improved_U, improved_score = _local_search(
                        best_U, max_iter=50, time_budget=min(5.0, deadline - (time.time() - start_time) - 2), seed=restart_seed
                    )
                    if improved_score > best_score:
                        best_score = improved_score
                        best_U = improved_U
    
    # Ensure 0 is in the result
    if 0 not in best_U:
        best_U.append(0)
    best_U = list(set(best_U))
    best_U.sort()
    
    return best_U

# EVOLVE-BLOCK-END
