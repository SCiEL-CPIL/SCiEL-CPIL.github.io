import time
import random
import sys

def _check_sidon_unique_sums(els):
    """Check that all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_nonzero_diffs(els):
    """Get set of nonzero differences (positive and negative)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = els[j] - els[i]
                if d != 0:
                    diffs.add(d)
    return diffs

def _is_valid_pair(A, B):
    """Check that A and B are Sidon and their nonzero difference sets are disjoint."""
    if not A or not B:
        return False
    if not _check_sidon_unique_sums(A):
        return False
    if not _check_sidon_unique_sums(B):
        return False
    diffsA = _get_nonzero_diffs(A)
    diffsB = _get_nonzero_diffs(B)
    if diffsA & diffsB:
        return False
    return True

def _build_sidon_with_diff_constraint(initial_set, forbidden_diffs, n=256, max_size=64, rng=None):
    """
    Greedily extend a Sidon set while avoiding forbidden differences.
    Returns the extended set.
    """
    if rng is None:
        rng = random.Random()
    
    current = list(initial_set)
    current.sort()
    used_sums = set()
    for i in range(len(current)):
        for j in range(i, len(current)):
            used_sums.add(current[i] + current[j])
    
    current_diffs = _get_nonzero_diffs(current)
    
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(current) >= max_size:
            break
        if x in current:
            continue
        
        # Check Sidon property
        valid = True
        for a in current:
            if (a + x) in used_sums:
                valid = False
                break
        if valid and (2 * x) in used_sums:
            valid = False
        if not valid:
            continue
        
        # Check difference constraints
        new_diffs = set()
        for a in current:
            d1 = x - a
            d2 = a - x
            new_diffs.add(d1)
            new_diffs.add(d2)
        
        # Must not conflict with forbidden diffs (from other set)
        if new_diffs & forbidden_diffs:
            continue
        
        # Must not conflict with current set's own diffs
        if new_diffs & current_diffs:
            continue
        
        # Accept x
        current.append(x)
        for a in current[:-1]:
            used_sums.add(a + x)
        used_sums.add(2 * x)
        current_diffs |= new_diffs
    
    return current

def _number_theoretic_seed(n=256, rng=None):
    """
    Generate initial Sidon sets using number-theoretic construction.
    Use quadratic residues modulo a prime to get structured sets.
    """
    if rng is None:
        rng = random.Random()
    
    # Find a prime p > n for modular construction
    # Use p = 257 (next prime after 256)
    p = 257
    
    # Generate quadratic residues mod p
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    residues.discard(0)
    
    # Map residues to [1, 256]
    # Take a subset of residues and map them
    res_list = sorted(residues)
    
    # For set A: use smaller residues
    # For set B: use larger residues with offset
    # This helps ensure disjoint differences
    
    # Strategy: A gets residues in [1, 128], B gets residues in [129, 256] mapped
    # But we need to ensure they fit in [1, 256] and are Sidon
    
    # Simpler: use two different modular constructions
    # A: {x : x^2 mod p in some range}
    # B: {x : x^2 mod p in another range}
    
    # Actually, let's use a different approach:
    # Pick a base Sidon set S mod p, then scale/shift to get A and B
    
    # Known: {i * p + f(i) for i in range} can be Sidon if f is chosen well
    # For simplicity, use the fact that {x^2 mod p} for x in [1, (p-1)/2] gives (p-1)/2 distinct values
    
    half = (p - 1) // 2  # 128
    quad_res = [(x * x) % p for x in range(1, half + 1)]
    quad_res = [r for r in quad_res if 1 <= r <= n]
    
    # Split into two groups
    mid = len(quad_res) // 2
    A_init = sorted(quad_res[:mid])
    B_init = sorted(quad_res[mid:])
    
    # Ensure non-empty
    if not A_init:
        A_init = [1]
    if not B_init:
        B_init = [2]
    
    return A_init, B_init

def _local_search_improve(A, B, n=256, max_iters=50, rng=None):
    """Try to improve by adding/removing/swapping elements."""
    if rng is None:
        rng = random.Random()
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding a random element to A
        for _attempt in range(30):
            x = rng.randint(1, n)
            if x in best_A:
                continue
            new_A = sorted(best_A + [x])
            if len(new_A) > 64:
                continue
            if _is_valid_pair(new_A, best_B):
                best_A = new_A
                best_size = len(A) + len(B)
                improved = True
                break
        
        # Try adding a random element to B
        for _attempt in range(30):
            x = rng.randint(1, n)
            if x in best_B:
                continue
            new_B = sorted(best_B + [x])
            if len(new_B) > 64:
                continue
            if _is_valid_pair(best_A, new_B):
                best_B = new_B
                best_size = len(A) + len(B)
                improved = True
                break
        
        # Try swapping: remove one from A, add to B (and vice versa)
        if len(best_A) > 1:
            for _attempt in range(5):
                idx = rng.randint(0, len(best_A) - 1)
                x = best_A[idx]
                new_A = [a for i, a in enumerate(best_A) if i != idx]
                if x not in best_B:
                    new_B = sorted(best_B + [x])
                    if len(new_B) <= 64 and _is_valid_pair(new_A, new_B):
                        best_A = new_A
                        best_B = new_B
                        best_size = len(A) + len(B)
                        improved = True
                        break
        
        if len(best_B) > 1:
            for _attempt in range(5):
                idx = rng.randint(0, len(best_B) - 1)
                x = best_B[idx]
                new_B = [b for i, b in enumerate(best_B) if i != idx]
                if x not in best_A:
                    new_A = sorted(best_A + [x])
                    if len(new_A) <= 64 and _is_valid_pair(new_A, new_B):
                        best_A = new_A
                        best_B = new_B
                        best_size = len(A) + len(B)
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_A, best_B

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # Safety margin within 90s limit
    n = 256
    max_size = 64
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    rng = random.Random()
    
    # Phase 1: Number-theoretic construction with multiple primes
    primes = [257, 263, 269, 271, 277, 281, 283]
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # Generate quadratic residues mod p
        half = (p - 1) // 2
        quad_res = set()
        for x in range(1, p):
            r = (x * x) % p
            if 1 <= r <= n:
                quad_res.add(r)
        
        quad_res_list = sorted(quad_res)
        
        if len(quad_res_list) < 4:
            continue
        
        # Try different splits
        for split_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if time.time() > deadline:
                break
            
            split_idx = int(len(quad_res_list) * split_frac)
            if split_idx < 1 or split_idx >= len(quad_res_list):
                continue
            
            A_init = quad_res_list[:split_idx]
            B_init = quad_res_list[split_idx:]
            
            # Extend both sets greedily
            B_diffs = _get_nonzero_diffs(B_init)
            A = _build_sidon_with_diff_constraint(A_init, B_diffs, n, max_size, rng)
            
            A_diffs = _get_nonzero_diffs(A)
            B = _build_sidon_with_diff_constraint(B_init, A_diffs, n, max_size, rng)
            
            if not A:
                A = [1]
            if not B:
                B = [2]
            
            if _is_valid_pair(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_A = list(A)
                    best_B = list(B)
                
                # Local search improvement
                if time.time() < deadline - 5:
                    A_ls, B_ls = _local_search_improve(A, B, n, max_iters=30, rng=rng)
                    if _is_valid_pair(A_ls, B_ls):
                        score_ls = len(A_ls) + len(B_ls)
                        if score_ls > best_score:
                            best_score = score_ls
                            best_A = list(A_ls)
                            best_B = list(B_ls)
    
    # Phase 2: Randomized greedy restarts (keep from original approach)
    for restart in range(15000):
        if time.time() > deadline:
            break
        
        # Strategy 1: Random seed for A, then greedy B, then greedy A
        if restart % 3 == 0:
            seed_a = rng.randint(1, n)
            A_init = [seed_a]
            B_diffs = _get_nonzero_diffs(A_init)
            B = _build_sidon_with_diff_constraint([], B_diffs, n, max_size, rng)
            A_diffs = _get_nonzero_diffs(B)
            A = _build_sidon_with_diff_constraint(A_init, A_diffs, n, max_size, rng)
        # Strategy 2: Random seed for B, then greedy A, then greedy B
        elif restart % 3 == 1:
            seed_b = rng.randint(1, n)
            B_init = [seed_b]
            A_diffs = _get_nonzero_diffs(B_init)
            A = _build_sidon_with_diff_constraint([], A_diffs, n, max_size, rng)
            B_diffs = _get_nonzero_diffs(A)
            B = _build_sidon_with_diff_constraint(B_init, B_diffs, n, max_size, rng)
        # Strategy 3: Random small sets, then extend
        else:
            size_a = rng.randint(1, 5)
            size_b = rng.randint(1, 5)
            A_init = sorted(rng.sample(range(1, n + 1), size_a))
            B_init = sorted(rng.sample(range(1, n + 1), size_b))
            if not _is_valid_pair(A_init, B_init):
                continue
            B_diffs = _get_nonzero_diffs(B_init)
            A = _build_sidon_with_diff_constraint(A_init, B_diffs, n, max_size, rng)
            A_diffs = _get_nonzero_diffs(A)
            B = _build_sidon_with_diff_constraint(B_init, A_diffs, n, max_size, rng)
        
        if not A:
            A = [1]
        if not B:
            B = [2]
        
        if _is_valid_pair(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
            
            # Local search improvement
            if time.time() < deadline - 5:
                A_ls, B_ls = _local_search_improve(A, B, n, max_iters=20, rng=rng)
                if _is_valid_pair(A_ls, B_ls):
                    score_ls = len(A_ls) + len(B_ls)
                    if score_ls > best_score:
                        best_score = score_ls
                        best_A = list(A_ls)
                        best_B = list(B_ls)
    
    # Final check
    if not _is_valid_pair(best_A, best_B):
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
