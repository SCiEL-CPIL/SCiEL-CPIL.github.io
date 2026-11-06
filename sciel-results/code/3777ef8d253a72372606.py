"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _sumset(A):
    """Compute A+A with equal summands allowed."""
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] + A[j])
    return s

def _prodset(A):
    """Compute A*A with equal factors allowed."""
    s = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s.add(A[i] * A[j])
    return s

def _score(A):
    """max(|A+A|, |A*A|)."""
    return max(len(_sumset(A)), len(_prodset(A)))

def _greedy_build(pool, k, rng, max_iters=5000):
    """
    Greedy construction: start with a small seed, then greedily add elements
    from pool that minimize the resulting max(sumset, prodset) size.
    """
    # Start with a random seed of 3 elements
    seed_size = 3
    current = set(rng.sample(pool, seed_size))
    
    for _ in range(k - seed_size):
        best_elem = None
        best_score = float('inf')
        best_ss = None
        best_ps = None
        
        # Precompute current sumset and prodset for scoring
        cur_ss = _sumset(sorted(current))
        cur_ps = _prodset(sorted(current))
        base_score = max(len(cur_ss), len(cur_ps))
        
        # Try adding each candidate from pool
        candidates = [x for x in pool if x not in current]
        # Shuffle for tie-breaking randomness
        rng.shuffle(candidates)
        
        for x in candidates:
            # Compute new sumset additions
            new_sums = set()
            for a in current:
                new_sums.add(a + x)
            new_sums.add(x + x)
            
            # Compute new prodset additions
            new_prods = set()
            for a in current:
                new_prods.add(a * x)
            new_prods.add(x * x)
            
            new_ss_size = len(cur_ss | new_sums)
            new_ps_size = len(cur_ps | new_prods)
            new_score = max(new_ss_size, new_ps_size)
            
            if new_score < best_score:
                best_score = new_score
                best_elem = x
                best_ss = cur_ss | new_sums
                best_ps = cur_ps | new_prods
        
        if best_elem is None:
            break
        
        current.add(best_elem)
    
    return sorted(current), best_score

def _local_search(A, pool, rng, max_swaps=2000, timeout=None):
    """
    Local search via swap moves: replace one element with another from pool.
    """
    A = list(A)
    current_score = _score(A)
    best_A = list(A)
    best_score = current_score
    k = len(A)
    
    # Precompute current sumset and prodset
    cur_ss = _sumset(A)
    cur_ps = _prodset(A)
    
    swap_count = 0
    start_time = time.time()
    
    while swap_count < max_swaps:
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        improved = False
        # Try each position for removal
        for i in range(k):
            old_elem = A[i]
            # Try replacing with candidates from pool
            candidates = [x for x in pool if x not in A]
            rng.shuffle(candidates)
            
            for new_elem in candidates:
                # Quick check: if new_elem is too large, products may explode
                # Compute new sumset
                new_A = A[:i] + [new_elem] + A[i+1:]
                
                # Compute incremental changes
                # Remove old contributions
                # Add new contributions
                
                # For efficiency, compute full sets only for promising candidates
                # First, a quick heuristic check
                new_ss = _sumset(new_A)
                new_ps = _prodset(new_A)
                new_score = max(len(new_ss), len(new_ps))
                
                if new_score < current_score:
                    # Accept the swap
                    A[i] = new_elem
                    current_score = new_score
                    cur_ss = new_ss
                    cur_ps = new_ps
                    improved = True
                    break
            
            if improved:
                swap_count += 1
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(A)
                # Continue searching from improved state
                continue
        
        if not improved:
            # No single swap improves; try random perturbations
            # Pick a random element to remove and a random candidate to add
            for _ in range(10):
                i = rng.randrange(k)
                old_elem = A[i]
                candidates = [x for x in pool if x not in A]
                if not candidates:
                    break
                new_elem = rng.choice(candidates)
                new_A = A[:i] + [new_elem] + A[i+1:]
                new_score = _score(new_A)
                
                # Accept with some probability if not worse (simulated annealing lite)
                if new_score <= current_score + 2:
                    A[i] = new_elem
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = list(A)
        
        if not improved:
            break
    
    return best_A, best_score

def construct():
    """
    Construct 20 distinct positive integers <= 10000 minimizing
    max(|A+A|, |A*A|).
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Pool of candidate integers
    # Focus on smaller numbers to keep products manageable
    # Numbers 1-200 give products up to 40000, sums up to 400
    # Numbers 1-100 give products up to 10000, sums up to 200
    # Let's use a range that balances both
    pool = list(range(1, 301))
    
    best_A = None
    best_score = float('inf')
    
    # Multiple restarts with different random seeds
    rng = random.Random(42)
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng_iter = random.Random(rng.randint(0, 10**9))
        
        # Greedy build
        A, greedy_score = _greedy_build(pool, 20, rng_iter)
        
        if greedy_score < best_score:
            best_score = greedy_score
            best_A = list(A)
        
        # Local search refinement
        remaining = deadline - time.time()
        if remaining > 5:
            timeout_ls = min(remaining - 2, 30)  # Cap local search time
            A_ls, ls_score = _local_search(A, pool, rng_iter, max_swaps=1000, timeout=timeout_ls)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = list(A_ls)
        
        if time.time() > deadline - 10:
            break
    
    # Final check
    if best_A is None:
        return list(range(1, 21))
    
    # Verify validity
    assert len(best_A) == 20
    assert len(set(best_A)) == 20
    assert all(1 <= x <= 10000 for x in best_A)
    
    return best_A
# EVOLVE-BLOCK-END
