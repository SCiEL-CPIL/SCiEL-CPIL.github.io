import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count_np(residues, L_MAX):
    """
    Builds a numpy array of counts for x in [0, L_MAX].
    """
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """
    Finds the largest L such that count[1..L] >= 2.
    """
    slice_view = count[1:L_MAX+1]
    if slice_view.min() >= 2:
        return L_MAX
    first_bad = np.argmin(slice_view < 2)
    return first_bad

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # 1. Initialization
    # Strategy: Start with residues that cover small numbers well.
    # Setting r_p = 0 covers multiples of p. This is dense for small p.
    # Alternatively, r_p = 1 covers 1, p+1, etc.
    # Let's try a mix or a greedy initialization on a small window.
    
    # Greedy Initialization on window [1, 500]
    L_INIT = 500
    residues = [0] * _NPRIMES
    count = _build_count_np(residues, L_INIT)
    
    # Greedy fill for small window to ensure high coverage density
    for _ in range(_NPRIMES): # At most one pass per prime
        L_curr = _find_prefix_L_np(count, L_INIT)
        if L_curr >= L_INIT:
            break
        gap = L_curr + 1
        best_cost = float('inf')
        best_move = None
        
        # Only consider small primes for initialization to save time
        # Actually, let's consider all, but break early if we find a 0 cost
        for i, p in enumerate(_PRIMES):
            old_r = residues[i]
            new_r = gap % p
            if old_r == new_r:
                continue
            
            # Cost: how many x in [1, L_curr] with x%p==old_r have count[x]==2?
            # If we change old_r to new_r, we lose coverage on those x.
            # If count[x] was 2, it becomes 1 (bad). If count[x] was >2, it becomes >1 (ok).
            # So cost is number of x where count[x] == 2 and x % p == old_r.
            
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
                
            if start_old > L_curr:
                cost = 0
            else:
                sub_count = count[start_old:L_curr+1:p]
                cost = np.sum(sub_count == 2)
                
            if cost < best_cost:
                best_cost = cost
                best_move = (i, new_r, old_r)
                if cost == 0:
                    break
        
        if best_move is not None:
            i, new_r, old_r = best_move
            p = _PRIMES[i]
            
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
                
            count[start_old:L_INIT+1:p] -= 1
            count[start_new:L_INIT+1:p] += 1
            residues[i] = new_r
        else:
            break

    # Now expand the window and continue optimization
    L_LIMIT = 2000
    count = _build_count_np(residues, L_LIMIT)
    current_L = _find_prefix_L_np(count, L_LIMIT)
    best_L = current_L
    best_residues = list(residues)
    
    def expand_window(new_limit):
        nonlocal L_LIMIT, count, current_L
        count = _build_count_np(residues, new_limit)
        L_LIMIT = new_limit
        current_L = _find_prefix_L_np(count, L_LIMIT)

    # Main Loop: Greedy Repair with Cost Evaluation
    stagnation_count = 0
    max_stagnation = 50
    
    while time.time() < deadline - 10:
        
        # Check if we can extend L_LIMIT if we are close to it
        if current_L > L_LIMIT - 100:
            expand_window(L_LIMIT * 2)
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
        
        gap = current_L + 1
        
        if gap > L_LIMIT:
            expand_window(L_LIMIT * 2)
            continue
            
        # If the gap is already covered, extend L
        if count[gap] >= 2:
            new_L = _find_prefix_L_np(count, L_LIMIT)
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
                stagnation_count = 0
            continue
            
        # Gap is not covered (count[gap] < 2). We need to fix it.
        # Greedy Repair: Find a move to cover 'gap'.
        # We want to change residue of prime i from old_r to new_r = gap % p_i.
        # Cost: Number of x in [1, current_L] such that x % p_i == old_r and count[x] == 2.
        # We prioritize moves with cost 0. If multiple, prefer smaller primes?
        # Smaller primes have higher frequency, so changing them has larger impact.
        # But if cost is 0, it's safe.
        
        best_move = None
        min_cost = float('inf')
        
        # Optimization: Only consider primes where changing the residue actually adds coverage to 'gap'
        # And prioritize smaller primes in tie-breaking?
        
        for i, p in enumerate(_PRIMES):
            old_r = residues[i]
            new_r = gap % p
            
            if old_r == new_r:
                continue
                
            # Calculate cost
            if old_r == 0:
                start_x = p
            else:
                start_x = old_r
                
            if start_x > current_L:
                cost = 0
            else:
                sub_count = count[start_x:current_L+1:p]
                cost = np.sum(sub_count == 2)
                
            # Tie-breaking: Prefer smaller primes? 
            # Actually, if cost is 0, any is fine. 
            # If cost > 0, we want the smallest cost.
            # If cost is same, maybe prefer larger primes? 
            # Larger primes have less frequent hits, so changing them disturbs fewer future points?
            # Let's stick to min cost.
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r, old_r)
                if cost == 0:
                    break
            
        if best_move is not None:
            i, new_r, old_r = best_move
            p = _PRIMES[i]
            
            # Apply move
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
                
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
                
            # Update count array
            count[start_old:L_LIMIT+1:p] -= 1
            count[start_new:L_LIMIT+1:p] += 1
            
            residues[i] = new_r
            
            # Recalculate L
            new_L = _find_prefix_L_np(count, L_LIMIT)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
                stagnation_count = 0
            else:
                # L decreased or stayed same.
                # Revert
                count[start_new:L_LIMIT+1:p] -= 1
                count[start_old:L_LIMIT+1:p] += 1
                residues[i] = old_r
                stagnation_count += 1
        else:
            stagnation_count += 1
            
        # SA Burst if Stagnant
        if stagnation_count > max_stagnation:
            # Perform SA for a short time
            # Focus on small primes which are more critical
            T = 5.0
            cooling = 0.95
            min_T = 0.5
            sa_start_time = time.time()
            
            # SA Loop
            while time.time() < sa_start_time + 2.0 and T > min_T:
                # Bias towards small primes
                i = random.randint(0, min(50, _NPRIMES - 1))
                p = _PRIMES[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                if old_r == 0:
                    start_old = p
                else:
                    start_old = old_r
                    
                if new_r == 0:
                    start_new = p
                else:
                    start_new = new_r
                    
                # Apply
                count[start_old:L_LIMIT+1:p] -= 1
                count[start_new:L_LIMIT+1:p] += 1
                residues[i] = new_r
                
                new_L = _find_prefix_L_np(count, L_LIMIT)
                
                delta = new_L - current_L
                if delta > 0:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = list(residues)
                        stagnation_count = 0
                else:
                    if delta == 0:
                        pass
                    else:
                        prob = np.exp(delta / T)
                        if random.random() > prob:
                            # Revert
                            count[start_new:L_LIMIT+1:p] -= 1
                            count[start_old:L_LIMIT+1:p] += 1
                            residues[i] = old_r
                        else:
                            current_L = new_L
                            
                T *= cooling
            
            # Reset stagnation counter after SA
            stagnation_count = 0

    # Final Verification
    L_MAX_FINAL = 100000
    count_final = _build_count_np(best_residues, L_MAX_FINAL)
    final_L = _find_prefix_L_np(count_final, L_MAX_FINAL)
    
    return best_residues
