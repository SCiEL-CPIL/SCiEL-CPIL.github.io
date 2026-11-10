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
        # We want to increment count at indices r, r+p, r+2p, ... <= L_MAX
        # If r == 0, we start at p (since x>=1 and x% p == 0 means x is multiple of p, 
        # but the congruence is x = r mod p. If r=0, x can be p, 2p... 
        # The problem says 0 <= r_p < p. 
        # If r=0, x=0 is not in [1..L]. x=p is the first positive.
        # If r>0, x=r is the first positive (if r>=1).
        
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
    # We need the first index x in [1, L_MAX] where count[x] < 2.
    # If all are >= 2, return L_MAX.
    # np.nonzero(count[1:] < 2)[0] gives indices relative to count[1:].
    # Let's just loop, it's O(L) and L is not that huge for the check.
    # Or use numpy:
    # bad_indices = np.nonzero(count[1:L_MAX+1] < 2)[0]
    # if len(bad_indices) == 0: return L_MAX
    # else: return bad_indices[0]
    
    # Numpy approach is faster for large L
    slice_view = count[1:L_MAX+1]
    if slice_view.min() >= 2:
        return L_MAX
    first_bad = np.argmin(slice_view < 2)
    return first_bad
    
def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # 1. Initialization
    # Use the "all 1s" strategy from the inspiration, which is strong for early coverage.
    residues = [1 % p for p in _PRIMES]
    
    # Work with a dynamic window. Start with a moderate size.
    # 5000 is a good start. We will expand as needed.
    L_LIMIT = 5000
    count = _build_count_np(residues, L_LIMIT)
    current_L = _find_prefix_L_np(count, L_LIMIT)
    
    best_L = current_L
    best_residues = list(residues)
    
    # Helper to expand window if we hit the limit
    def expand_window():
        nonlocal L_LIMIT, count, current_L
        new_limit = L_LIMIT * 2
        count = _build_count_np(residues, new_limit)
        L_LIMIT = new_limit
        current_L = _find_prefix_L_np(count, L_LIMIT)
        
    # Main Loop: Hybrid Greedy + SA
    stagnation_count = 0
    max_stagnation = 20 # Lower threshold for SA to kick in, as NumPy is fast
    
    while time.time() < deadline - 10:
        
        # Check if we can extend L_LIMIT if we are close to it
        if current_L > L_LIMIT - 100:
            expand_window()
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
        
        gap = current_L + 1
        
        if gap > L_LIMIT:
            expand_window()
            continue
            
        # If the gap is already covered, extend L
        if count[gap] >= 2:
            # Find how far we can extend
            # This is effectively re-running _find_prefix_L_np but starting from gap
            # Since count[gap] >= 2, we can just call _find_prefix_L_np
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
        # If cost == 0, it's a "safe" move.
        
        best_move = None
        min_cost = float('inf')
        
        for i, p in enumerate(_PRIMES):
            old_r = residues[i]
            new_r = gap % p
            
            if old_r == new_r:
                continue
                
            # Calculate cost
            # x in [1, current_L] such that x % p == old_r
            # start_x = old_r if old_r != 0 else p
            if old_r == 0:
                start_x = p
            else:
                start_x = old_r
                
            if start_x > current_L:
                cost = 0
            else:
                # Use NumPy to count how many x in [start_x, current_L] with step p have count[x] == 2
                # Slice: count[start_x:current_L+1:p]
                # We need to check if these are == 2
                # Note: The slice might be empty if start_x > current_L, handled above.
                sub_count = count[start_x:current_L+1:p]
                cost = np.sum(sub_count == 2)
                
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r, old_r)
                if cost == 0:
                    # Found a safe move. We can break early? 
                    # It's better to check if there's a "better" safe move? 
                    # All safe moves have cost 0. 
                    # Does prime size matter? Smaller primes hit more frequently, 
                    # so moving a small prime might have higher "future" impact?
                    # But cost is 0, so it doesn't hurt current L.
                    # Let's just take the first safe move to save time.
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
            T = 5.0
            cooling = 0.99
            min_T = 0.5
            sa_start_time = time.time()
            
            # SA Loop
            while time.time() < sa_start_time + 3.0 and T > min_T:
                i = random.randint(0, _NPRIMES - 1)
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
    # The score is the largest L up to 100,000.
    # We need to check if best_residues can go further than our L_LIMIT.
    # Rebuild count for 100,000.
    
    L_MAX_FINAL = 100000
    count_final = _build_count_np(best_residues, L_MAX_FINAL)
    final_L = _find_prefix_L_np(count_final, L_MAX_FINAL)
    
    return best_residues
