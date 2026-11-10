import time
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count(residues, L_MAX):
    count = [0] * (L_MAX + 1)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    return count

def _find_prefix_L(count, L_MAX):
    L = 0
    for x in range(1, L_MAX + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Initial Construction: Random or Greedy
    # Let's start with a random initialization to avoid bias, then optimize.
    # Or better, start with a "dense" initialization.
    # Since we need double coverage, small primes are very useful.
    # Let's initialize residues such that they cover small numbers.
    # Simple strategy: Set r_p = 1 for all p. This covers x=1, p+1, etc.
    # Actually, let's use a smarter seed:
    # For each prime p, choose r_p randomly.
    random.seed(42)
    residues = [random.randint(0, p - 1) for p in _PRIMES]
    
    # We will work with a dynamic window. Start with a smaller one for speed, expand later.
    # But for the "Safe Move" logic, we need to know the state up to current L.
    # Let's maintain a count array up to a generous limit, say 20000, initially.
    # If we exceed this, we expand.
    
    L_LIMIT = 20000
    count = _build_count(residues, L_LIMIT)
    current_L = _find_prefix_L(count, L_LIMIT)
    
    best_L = current_L
    best_residues = list(residues)
    
    # Main Loop: Hybrid Greedy + SA
    # We alternate between Greedy Repair and SA bursts.
    # Greedy Repair: Try to fix the first gap.
    # SA Burst: If stuck, randomize slightly to escape.
    
    stagnation_count = 0
    max_stagnation = 50 # If no improvement in 50 greedy steps, do SA
    
    while time.time() < deadline - 10:
        # Check if we can extend L_LIMIT if we are close to it
        if current_L > L_LIMIT - 1000:
            new_limit = L_LIMIT * 2
            count = _build_count(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L(count, L_LIMIT)
        
        # 1. Greedy Repair Phase
        improved = False
        gap = current_L + 1
        
        if gap > L_LIMIT:
            # Expand window
            new_limit = L_LIMIT * 2
            count = _build_count(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L(count, L_LIMIT)
            continue
            
        if count[gap] >= 2:
            # Extend L
            while gap <= L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            improved = True
            stagnation_count = 0
            continue
            
        # Find a move to cover 'gap'
        # We want to change residue of some prime p_i from r_i to r_new = gap % p_i
        # Cost: Number of x in [1, current_L] such that x % p_i == r_i and count[x] == 2
        # If cost == 0, it's a "safe" move (preserves L).
        # If cost > 0, it breaks L. We prefer safe moves.
        # Among safe moves, we might prefer smaller p (higher frequency) or just any.
        # If no safe moves, we pick the one with minimum cost.
        
        best_move = None
        min_cost = float('inf')
        safe_move_found = False
        
        # Prioritize small primes? Or iterate all?
        # Iterating all 168 primes is fast.
        for i, p in enumerate(_PRIMES):
            old_r = residues[i]
            new_r = gap % p
            
            if old_r == new_r:
                continue
                
            # Calculate cost: how many x in [1, current_L] lose coverage and drop below 2?
            # x must satisfy x % p == old_r.
            # Let x = k*p + old_r.
            # We need count[x] == 2 before removing old_r.
            # After removing old_r, count[x] becomes 1.
            # So we count such x.
            
            cost = 0
            # Iterate x in [1, current_L] with x % p == old_r
            start_x = old_r
            if start_x == 0:
                start_x = p
            # If old_r is 0, the first multiple is p.
            # If old_r > 0, the first is old_r (if <= L) or old_r + p...
            
            # Actually, range(start, end, step)
            # start = old_r if old_r != 0 else p
            # But if old_r > current_L, cost is 0.
            
            if old_r > current_L:
                cost = 0
            else:
                # start is the first x >= 1 such that x % p == old_r
                if old_r == 0:
                    start_x = p
                else:
                    start_x = old_r
                    
                # Check if start_x is within [1, current_L]
                if start_x > current_L:
                    cost = 0
                else:
                    for x in range(start_x, current_L + 1, p):
                        if count[x] == 2:
                            cost += 1
                        # Optimization: if cost > min_cost, we can break early?
                        # Yes, if we already have a better candidate.
                        if cost > min_cost:
                            break
        
            # Evaluate move
            # We want to minimize cost.
            # If cost == 0, it's safe.
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r)
                if cost == 0:
                    safe_move_found = True
                    # We can break early if we found a safe move?
                    # Maybe not, a smaller prime might be better for future?
                    # But for now, any safe move extends L.
                    # Let's keep looking for a bit to see if there's a "better" safe move?
                    # No, cost is 0 for all safe moves.
                    # So break.
                    break
            
        if best_move is not None:
            i, new_r = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            
            # Apply move
            start_old = p if old_r == 0 else old_r
            start_new = p if new_r == 0 else new_r
            
            for x in range(start_old, L_LIMIT + 1, p):
                count[x] -= 1
            for x in range(start_new, L_LIMIT + 1, p):
                count[x] += 1
                
            residues[i] = new_r
            
            # Recalculate L
            # If cost was 0, L should at least be current_L.
            # And since we covered gap, L should extend.
            # Just recalculate from current_L?
            # No, if we broke some x < current_L, L might decrease.
            # But we minimized cost. If cost > 0, L might decrease.
            # If cost == 0, L >= current_L.
            
            new_L = _find_prefix_L(count, L_LIMIT)
            
            if new_L > current_L:
                current_L = new_L
                improved = True
                stagnation_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
            else:
                # L decreased or stayed same.
                # This is bad. We accepted a move that broke the prefix.
                # We should probably revert if it broke significantly?
                # Or accept if it was the best available?
                # In Greedy, we usually only accept if it improves or maintains.
                # If it decreases, we should revert.
                
                # Revert
                for x in range(start_new, L_LIMIT + 1, p):
                    count[x] -= 1
                for x in range(start_old, L_LIMIT + 1, p):
                    count[x] += 1
                residues[i] = old_r
                
                improved = False
                stagnation_count += 1
        
        if not improved:
            stagnation_count += 1
            
        # 2. SA Burst if Stagnant
        if stagnation_count > max_stagnation:
            # Perform SA for a short time
            T = 5.0
            cooling = 0.99
            min_T = 0.5
            sa_start_time = time.time()
            
            # SA Loop
            while time.time() < sa_start_time + 5.0 and T > min_T: # 5s burst
                i = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                start_old = p if old_r == 0 else old_r
                start_new = p if new_r == 0 else new_r
                
                for x in range(start_old, L_LIMIT + 1, p):
                    count[x] -= 1
                for x in range(start_new, L_LIMIT + 1, p):
                    count[x] += 1
                    
                residues[i] = new_r
                
                new_L = _find_prefix_L(count, L_LIMIT)
                
                delta = new_L - current_L
                if delta > 0:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = list(residues)
                        stagnation_count = 0
                else:
                    # Accept with probability exp(delta / T)
                    # delta <= 0
                    if delta == 0:
                        # Keep state
                        pass
                    else:
                        prob = 2.718281828 ** (delta / T)
                        if random.random() > prob:
                            # Revert
                            for x in range(start_new, L_LIMIT + 1, p):
                                count[x] -= 1
                            for x in range(start_old, L_LIMIT + 1, p):
                                count[x] += 1
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
    count_final = _build_count(best_residues, L_MAX_FINAL)
    final_L = _find_prefix_L(count_final, L_MAX_FINAL)
    
    return best_residues
