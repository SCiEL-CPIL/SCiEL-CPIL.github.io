import math
import random
import time
from fractions import Fraction

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
NUM_PRIMES = len(_PRIMES)
_COSTS = [Fraction(1, p) for p in _PRIMES]
MAX_CHECK = 100000

def _seed_solution():
    """
    Heuristic seed: Use primes starting from 13.
    The i-th prime (0-indexed from 13) covers integer i+1.
    This ensures coverage of initial integers.
    """
    primes_from_13 = [p for p in _PRIMES if p >= 13]
    assign = [0] * NUM_PRIMES
    for i, p in enumerate(primes_from_13):
        idx = _PRIMES.index(p)
        r = (i + 1) % p
        if r == 0:
            r = p # Residues are 1..p, so 0 mod p is p
        assign[idx] = r
    return assign

def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _COSTS[idx]
    return c <= Fraction(1, 1)

def _build_cover_state(assign):
    """
    Build cover_count array from assign.
    Returns (cover_count, L, cost).
    """
    cover_count = [0] * (MAX_CHECK + 1)
    cost = Fraction(0)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        cost += _COSTS[idx]
        # Mark coverage
        for n in range(r, MAX_CHECK + 1, p):
            cover_count[n] += 1
    
    # Find L
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if cover_count[n] > 0:
            L = n
        else:
            break
    return cover_count, L, cost

def _add_class(cover_count, idx, r):
    """Incrementally add a class to cover_count."""
    p = _PRIMES[idx]
    for n in range(r, MAX_CHECK + 1, p):
        cover_count[n] += 1

def _remove_class(cover_count, idx, r):
    """Incrementally remove a class from cover_count."""
    p = _PRIMES[idx]
    for n in range(r, MAX_CHECK + 1, p):
        cover_count[n] -= 1

def _recompute_L(cover_count):
    """Recompute coverage length from cover_count."""
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if cover_count[n] > 0:
            L = n
        else:
            break
    return L

def _find_first_gap(cover_count, start_from=1):
    """Find the first n >= start_from where cover_count[n] == 0."""
    for n in range(start_from, MAX_CHECK + 1):
        if cover_count[n] == 0:
            return n
    return -1

def _gap_targeted_search(assign, cover_count, current_L, current_cost, time_limit):
    """
    Local search that specifically targets the first gap.
    Tries to cover the gap by:
    1. Changing residue of an existing prime.
    2. Adding a new prime.
    """
    start = time.time()
    current = assign[:]
    curr_L = current_L
    curr_cost = current_cost
    used_indices = [i for i in range(NUM_PRIMES) if current[i] != 0]
    
    best_assign = current[:]
    best_L = curr_L
    best_cost = curr_cost
    
    # We will modify cover_count in place, so we need to be careful with reverts
    # Or we can just recompute if we get stuck, but incremental is faster.
    
    max_rounds = 500
    for _ in range(max_rounds):
        if time.time() - start > time_limit:
            break
        
        gap = _find_first_gap(cover_count, curr_L + 1)
        if gap == -1:
            break # No gap found up to MAX_CHECK
        
        # Try to cover 'gap'
        improved = False
        
        # Option 1: Change residue of an existing used prime
        # We prioritize primes that already cover near the gap or are small
        # But simply trying all used primes is O(K * L/p) which might be slow if K is large.
        # Let's try a random subset or sorted by prime size.
        # Smaller primes have larger steps, so fewer updates.
        shuffled_used = used_indices[:]
        random.shuffle(shuffled_used)
        
        for i in shuffled_used:
            if time.time() - start > time_limit:
                return best_assign, best_L, best_cost
            
            p = _PRIMES[i]
            old_r = current[i]
            
            # Residue needed to cover gap: gap % p
            # If gap % p == 0, residue is p (since 1..p)
            r_needed = gap % p
            if r_needed == 0:
                r_needed = p
            
            if r_needed == old_r:
                continue
            
            # Apply change
            _remove_class(cover_count, i, old_r)
            _add_class(cover_count, i, r_needed)
            current[i] = r_needed
            
            new_L = _recompute_L(cover_count)
            
            if new_L > curr_L:
                curr_L = new_L
                improved = True
                if curr_L > best_L:
                    best_L = curr_L
                    best_assign = current[:]
                    best_cost = curr_cost
                break
            else:
                # Revert
                _remove_class(cover_count, i, r_needed)
                _add_class(cover_count, i, old_r)
                current[i] = old_r
        
        if improved:
            continue
            
        # Option 2: Add a new prime
        # Try to add a prime that covers the gap and is affordable
        unused_indices = [i for i in range(NUM_PRIMES) if current[i] == 0]
        # Sort by cost (larger p = smaller cost) to try cheaper ones first?
        # Or smaller p (denser coverage)?
        # Let's try random selection from affordable unused primes.
        affordable_unused = [i for i in unused_indices if curr_cost + _COSTS[i] <= Fraction(1, 1)]
        
        if affordable_unused:
            # Try a few random ones
            random.shuffle(affordable_unused)
            for i in affordable_unused[:10]: # Limit attempts
                if time.time() - start > time_limit:
                    return best_assign, best_L, best_cost
                
                p = _PRIMES[i]
                r_needed = gap % p
                if r_needed == 0:
                    r_needed = p
                
                # Apply change
                _add_class(cover_count, i, r_needed)
                current[i] = r_needed
                used_indices.append(i)
                old_cost = curr_cost
                curr_cost += _COSTS[i]
                
                new_L = _recompute_L(cover_count)
                
                if new_L > curr_L:
                    curr_L = new_L
                    improved = True
                    if curr_L > best_L:
                        best_L = curr_L
                        best_assign = current[:]
                        best_cost = curr_cost
                    break
                else:
                    # Revert
                    _remove_class(cover_count, i, r_needed)
                    current[i] = 0
                    if i in used_indices:
                        used_indices.remove(i)
                    curr_cost = old_cost
        
        if not improved:
            # Stuck. Try a random perturbation to escape local minimum?
            # Or just break and let SA handle it.
            # For now, break to SA.
            break
            
    return best_assign, best_L, best_cost

def _simulated_annealing(assign, cover_count, current_L, current_cost, time_limit):
    """
    Simulated Annealing with incremental updates.
    """
    start = time.time()
    current = assign[:]
    curr_L = current_L
    curr_cost = current_cost
    used_indices = set(i for i in range(NUM_PRIMES) if current[i] != 0)
    
    best_assign = current[:]
    best_L = curr_L
    best_cost = curr_cost
    
    T = 10.0
    cooling = 0.999
    
    # Precompute list of used and unused for faster access?
    # We will maintain them dynamically.
    
    while time.time() - start < time_limit:
        if T < 0.01:
            break
            
        move_type = random.random()
        
        # Determine move
        old_r = 0
        old_i = -1
        added_prime = False
        removed_prime = False
        
        if move_type < 0.6:
            # Change residue of a used prime
            if not used_indices:
                continue
            i = random.choice(list(used_indices))
            p = _PRIMES[i]
            old_r = current[i]
            new_r = random.randint(1, p)
            if new_r == old_r:
                continue
            
            # Apply
            _remove_class(cover_count, i, old_r)
            _add_class(cover_count, i, new_r)
            current[i] = new_r
            
        elif move_type < 0.8:
            # Add a prime
            unused = [i for i in range(NUM_PRIMES) if i not in used_indices]
            if not unused:
                continue
            i = random.choice(unused)
            if curr_cost + _COSTS[i] > Fraction(1, 1):
                continue
            
            p = _PRIMES[i]
            new_r = random.randint(1, p)
            
            # Apply
            _add_class(cover_count, i, new_r)
            current[i] = new_r
            used_indices.add(i)
            added_prime = True
            old_cost_val = curr_cost
            curr_cost += _COSTS[i]
            
        else:
            # Remove a prime
            if not used_indices:
                continue
            i = random.choice(list(used_indices))
            old_r = current[i]
            
            # Apply
            _remove_class(cover_count, i, old_r)
            current[i] = 0
            used_indices.discard(i)
            removed_prime = True
            old_cost_val = curr_cost
            curr_cost -= _COSTS[i]
            
        # Evaluate
        new_L = _recompute_L(cover_count)
        delta_L = new_L - curr_L
        
        # Acceptance
        accept = False
        if delta_L >= 0:
            accept = True
        else:
            # Probabilistic acceptance
            # Note: delta_L is integer, T is float
            # If delta_L is very negative, exp is 0
            if random.random() < math.exp(delta_L / T):
                accept = True
        
        if accept:
            curr_L = new_L
            if curr_L > best_L:
                best_L = curr_L
                best_assign = current[:]
                best_cost = curr_cost
        else:
            # Revert
            if move_type < 0.6:
                _remove_class(cover_count, i, new_r)
                _add_class(cover_count, i, old_r)
                current[i] = old_r
            elif move_type < 0.8:
                _remove_class(cover_count, i, new_r)
                current[i] = 0
                used_indices.discard(i)
                curr_cost = old_cost_val
            else:
                _add_class(cover_count, i, old_r)
                current[i] = old_r
                used_indices.add(i)
                curr_cost = old_cost_val
            
        T *= cooling
        
    return best_assign, best_L, best_cost

def construct():
    start = time.time()
    deadline = start + 145 # Safety margin
    
    # 1. Seed
    assign = _seed_solution()
    if not _is_valid(assign):
        # Fallback to empty if seed is invalid (shouldn't happen)
        assign = [0] * NUM_PRIMES
    
    # 2. Build State
    cover_count, L, cost = _build_cover_state(assign)
    
    # 3. Gap Targeted Search
    time_for_gts = min(20, deadline - time.time())
    if time_for_gts > 0:
        assign, L, cost = _gap_targeted_search(assign, cover_count, L, cost, time_for_gts)
        
        # Rebuild state to ensure consistency if GTS modified cover_count
        # Actually GTS modifies cover_count in place and returns the best_assign
        # We need to rebuild cover_count from best_assign to be safe for SA
        cover_count, L, cost = _build_cover_state(assign)
        
    # 4. Simulated Annealing
    time_for_sa = min(100, deadline - time.time())
    if time_for_sa > 0:
        assign, L, cost = _simulated_annealing(assign, cover_count, L, cost, time_for_sa)
        
        # Final rebuild to get exact L and cost from the final assign
        cover_count, L, cost = _build_cover_state(assign)
        
    # Final Validation
    if not _is_valid(assign):
        # Should not happen if we tracked cost, but just in case
        assign = _seed_solution()
        
    # Return format: [[p, r], ...]
    result = []
    for i, r in enumerate(assign):
        if r != 0:
            result.append([_PRIMES[i], r])
            
    return result

# To allow testing
if __name__ == "__main__":
    res = construct()
    # Verify
    cost = sum(Fraction(1, p) for p, r in res)
    print(f"Cost: {cost}")
    print(f"Valid: {cost <= 1}")
    # Check coverage
    covered = [0] * (100001)
    for p, r in res:
        for n in range(r, 100001, p):
            covered[n] = 1
    L = 0
    for n in range(1, 100001):
        if covered[n]:
            L = n
        else:
            break
    print(f"Coverage L: {L}")
