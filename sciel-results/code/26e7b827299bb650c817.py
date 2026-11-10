"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Global deadline guard
_START_TIME = time.time()
# 90s kill limit. Use 85s for internal safety.
_DEADLINE = 85.0 

N_MAX = 192
K = 3

def _validate_set(indices):
    """
    Vectorized validity check for a set of indices.
    Returns True if the set is valid.
    
    Condition: For no i in indices, does indices[i] divide the sum of any 3 distinct 
    other indices in the set.
    
    We check: 
    For each i, let x = indices[i].
    We need to ensure there are no distinct j, k, l in indices (j,k,l != i, j!=k, k!=l, l!=j)
    such that (indices[j] + indices[k] + indices[l]) % x == 0.
    
    Optimized check:
    1. Compute all pairwise sums of distinct indices in the set.
    2. For each candidate divisor x (at index i), check if any sum of (x_pair + x_k) is divisible by x,
       where x_pair is a sum of two distinct indices not involving i, and x_k is an index not involving i or the pair.
    
    Actually, a simpler vectorized approach for N <= 48:
    Let V be the array of values.
    Let S be the set of indices.
    
    We can compute all triple sums.
    Number of triples is C(N, 3) ~ 17k.
    For each triple (a,b,c), sum = V[a]+V[b]+V[c].
    We need to check if any V[i] (where i not in {a,b,c}) divides sum.
    
    This is still O(N * C(N,3)) = O(N^4) which is 48^4 ~ 5M ops. 
    In NumPy, this can be done via broadcasting or matrix multiplication tricks?
    
    Alternative:
    For each i, we want to know if there exist j,k,l distinct from i such that sum % V[i] == 0.
    This is equivalent to: 
    Let R_i be the residues of all other elements modulo V[i].
    We need to know if there are 3 distinct elements from the "others" whose residues sum to 0 mod V[i].
    
    Since N is small (<=48), we can do a semi-vectorized check per element.
    For each i:
        others = [v for j, v in enumerate(V) if j != i]
        # Check if any 3 distinct elements in 'others' sum to 0 mod V[i]
        # This is a 3-sum problem modulo M.
        # M = V[i] can be up to 192.
        # We can use a bitset or just brute force 3-sum on the small list 'others'.
        # C(47, 3) ~ 16k checks per i. 48 * 16k = 768k checks.
        # In Python loops this is ~0.1s. 48 restarts * 0.1s = 4.8s. Acceptable.
        # But we can optimize the 3-sum check.
        
        # Let's implement the 3-sum modulo check efficiently.
        M = V[i]
        if M == 1:
            # 1 divides everything. So if there are >=3 others, it's invalid.
            if len(others) >= 3:
                return False
            continue
            
        # Create a set of available values? No, values are distinct, but we care about indices for distinctness.
        # Since values are distinct, checking values is same as checking indices for distinctness.
        # Wait, the problem says "distinct positive integers". So values are unique.
        # So we can just work with the list of values.
        
        # Check if any 3 distinct values in 'others' sum to a multiple of M.
        # Let's sort 'others'.
        # Actually, just nested loops with early break?
        # Or use a set for sums of pairs?
        
        # Pair sums:
        # We need a+b+c = k*M.
        # a+b = k*M - c.
        
        # Let's just do the O(N^3) check in pure Python for the 3-sum part, 
        # but we can prune if M is large.
        # Sum of 3 distinct positives is at least 1+2+3=6.
        # Max sum is ~ 190+189+188 ~ 567.
        # So k*M is in [6, 567].
        
        # Let's just iterate triples in 'others'.
        # To speed up, we can precompute pair sums?
        # For a given M, we want to know if there is a triple summing to 0 mod M.
        
        # Let's use a boolean array for residues?
        # Residues of 'others' mod M.
        # We need 3 distinct elements.
        
        # Simple O(N^3) loop is probably fine for N=47.
        # 47*46*45 / 6 = 16215 iterations.
        # 48 * 16215 = 778k iterations.
        # In Python, 1M iterations is ~0.05s.
        # So 0.05s per validation.
        # SA with 10k steps -> 500s. Too slow.
        
        # We need a faster validation.
        
        # Optimization:
        # Precompute all pair sums of the set?
        # Let P be the set of all sums of 2 distinct elements in V.
        # For each i, we need to check if there exists c in V (c != V[i]) and p in P 
        # such that p + c is divisible by V[i], AND the elements forming p are distinct from c and V[i].
        
        # This is tricky because of the distinctness constraint.
        
        # Alternative:
        # Since N is small, maybe we can just use the O(N^3) check but implemented in C via NumPy?
        # No, pure Python is faster for small loops than NumPy overhead sometimes, but here we can use NumPy for the triple generation?
        
        # Let's try a different angle.
        # The condition is: No x in V divides sum of any 3 distinct others.
        
        # Let's stick to the O(N^3) Python loop but optimize it heavily.
        # We can precompute the list of values.
        
        # Actually, let's look at the structure.
        # If we pick numbers > 96, they can only be divided by 1? No.
        # If x > 96, the only multiple of x in the range of sum (max ~576) is x itself or 2x.
        # Sum of 3 distinct positives is at least 6.
        # If x > 96, x is not in the sum (distinct others).
        # Can sum be 2x?
        # If x > 96, 2x > 192.
        # Max sum of 3 distinct from 1..192 is 192+191+190 = 573.
        # So multiples of x in [6, 573] are x, 2x, 3x, 4x, 5x, 6x.
        # If x > 96, x is not in sum.
        # So we check if sum is 2x, 3x, ...
        
        # This doesn't simplify the check much.
        
        # Let's try to speed up the O(N^3) check.
        # We can break early if we find a violation.
        
        # For the SA, we need to evaluate many states.
        # Maybe we can use a "Delta" evaluation?
        # When we swap i and j, only the validity of i, j, and their neighbors change?
        # No, the sum of triples involving i or j changes.
        # Triples involving i: C(N-1, 2) ~ 1000.
        # Triples involving j: C(N-1, 2) ~ 1000.
        # So we only need to recheck ~2000 triples.
        # This is much faster than full re-validation.
        
    """
    pass

def _check_violations(values, indices_to_check=None):
    """
    Returns a list of violations (tuples of (divisor_idx, summand_idx1, summand_idx2, summand_idx3)).
    If indices_to_check is provided, only check divisibility for those indices.
    Otherwise check all.
    """
    n = len(values)
    violations = []
    if indices_to_check is None:
        indices_to_check = range(n)
    
    # Precompute values as numpy array for fast access?
    # values is a list of ints.
    
    for i in indices_to_check:
        x = values[i]
        if x == 1:
            # 1 divides everything.
            # If there are at least 3 other elements, violation.
            if n > 3:
                violations.append((i, -1, -1, -1)) # Special marker
                continue
            
        # Check all triples of distinct indices j, k, l != i
        # To avoid O(N^3) per i, we can precompute?
        # No, just loop.
        # Optimization: 
        # If x is large, the number of multiples is small.
        # But we still need to find if a triple sums to one of them.
        
        # Let's just do the triple loop.
        # We can sort indices? No, values are fixed.
        
        # Iterate j
        for j in range(n):
            if j == i: continue
            vj = values[j]
            # Iterate k
            for k in range(j+1, n):
                if k == i: continue
                vk = values[k]
                # Iterate l
                for l in range(k+1, n):
                    if l == i: continue
                    vl = values[l]
                    s = vj + vk + vl
                    if s % x == 0:
                        violations.append((i, j, k, l))
                        return violations # Early exit for this i? 
                        # Actually, if we just need validity, we can return False.
                        # But for SA, we might want the count or specific violation to guide?
                        # Let's just return True/False for now.
    
    return violations

def _is_valid_fast(values):
    """
    Fast validity check.
    """
    n = len(values)
    for i in range(n):
        x = values[i]
        if x == 1 and n > 3:
            return False
        
        # Check triples
        # Optimization: 
        # We can precompute pair sums for the set excluding i?
        # No, let's just loop.
        # To speed up, we can check if x is large.
        
        # Let's try to prune.
        # Min sum of 3 distinct others.
        # Max sum of 3 distinct others.
        
        # Just loop.
        found = False
        # We can use a set for seen sums? No, we need specific triple.
        
        # Iterate j < k < l, j,k,l != i
        # To make it faster, we can iterate over indices.
        for j in range(n):
            if j == i: continue
            for k in range(j+1, n):
                if k == i: continue
                for l in range(k+1, n):
                    if l == i: continue
                    if (values[j] + values[k] + values[l]) % x == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    
    # Initial solution: Greedy descending
    # This provides a valid starting point.
    initial_set = []
    for c in range(192, 0, -1):
        # Check if c can be added
        # Use a quick check for initial greedy
        # Since initial set is small, O(N^3) is fine.
        valid = True
        if len(initial_set) >= 3:
            # Check if c divides sum of any 3 in initial_set
            for i in range(len(initial_set)):
                for j in range(i+1, len(initial_set)):
                    for k in range(j+1, len(initial_set)):
                        if (initial_set[i] + initial_set[j] + initial_set[k]) % c == 0:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid:
                continue
        
        # Check if any existing a divides sum of (c + 2 others)
        for idx_a, a in enumerate(initial_set):
            # Need 2 others from initial_set \ {a}
            # Sum = c + x + y
            # (c + x + y) % a == 0
            others = [initial_set[m] for m in range(len(initial_set)) if m != idx_a]
            found = False
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    if (c + others[i] + others[j]) % a == 0:
                        found = True
                        break
                if found: break
            if found:
                valid = False
                break
        
        if valid:
            initial_set.append(c)
            
    best_set = initial_set
    best_size = len(best_set)
    
    # Simulated Annealing
    # State: list of values
    # Moves:
    # 1. Swap two elements (maintains size)
    # 2. Remove one element (size -1)
    # 3. Add one element (size +1) - only if valid
    
    # We want to maximize size.
    # So we prefer moves that increase size.
    # If size is same, we prefer moves that are "more valid" (fewer violations)?
    # But we must maintain validity.
    # So we only accept moves that result in a VALID set.
    # If no valid move increases size, we might be stuck.
    # Then we can try to swap to a different valid configuration of same size, hoping to enable an addition later.
    
    current_set = list(best_set)
    current_size = len(current_set)
    
    # Temperature
    T = 1.0
    cooling = 0.99
    
    # Time budget for SA
    sa_deadline = start_time + 75.0
    
    while time.time() < sa_deadline:
        T *= cooling
        if T < 0.01:
            T = 0.01
            
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap
            if len(current_set) < 2:
                continue
            i, j = random.sample(range(len(current_set)), 2)
            new_set = current_set.copy()
            new_set[i], new_set[j] = new_set[j], new_set[i]
            
            # Check validity
            # We can just re-validate fully for simplicity, as N is small.
            if _is_valid_fast(new_set):
                # Accept if size is same (it is)
                # We accept all valid swaps to explore the space.
                current_set = new_set
                if len(current_set) > best_size:
                    best_set = current_set
                    best_size = len(current_set)
                
        elif move_type < 0.75:
            # Remove
            if len(current_set) > 1:
                i = random.randint(0, len(current_set) - 1)
                new_set = current_set.copy()
                removed_val = new_set.pop(i)
                
                # Removing an element always preserves validity?
                # If a set is valid, any subset is valid.
                # Yes!
                # So we can always remove.
                # But we don't want to shrink.
                # We only remove if we plan to add something else?
                # In SA, we accept removals with low probability to escape local optima.
                # Probability = exp(-1/T) ?
                # Since removal decreases objective (size), we penalize it.
                if random.random() < np.exp(-1.0 / T):
                    current_set = new_set
                    # Note: best_set is not updated as size decreased
                
        else:
            # Add
            # Try to add a random element from 1..192 not in current_set
            candidates = [x for x in range(1, 193) if x not in current_set]
            if candidates:
                c = random.choice(candidates)
                new_set = current_set + [c]
                
                if _is_valid_fast(new_set):
                    # Accept if size increases
                    current_set = new_set
                    if len(current_set) > best_size:
                        best_set = current_set
                        best_size = len(current_set)
    
    # Final local search on best_set
    # Try to add any remaining element
    improved = True
    while improved and time.time() < _DEADLINE - 1:
        improved = False
        unselected = [x for x in range(1, 193) if x not in best_set]
        for c in unselected:
            new_set = best_set + [c]
            if _is_valid_fast(new_set):
                best_set = new_set
                improved = True
                break
    
    return best_set

# EVOLVE-BLOCK-END
