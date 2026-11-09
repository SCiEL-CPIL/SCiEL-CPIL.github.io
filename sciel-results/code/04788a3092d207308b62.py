"""
Strategy: CSP with Backtracking and Forward Checking.
Instead of Simulated Annealing (stochastic local search), we use a deterministic 
backtracking solver to find a valid assignment of residues for primes <= k.
We binary search for the maximum k.
The key optimization is "forward checking": as we assign residues to primes, 
we maintain a count of survivors. If at any point the number of survivors 
drops below k, we immediately backtrack (prune). This is much more efficient 
than evaluating full states repeatedly in SA, especially when k is large 
and the constraint is tight.
"""
import time
import math
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

# Precompute primes up to N
ALL_PRIMES = _primes(N)

# Precompute modulo indices for each prime for fast masking/counting
# mod_indices[p] is an array of length N+1 containing x % p for x in [0, N]
# We also precompute the "buckets" for each prime: 
# buckets[p][r] is a numpy array of indices x such that x % p == r
# This allows fast intersection operations if needed, but for counting survivors,
# we can just use bincount on the valid mask.
# Actually, for backtracking, we need to quickly update the "valid" mask.
# Let's precompute the masks themselves as boolean arrays? 
# No, that's too much memory. 
# Let's precompute the indices for each residue class.
# indices_by_residue[p] is a list of arrays: indices_by_residue[p][r] = np.array([x for x in range(N+1) if x%p==r])

indices_by_residue = {}
for p in ALL_PRIMES:
    residues = {}
    for r in range(p):
        # x = r, r+p, r+2p, ... <= N
        residues[r] = np.arange(r, N + 1, p)
    indices_by_residue[p] = residues

def construct():
    """
    Constructs the largest admissible tuple in [0, N].
    """
    
    # Binary search for the maximum k
    # We know k cannot exceed N+1.
    # Also, for large k, the number of primes is large.
    # Let's start high and go down, or binary search.
    # Binary search is safer.
    
    low = 0
    high = N + 1
    best_k = 0
    best_residues = []
    
    # Helper: check if a specific k is feasible
    def is_feasible(k):
        """
        Returns True if there exists a set of residues for primes <= k 
        such that at least k numbers in [0, N] are not removed.
        Uses Backtracking with Forward Checking.
        """
        if k == 0:
            return True, []
        
        # Primes involved: all primes p <= k
        primes_involved = [p for p in ALL_PRIMES if p <= k]
        
        if not primes_involved:
            # No primes to constrain, any k numbers work
            return True, []
        
        # Sort primes by size? 
        # Heuristic: Process larger primes first? Or smaller?
        # Smaller primes remove more numbers on average (1/p). 
        # Actually, for forward checking, the order matters for pruning speed.
        # If we process primes that remove many numbers first, we might prune earlier.
        # But if we process "tight" constraints first, we might fail faster.
        # Let's try sorting by p (ascending) as is standard, or maybe descending?
        # Let's stick to ascending for now, but we can experiment.
        # Actually, a good heuristic is to process primes that have the most 
        # "impact" on the current survivor count. But that's dynamic.
        # Static heuristic: Process primes in order of increasing p.
        
        primes_to_assign = primes_involved
        
        # Initial valid mask: all True
        # We will maintain a boolean array 'valid' of size N+1.
        # To speed up backtracking, we can use a stack of changes.
        
        # Optimization: 
        # Instead of copying the valid array (O(N)) at each step, 
        # we can track the number of survivors.
        # When we remove a residue class r for prime p:
        # removed_count = number of x in valid where x % p == r.
        # new_survivors = current_survivors - removed_count.
        # If new_survivors < k, prune.
        
        # We need to efficiently compute `removed_count`.
        # We can maintain `valid` as a numpy boolean array.
        # removed_count = np.sum(valid & (mod_indices[p] == r))
        # This is O(N) per step.
        # With N=5000 and ~100 primes, worst case 100*5000 = 500k ops per path.
        # Backtracking might explore many paths.
        
        # Let's try a different representation for speed:
        # Use a bitset? Python ints are arbitrary precision.
        # valid_bits = (1 << (N+1)) - 1
        # mask_for_p_r = precomputed bitset for x % p == r
        # removed_count = (valid_bits & mask_for_p_r).bit_count()
        # This is very fast in Python 3.10+.
        
        # Let's precompute bitsets for each (p, r).
        # Memory: sum(p for p in primes <= N) * (N+1)/8 bytes?
        # No, each bitset is an int. Python ints are efficient for small bit counts?
        # N=5000, so bitset is ~625 bytes. 
        # Number of (p,r) pairs is sum(p) ~ 76k for p<=5000? No, sum of primes is ~ 1.5M.
        # 1.5M * 625 bytes = ~1GB. Too much memory.
        
        # So we stick to numpy arrays or lists.
        # Numpy boolean array sum is fast.
        
        valid = np.ones(N + 1, dtype=bool)
        current_survivors = N + 1
        
        # To allow fast rollback, we don't copy the array.
        # We record the changes: (p, r, removed_indices)
        # But removed_indices can be large.
        # Alternative: Just recompute? No, too slow.
        # Alternative: Keep a stack of (p, r, count_removed, indices_removed_mask)
        # When backtracking, we restore the mask.
        
        # Let's use a recursive function with explicit stack management.
        
        def backtrack(idx, valid_mask, survivors):
            """
            idx: index in primes_to_assign
            valid_mask: numpy bool array
            survivors: int count of True in valid_mask
            """
            nonlocal best_residues_local
            
            if time.time() > _DEADLINE - 5:
                return False, None # Timeout signal
            
            if idx == len(primes_to_assign):
                # All primes assigned
                if survivors >= k:
                    return True, None
                else:
                    return False, None
            
            p = primes_to_assign[idx]
            
            # Try each residue r for prime p
            # Heuristic: Try residues that remove the FEWEST survivors first?
            # This maximizes the chance of staying above k.
            # Count survivors in each residue class.
            
            # We need counts of valid[x] where x % p == r for all r.
            # Use bincount on valid_mask with weights? 
            # No, valid_mask is boolean. 
            # indices = np.arange(N+1) % p
            # counts = np.bincount(indices, weights=valid_mask, minlength=p)
            
            # To avoid recomputing indices % p every time, use precomputed mod_indices.
            m_idx = np.arange(N + 1) % p # This is expensive if done every time.
            # Let's precompute mod_indices for all primes involved in this call?
            # Or globally?
            # Global precomputation of mod_indices[p] for all p <= N was done in previous thought, 
            # but here I only did indices_by_residue.
            # Let's add a global cache for mod_indices.
            
            # Since we are inside a closure, let's access a global or nonlocal cache.
            # For this implementation, I will precompute mod_indices for all primes <= N 
            # at module level to avoid overhead.
            
            mod_idx = MOD_INDICES[p]
            counts = np.bincount(mod_idx, weights=valid_mask, minlength=p)
            
            # Sort residues by count ascending (remove fewest first)
            # This is a strong heuristic for feasibility.
            sorted_residues = np.argsort(counts)
            
            for r in sorted_residues:
                r = int(r)
                c = counts[r]
                new_survivors = survivors - c
                
                if new_survivors < k:
                    # Even if we remove the minimum possible for this prime, 
                    # we drop below k. Since we sorted ascending, all other residues 
                    # remove >= c survivors. So all options for this prime lead to failure.
                    break
                # Actually, wait. If we remove c survivors, we get new_survivors.
                # If new_survivors < k, we prune.
                # Since we are iterating from smallest c to largest c, 
                # if the smallest c leads to < k, then all larger c will also lead to < k.
                # So we can break immediately.
                
                # Apply removal
                # We need to update valid_mask.
                # To avoid copying valid_mask (O(N)), we can modify it in place 
                # and then undo it.
                
                # Identify indices to remove
                # mask_to_remove = (mod_idx == r) & valid_mask
                # But we already know count c.
                # We need the actual indices to set to False.
                
                # Using precomputed indices_by_residue[p][r]
                idxs = INDICES_BY_RESIDUE[p][r]
                
                # Only those that are currently valid
                # We can just set valid_mask[idxs] = False, but we must be careful 
                # to only affect those that were True.
                # Since we are removing a residue class, any x in idxs that is 
                # currently valid will be removed.
                
                # Save the state of these indices? 
                # Or just set to False, and when backtracking, set back to True?
                # But what if they were already False?
                # We need to know which ones we actually changed.
                
                # Efficient way:
                # changed_mask = valid_mask[idxs]
                # valid_mask[idxs] = False
                # Then backtrack: valid_mask[idxs] = changed_mask
                
                # But valid_mask[idxs] creates a copy.
                # Let's do it. N=5000, idxs length ~ 5000/p.
                # For p=2, length 2500. Copy is fast.
                
                orig_vals = valid_mask[idxs].copy()
                valid_mask[idxs] = False
                
                # Recurse
                success, _ = backtrack(idx + 1, valid_mask, new_survivors)
                
                if success:
                    # We found a solution. 
                    # We need to record the residue choice.
                    # But we are using a local variable for the path?
                    # Let's pass the path or use a nonlocal list.
                    nonlocal current_path
                    current_path[idx] = r
                    # Restore mask before returning up? 
                    # No, if we return True, we don't need to restore 
                    # if we are done. But for cleanliness, or if we want 
                    # to continue searching for other solutions (we don't), 
                    # we can leave it.
                    # However, the caller might want to restore its own mask.
                    # Let's restore before returning to keep state consistent 
                    # in case of multiple calls or if we want to reuse valid_mask.
                    valid_mask[idxs] = orig_vals
                    return True, None
                
                # Undo
                valid_mask[idxs] = orig_vals
                
            return False, None
        
        # Initialize path
        current_path = [0] * len(primes_to_assign)
        
        # Call backtrack
        success, _ = backtrack(0, valid, N + 1)
        
        if success:
            return True, current_path
        else:
            return False, []

    # Binary search loop
    while low <= high:
        if time.time() > _DEADLINE - 20:
            break
            
        mid = (low + high) // 2
        
        feasible, residues = is_feasible(mid)
        
        if feasible:
            best_k = mid
            best_residues = residues
            low = mid + 1
        else:
            high = mid - 1
            
    # Construct the final set
    if best_k == 0:
        return []
        
    primes_to_use = [p for p in ALL_PRIMES if p <= best_k]
    
    if not primes_to_use:
        return list(range(best_k))
        
    # Apply the found residues to generate the set
    valid = np.ones(N + 1, dtype=bool)
    for p, r in zip(primes_to_use, best_residues):
        # Use precomputed indices
        idxs = INDICES_BY_RESIDUE[p][r]
        valid[idxs] = False
        
    result_indices = np.where(valid)[0]
    
    # We need exactly best_k elements.
    if len(result_indices) < best_k:
        # Should not happen if is_feasible was correct, but safety check
        return result_indices.tolist()
        
    final_set = result_indices[:best_k].tolist()
    return final_set

# Global precomputation for speed
MOD_INDICES = {}
INDICES_BY_RESIDUE = {}
for p in ALL_PRIMES:
    MOD_INDICES[p] = np.arange(N + 1) % p
    residues = {}
    for r in range(p):
        residues[r] = np.arange(r, N + 1, p)
    INDICES_BY_RESIDUE[p] = residues

