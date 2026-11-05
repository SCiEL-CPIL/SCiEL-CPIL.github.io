import time
from fractions import Fraction
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    random.seed(42)

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Helper to build a prime set given a mandatory base and a skip set
    def build_set(base, skip):
        base_set = set(base)
        skip_set = set(skip)
        s = sum(Fraction(1, p) for p in base_set)
        if s > 1:
            return []
        
        final_set = list(base_set)
        for p in primes:
            if p in base_set or p in skip_set:
                continue
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return sorted(final_set)

    # Generate diverse candidate sets
    # The insight from inspiration 1 is that specific base sets can work well.
    # The current program's weakness was a rigid greedy residue assignment.
    # We will use local search (hill climbing) for residues, which is a structural change from pure greedy.
    
    candidates = []
    
    # 1. Standard greedy fill (smallest primes first)
    candidates.append(build_set([], []))
    
    # 2. Skip 3
    candidates.append(build_set([], {3}))
    
    # 3. Skip 5
    candidates.append(build_set([], {5}))
    
    # 4. Skip 3, 5
    candidates.append(build_set([], {3, 5}))
    
    # 5. Force 2
    candidates.append(build_set([2], []))
    
    # 6. Force 2, skip 3
    candidates.append(build_set([2], {3}))
    
    # 7. Force 2, skip 5
    candidates.append(build_set([2], {5}))
    
    # 8. Force 2, 5
    candidates.append(build_set([2, 5], []))
    
    # 9. Force 3
    candidates.append(build_set([3], []))
    
    # 10. Force 3, skip 5
    candidates.append(build_set([3], {5}))
    
    # 11. Force 5
    candidates.append(build_set([5], []))
    
    # 12. Force 2, 7
    candidates.append(build_set([2, 7], []))
    
    # 13. Force 2, 11
    candidates.append(build_set([2, 11], []))
    
    # 14. Skip 2, 3
    candidates.append(build_set([], {2, 3}))
    
    # 15. Skip 2, 5
    candidates.append(build_set([], {2, 5}))
    
    # 16. Skip 3, 7
    candidates.append(build_set([], {3, 7}))
    
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    best_L = 0
    best_classes = []
    
    # Evaluation limit for the prefix.
    # We check up to 10,000 to be safe, but the target is 999.
    # Using NumPy for fast coverage updates.
    eval_limit = 10000
    
    for P in candidates:
        if time.time() > deadline - 20:
            break
        if not P:
            continue
        
        # Local Search for Residues
        # Start with a greedy initialization to get a good starting point
        covered = np.zeros(eval_limit + 1, dtype=bool)
        residues = {}
        used_primes = set()
        
        # Greedy Init
        P_sorted = sorted(P)
        for n in range(1, eval_limit + 1):
            if time.time() > deadline - 5:
                break
            if covered[n-1]: # index n-1 corresponds to number n
                continue
            
            assigned = False
            # Try to find a prime that covers n and maximizes immediate future coverage?
            # For init, just pick smallest available.
            for p in P_sorted:
                if p in used_primes:
                    continue
                r = n % p
                residues[p] = r
                used_primes.add(p)
                
                # Update covered
                # Indices i such that (i+1) % p == r
                # i = r - 1, r - 1 + p, ...
                start_idx = r - 1
                if start_idx < 0:
                    start_idx += p # If r=0, start_idx=-1 -> p-1. Wait.
                    # If r=0, numbers are p, 2p... indices p-1, 2p-1...
                    # Formula: i+1 = k*p. i = k*p - 1.
                    # If r=0, n%p=0. n is a multiple of p.
                    # So we cover n, n+p...
                    # Indices: n-1, n-1+p...
                    # My formula: start_idx = r - 1. If r=0, -1.
                    # np.arange(-1, limit, p) gives -1, p-1, 2p-1...
                    # -1 is invalid index. We filter >= 0.
                
                indices = np.arange(start_idx, eval_limit, p)
                indices = indices[indices >= 0]
                covered[indices] = True
                assigned = True
                break
            
            if not assigned:
                break
        
        # Calculate current L
        def get_L(covered_arr):
            # Find first False
            try:
                first_false = np.argmax(~covered_arr)
                if covered_arr[first_false]:
                    return eval_limit
                else:
                    return first_false
            except:
                return 0
        
        current_L = get_L(covered)
        
        # Hill Climbing
        # Try to change residue of one prime to improve L.
        # Or swap residues?
        # Simple hill climb: for each prime, try all possible residues, keep if better.
        
        # We need to be careful with time.
        # Number of primes is small (maybe 10-20).
        # Residues per prime is p.
        # Total ops per pass: sum(p).
        # For each change, we need to recalculate L.
        # Recalculating L from scratch is O(eval_limit).
        # Updating covered array for a change is O(eval_limit/p).
        
        # Let's do a few passes of hill climbing.
        # To speed up, we can just check if the new L is better.
        
        for iteration in range(50): # Max 50 passes
            if time.time() > deadline - 5:
                break
            
            improved = False
            for p in P_sorted:
                if time.time() > deadline - 5:
                    break
                
                old_r = residues.get(p)
                if old_r is None:
                    continue
                    
                # Try all residues
                # Optimization: Only try residues that might help?
                # For now, try all.
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    
                    # Simulate change
                    # We need to update covered array.
                    # It's faster to copy? No, eval_limit is 10k. Copy is fast.
                    new_covered = covered.copy()
                    
                    # Remove old coverage
                    # Indices for old_r
                    start_idx_old = old_r - 1
                    indices_old = np.arange(start_idx_old, eval_limit, p)
                    indices_old = indices_old[indices_old >= 0]
                    
                    # This is tricky because other primes might cover these spots.
                    # We cannot simply "remove" coverage because it might be covered by others.
                    # We must recalculate coverage from scratch for the set of residues?
                    # That is O(N * eval_limit). Too slow for many iterations.
                    
                    # Alternative: Maintain a count array?
                    # count[i] = number of primes covering i.
                    # Then covered[i] = count[i] > 0.
                    # When changing p from r1 to r2:
                    #   Decrement count for indices of r1.
                    #   Increment count for indices of r2.
                    #   Update covered where count becomes 0 or 1.
                    #   Then find L.
                    
                    # This is much better.
                    
                    # Let's refactor to use count array.
                    pass # This approach requires restructuring the loop.

        # Given time constraints and complexity of implementing count-array hill climbing correctly and quickly,
        # let's stick to a simpler heuristic if the above is too complex.
        # However, the "structural change" directive suggests we should try harder.
        
        # Let's implement the count-array hill climbing properly for this candidate set.
        
        # Re-initialize with count array
        counts = np.zeros(eval_limit + 1, dtype=np.int32)
        covered = (counts > 0)
        
        # Apply initial residues
        for p in P_sorted:
            if p in residues:
                r = residues[p]
                start_idx = r - 1
                indices = np.arange(start_idx, eval_limit, p)
                indices = indices[indices >= 0]
                counts[indices] += 1
        
        covered = (counts > 0)
        current_L = get_L(covered)
        
        # Hill Climb
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Shuffle primes to avoid bias
            P_shuffled = P_sorted.copy()
            random.shuffle(P_shuffled)
            
            for p in P_shuffled:
                if time.time() > deadline - 5:
                    break
                
                old_r = residues.get(p)
                if old_r is None:
                    continue
                
                best_new_r = old_r
                best_L_candidate = current_L
                
                # Sample a few residues if p is large?
                # p <= 1000. 1000 residues is fine.
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    
                    # Update counts
                    # Remove old
                    start_old = old_r - 1
                    idx_old = np.arange(start_old, eval_limit, p)
                    idx_old = idx_old[idx_old >= 0]
                    counts[idx_old] -= 1
                    
                    # Add new
                    start_new = new_r - 1
                    idx_new = np.arange(start_new, eval_limit, p)
                    idx_new = idx_new[idx_new >= 0]
                    counts[idx_new] += 1
                    
                    # Check L
                    # We only care if the prefix is extended.
                    # We can check if current_L+1 is covered, etc.
                    # But L can drop too.
                    # To be safe, calculate L.
                    # Optimization: If counts[0:current_L+1] are all > 0, and we check further...
                    # Simple check:
                    # If any count in [0, current_L] becomes 0, L drops.
                    # We can check min(counts[0:current_L+1]) > 0.
                    
                    # Actually, just finding the first 0 is O(L).
                    # Let's try to be smart:
                    # If the change affects indices < current_L, check if any become 0.
                    # If so, L might drop. We can calculate new L.
                    # If not, L is at least current_L. Then check if it extended.
                    
                    # For simplicity and correctness, let's just find L.
                    # But doing this for every residue is expensive.
                    # Let's approximate:
                    # If new_r covers n = current_L + 1, it might extend.
                    # If old_r covered n = current_L + 1, it might shrink.
                    
                    # Let's just calculate L if it looks promising.
                    # Promising if:
                    # 1. It covers current_L + 1 (potential extension)
                    # 2. It doesn't uncover any index < current_L (potential preservation)
                    
                    # Check 2:
                    # Did we uncover anything < current_L?
                    # The indices removed are idx_old.
                    # If any idx in idx_old is < current_L and counts[idx] == 0 after decrement, then L drops.
                    # We already decremented. So check counts[idx_old[idx_old < current_L]].
                    # If any is 0, L drops. We can skip this candidate or calculate exact L.
                    # If any is 0, it's likely worse, so skip?
                    # Not necessarily, if it extends much further.
                    # But usually, keeping the prefix is key.
                    
                    # Let's just calculate L for candidates that cover current_L+1.
                    # If it doesn't cover current_L+1, it can't extend.
                    # Can it shrink? Yes.
                    # So we must ensure it doesn't shrink.
                    
                    # Check if any count in [0, current_L] is 0.
                    # This is a numpy operation: np.any(counts[:current_L+1] == 0)
                    # This is fast.
                    
                    if np.any(counts[:current_L+1] == 0):
                        # L dropped.
                        # Revert
                        counts[idx_old] += 1
                        counts[idx_new] -= 1
                        continue
                    
                    # If prefix is safe, check if it extended.
                    # Check if current_L + 1 is covered.
                    if current_L + 1 <= eval_limit and counts[current_L] > 0: # index current_L is number current_L+1
                        # It might extend. Calculate new L.
                        try:
                            first_false = np.argmax(~(counts > 0))
                            if (counts > 0)[first_false]:
                                new_L = eval_limit
                            else:
                                new_L = first_false
                        except:
                            new_L = current_L # Fallback
                        
                        if new_L > best_L_candidate:
                            best_L_candidate = new_L
                            best_new_r = new_r
                    else:
                        # No extension, but prefix safe. L stays same.
                        # No improvement.
                        pass

                    # Revert
                    counts[idx_old] += 1
                    counts[idx_new] -= 1
                
                if best_new_r != old_r:
                    # Apply best change
                    start_old = old_r - 1
                    idx_old = np.arange(start_old, eval_limit, p)
                    idx_old = idx_old[idx_old >= 0]
                    counts[idx_old] -= 1
                    
                    start_new = best_new_r - 1
                    idx_new = np.arange(start_new, eval_limit, p)
                    idx_new = idx_new[idx_new >= 0]
                    counts[idx_new] += 1
                    
                    residues[p] = best_new_r
                    current_L = best_L_candidate
                    improved = True
            
            if not improved:
                break
        
        # Final check
        if current_L > best_L:
            best_L = current_L
            best_classes = [[p, residues[p]] for p in P if p in residues]
            
    return best_classes

# Test
if __name__ == "__main__":
    classes = construct()
    print(classes)
