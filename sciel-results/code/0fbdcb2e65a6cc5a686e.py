"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit
    
    def _is_valid_set(S):
        """
        Check if S is valid.
        Returns True if no element is the mean of any subset of other elements of size >= 2.
        Uses per-element subset-sum DP with bitsets.
        """
        n = len(S)
        if n < 3:
            return True
        
        S_list = list(S)
        for i, a in enumerate(S_list):
            # Build DP for S \ {a}
            n_t = n - 1
            dp = [0] * (n_t + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            for j, x in enumerate(S_list):
                if j == i:
                    continue
                # Update in reverse
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            # Check if k*a is in dp[k] for any k >= 2
            # Max possible sum with k elements from T: k * max(T)
            max_val = max(S_list)
            for k in range(2, n_t + 1):
                target = k * a
                if target > k * max_val:
                    break
                if (dp[k] >> target) & 1:
                    return False
                    
        return True

    def _find_violations(S):
        """
        For each element a in S, find if it is the mean of any subset of S\{a}.
        Returns dict: a -> list of (k, subset_indices) for violations, or empty if none.
        Also returns a 'badness' score for each element.
        """
        n = len(S)
        S_list = list(S)
        violations = {a: [] for a in S_list}
        
        for i, a in enumerate(S_list):
            n_t = n - 1
            dp = [0] * (n_t + 1)
            dp[0] = 1
            
            for j, x in enumerate(S_list):
                if j == i:
                    continue
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            max_val = max(S_list)
            for k in range(2, n_t + 1):
                target = k * a
                if target > k * max_val:
                    break
                if (dp[k] >> target) & 1:
                    violations[a].append(k)
                    
        return violations

    def _count_conflicts(S):
        """
        Calculate a badness score for each element.
        Score = number of k >= 2 such that k*a is achievable by S\{a}.
        """
        violations = _find_violations(S)
        scores = {a: len(v) for a, v in violations.items()}
        return scores

    def _repair_set(S):
        """
        Iteratively remove elements until S is valid.
        Returns the valid subset.
        """
        S = list(S)
        while not _is_valid_set(S):
            if time.time() > deadline:
                break
            scores = _count_conflicts(S)
            max_score = max(scores.values())
            if max_score == 0:
                # Should not happen if _is_valid_set returned False, but safety
                break
            candidates = [a for a, s in scores.items() if s == max_score]
            to_remove = random.choice(candidates)
            S.remove(to_remove)
            
        return S

    def _local_search(S, removed_elements, max_iters=200):
        """
        Try to add back removed elements one by one if the set remains valid.
        Also try swaps: remove one from S, add one from removed_elements.
        """
        S = list(S)
        removed = list(removed_elements)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try adding back each removed element
            for i, elem in enumerate(removed):
                if time.time() > deadline:
                    return S
                candidate = S + [elem]
                if _is_valid_set(candidate):
                    S = candidate
                    removed.pop(i)
                    improved = True
                    break
            
            # Try swaps: for each s in S and r in removed, check if S - {s} + {r} is valid
            if not improved and len(removed) > 0:
                for si, s_elem in enumerate(S):
                    if time.time() > deadline:
                        return S
                    for ri, r_elem in enumerate(removed):
                        candidate = S[:si] + S[si+1:] + [r_elem]
                        if _is_valid_set(candidate):
                            S = candidate
                            removed.pop(ri)
                            improved = True
                            break
                    if improved:
                        break
                    
        return S

    def _structured_sample(size):
        """
        Generate a structured initial set that avoids obvious arithmetic progressions.
        Mix random and structured elements.
        """
        # Use a mix: some from low range, some from high, avoid dense clusters
        all_vals = list(range(1, 513))
        random.shuffle(all_vals)
        
        # Ensure good spread: take from different "buckets"
        bucket_size = 512 // 8  # 64 per bucket
        buckets = [all_vals[i*bucket_size:(i+1)*bucket_size] for i in range(8)]
        
        selected = []
        per_bucket = size // 8
        for b in buckets:
            random.shuffle(b)
            selected.extend(b[:per_bucket])
        
        # Fill remaining from all_vals
        remaining = [x for x in all_vals if x not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:size - len(selected)])
        
        return selected[:size]

    best_set = []
    best_size = 0
    
    # Phase 1: Multiple random restarts with repair
    for restart in range(40):
        if time.time() > deadline:
            break
        
        initial_size = random.randint(55, 70)
        
        # Alternate between pure random and structured
        if restart % 2 == 0:
            candidates = random.sample(range(1, 513), initial_size)
        else:
            candidates = _structured_sample(initial_size)
        
        valid_set = _repair_set(candidates)
        
        if len(valid_set) > best_size:
            best_size = len(valid_set)
            best_set = valid_set
            
        if best_size == 48:
            break
    
    # Phase 2: Local search to try to improve best_set
    if best_size < 48 and best_set:
        if time.time() < deadline - 5:
            # Get elements not in best_set
            best_set_set = set(best_set)
            removed = [x for x in range(1, 513) if x not in best_set_set]
            random.shuffle(removed)
            # Only consider a subset of removed elements for efficiency
            removed_subset = removed[:30]
            
            improved = _local_search(best_set, removed_subset, max_iters=300)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # Phase 3: If still short, try a few more targeted restarts
    if best_size < 48:
        for restart in range(15):
            if time.time() > deadline - 2:
                break
            
            # Try starting from best_set and perturbing
            if best_set:
                S = list(best_set)
                # Remove a random element
                if len(S) > 0:
                    S.pop(random.randrange(len(S)))
                # Add a random new element
                new_elem = random.randint(1, 512)
                if new_elem not in S:
                    S.append(new_elem)
                # Repair
                valid_set = _repair_set(S)
                if len(valid_set) > best_size:
                    best_size = len(valid_set)
                    best_set = valid_set
            
            if best_size == 48:
                break
    
    return best_set

# EVOLVE-BLOCK-END
