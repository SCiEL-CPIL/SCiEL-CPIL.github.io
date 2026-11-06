import time
import sys
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin for 60s limit
    max_val = 4096

    best_set = []
    best_len = 0

    def compute_sums(S):
        """Compute all unordered triple sums for set S."""
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(S[i] + S[j] + S[k])
        return sums

    def can_add(S, used_sums, x):
        """Check if x can be added to S without creating duplicate triple sums."""
        k = len(S)
        # 1. x + x + x
        s1 = 3 * x
        if s1 in used_sums:
            return False
        # 2. x + x + s
        for s in S:
            if (2 * x + s) in used_sums:
                return False
        # 3. x + s_i + s_j
        for i in range(k):
            for j in range(i, k):
                if (x + S[i] + S[j]) in used_sums:
                    return False
        return True

    def add_to_used(used_sums, S, x):
        """Add all new triple sums involving x to used_sums."""
        used_sums.add(3 * x)
        for s in S:
            used_sums.add(2 * x + s)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                used_sums.add(x + S[i] + S[j])

    def remove_from_used(used_sums, S, x):
        """Remove all triple sums involving x from used_sums."""
        used_sums.discard(3 * x)
        for s in S:
            used_sums.discard(2 * x + s)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                used_sums.discard(x + S[i] + S[j])

    # Phase 1: Greedy with different strategies
    def greedy(start_offset=0, skip_prob=0.0, max_candidates=None):
        S = []
        used_sums = set()
        for x in range(1 + start_offset, max_val + 1):
            if max_candidates is not None and len(S) >= max_candidates:
                break
            if skip_prob > 0 and random.random() < skip_prob:
                continue
            if can_add(S, used_sums, x):
                S.append(x)
                add_to_used(used_sums, S[:-1], x)
        return S

    # Standard greedy
    S1 = greedy()
    best_set = S1
    best_len = len(S1)

    # Try greedy with small random skips to explore different configurations
    for trial in range(30):
        if time.time() - start_time > DEADLINE:
            break
        S_trial = greedy(skip_prob=0.05 + 0.02 * trial)
        if len(S_trial) > best_len:
            best_set = S_trial
            best_len = len(S_trial)

    # Phase 2: Local search - try to grow the set by inserting elements
    # Strategy: for each candidate not in set, try to add it; if conflicts, try removing one conflicting element
    def local_search(S_init, max_iterations=500):
        S = list(S_init)
        used_sums = compute_sums(S)
        current_len = len(S)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to add a new element
            improved = False
            # Candidates: numbers not in S, sorted by some heuristic
            S_set = set(S)
            candidates = [x for x in range(1, max_val + 1) if x not in S_set]
            
            # Sort candidates: prefer those with fewer conflicts
            # But computing conflicts is expensive, so just try in random order for diversity
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(S, used_sums, x):
                    # Can add directly
                    S.append(x)
                    add_to_used(used_sums, S[:-1], x)
                    current_len += 1
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one element, then try to add a new one
                # Pick a random element to remove
                if len(S) > 1:
                    idx_to_remove = random.randint(0, len(S) - 1)
                    elem_to_remove = S[idx_to_remove]
                    
                    # Remove it
                    remove_from_used(used_sums, S, elem_to_remove)
                    S.pop(idx_to_remove)
                    
                    # Try to add a new element
                    S_set = set(S)
                    candidates = [x for x in range(1, max_val + 1) if x not in S_set]
                    random.shuffle(candidates)
                    
                    added = False
                    for x in candidates:
                        if can_add(S, used_sums, x):
                            S.append(x)
                            add_to_used(used_sums, S[:-1], x)
                            added = True
                            break
                    
                    if not added:
                        # Put it back
                        S.insert(idx_to_remove, elem_to_remove)
                        add_to_used(used_sums, S, elem_to_remove)
        
        return S

    # Run local search from the best set found so far
    if time.time() - start_time < DEADLINE:
        S_local = local_search(best_set, max_iterations=100)
        if len(S_local) > best_len:
            best_set = S_local
            best_len = len(S_local)

    # Phase 3: More aggressive local search with multiple restarts
    for restart in range(10):
        if time.time() - start_time > DEADLINE:
            break
        
        # Start from a slightly perturbed version of the best set
        S_perturbed = list(best_set)
        # Remove a few random elements
        for _ in range(min(3, len(S_perturbed) // 4)):
            if len(S_perturbed) > 1:
                idx = random.randint(0, len(S_perturbed) - 1)
                S_perturbed.pop(idx)
        
        S_local2 = local_search(S_perturbed, max_iterations=200)
        if len(S_local2) > best_len:
            best_set = S_local2
            best_len = len(S_local2)

    # Phase 4: Try greedy from different starting points
    for offset in [0, 5, 10, 20, 50, 100]:
        if time.time() - start_time > DEADLINE:
            break
        S_off = greedy(start_offset=offset)
        if len(S_off) > best_len:
            best_set = S_off
            best_len = len(S_off)
        
        # Local search from this
        if time.time() - start_time < DEADLINE - 5:
            S_off_local = local_search(S_off, max_iterations=100)
            if len(S_off_local) > best_len:
                best_set = S_off_local
                best_len = len(S_off_local)

    # Final verification and cleanup
    if time.time() - start_time < DEADLINE - 2:
        # Verify the best set is valid
        S_final = best_set
        sums = compute_sums(S_final)
        # Check for duplicates: if any sum appears more than once, it's invalid
        # But compute_sums uses a set, so we need to count
        from collections import Counter
        sum_counts = Counter()
        n = len(S_final)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sum_counts[S_final[i] + S_final[j] + S_final[k]] += 1
        # If any sum has count > 1, the set is invalid
        if any(v > 1 for v in sum_counts.values()):
            # Fall back to the greedy set which should be valid
            S_greedy_final = greedy()
            best_set = S_greedy_final
            best_len = len(S_greedy_final)

    return best_set
