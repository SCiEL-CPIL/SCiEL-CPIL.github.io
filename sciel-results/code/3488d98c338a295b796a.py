import time
import sys
import random
import numpy as np

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
    for trial in range(40):
        if time.time() - start_time > DEADLINE:
            break
        S_trial = greedy(skip_prob=0.03 + 0.015 * trial)
        if len(S_trial) > best_len:
            best_set = S_trial
            best_len = len(S_trial)

    # Phase 2: Improved local search with smarter candidate ordering
    def local_search(S_init, max_iterations=200):
        S = list(S_init)
        used_sums = compute_sums(S)
        current_len = len(S)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to add a new element
            improved = False
            S_set = set(S)
            
            # Generate candidates more efficiently: only check numbers that might fit
            # Use a window around existing elements plus some random samples
            if current_len > 0:
                # Focus on gaps between existing elements and near the edges
                S_sorted = sorted(S)
                candidate_regions = []
                # Small numbers
                candidate_regions.extend(range(1, max(1, S_sorted[0])))
                # Gaps between consecutive elements
                for a, b in zip(S_sorted[:-1], S_sorted[1:]):
                    gap = b - a
                    if gap > 1:
                        # Sample from the gap
                        if gap < 20:
                            candidate_regions.extend(range(a + 1, b))
                        else:
                            # Sample a few from large gaps
                            for _ in range(min(5, gap - 1)):
                                candidate_regions.append(random.randint(a + 1, b - 1))
                # Large numbers above max
                if S_sorted[-1] < max_val:
                    above = max_val - S_sorted[-1]
                    if above < 50:
                        candidate_regions.extend(range(S_sorted[-1] + 1, max_val + 1))
                    else:
                        for _ in range(20):
                            candidate_regions.append(random.randint(S_sorted[-1] + 1, max_val))
                
                # Filter out already-in-set candidates
                candidates = [x for x in candidate_regions if x not in S_set]
                # Add some random candidates for diversity
                for _ in range(50):
                    r = random.randint(1, max_val)
                    if r not in S_set and r not in candidates:
                        candidates.append(r)
            else:
                candidates = list(range(1, max_val + 1))
            
            # Shuffle for diversity
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(S, used_sums, x):
                    S.append(x)
                    add_to_used(used_sums, S[:-1], x)
                    current_len += 1
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one element, then try to add a new one
                if len(S) > 1:
                    idx_to_remove = random.randint(0, len(S) - 1)
                    elem_to_remove = S[idx_to_remove]
                    
                    remove_from_used(used_sums, S, elem_to_remove)
                    S.pop(idx_to_remove)
                    
                    S_set = set(S)
                    # Try to add a new element
                    if len(S) > 0:
                        S_sorted = sorted(S)
                        candidate_regions = []
                        candidate_regions.extend(range(1, max(1, S_sorted[0])))
                        for a, b in zip(S_sorted[:-1], S_sorted[1:]):
                            gap = b - a
                            if gap > 1:
                                if gap < 20:
                                    candidate_regions.extend(range(a + 1, b))
                                else:
                                    for _ in range(min(5, gap - 1)):
                                        candidate_regions.append(random.randint(a + 1, b - 1))
                        if S_sorted[-1] < max_val:
                            above = max_val - S_sorted[-1]
                            if above < 50:
                                candidate_regions.extend(range(S_sorted[-1] + 1, max_val + 1))
                            else:
                                for _ in range(20):
                                    candidate_regions.append(random.randint(S_sorted[-1] + 1, max_val))
                        candidates = [x for x in candidate_regions if x not in S_set]
                        for _ in range(50):
                            r = random.randint(1, max_val)
                            if r not in S_set and r not in candidates:
                                candidates.append(r)
                    else:
                        candidates = list(range(1, max_val + 1))
                    
                    random.shuffle(candidates)
                    
                    added = False
                    for x in candidates:
                        if can_add(S, used_sums, x):
                            S.append(x)
                            add_to_used(used_sums, S[:-1], x)
                            added = True
                            break
                    
                    if not added:
                        S.insert(idx_to_remove, elem_to_remove)
                        add_to_used(used_sums, S, elem_to_remove)
        
        return S

    # Run local search from the best set found so far
    if time.time() - start_time < DEADLINE:
        S_local = local_search(best_set, max_iterations=150)
        if len(S_local) > best_len:
            best_set = S_local
            best_len = len(S_local)

    # Phase 3: More aggressive local search with multiple restarts
    for restart in range(15):
        if time.time() - start_time > DEADLINE:
            break
        
        S_perturbed = list(best_set)
        for _ in range(min(3, len(S_perturbed) // 4)):
            if len(S_perturbed) > 1:
                idx = random.randint(0, len(S_perturbed) - 1)
                S_perturbed.pop(idx)
        
        S_local2 = local_search(S_perturbed, max_iterations=150)
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
        
        if time.time() - start_time < DEADLINE - 5:
            S_off_local = local_search(S_off, max_iterations=100)
            if len(S_off_local) > best_len:
                best_set = S_off_local
                best_len = len(S_off_local)

    # Phase 5: Final polish - try to insert elements one by one
    if time.time() - start_time < DEADLINE - 3:
        S_polish = list(best_set)
        used_sums = compute_sums(S_polish)
        S_set = set(S_polish)
        
        # Try all numbers in order, inserting if possible
        for x in range(1, max_val + 1):
            if time.time() - start_time > DEADLINE - 2:
                break
            if x in S_set:
                continue
            if can_add(S_polish, used_sums, x):
                S_polish.append(x)
                add_to_used(used_sums, S_polish[:-1], x)
                S_set.add(x)
        
        if len(S_polish) > best_len:
            best_set = S_polish
            best_len = len(S_polish)

    # Final verification
    if time.time() - start_time < DEADLINE - 2:
        S_final = best_set
        n = len(S_final)
        sum_counts = {}
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = S_final[i] + S_final[j] + S_final[k]
                    if s in sum_counts:
                        # Invalid - fall back
                        S_greedy_final = greedy()
                        best_set = S_greedy_final
                        best_len = len(S_greedy_final)
                        break
                    sum_counts[s] = 1
                else:
                    continue
                break
            else:
                continue
            break

    return best_set
