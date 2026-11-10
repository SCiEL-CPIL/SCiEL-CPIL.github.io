import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    def is_valid_addition(S, cnt, x):
        """Check if adding x to S keeps all sums <= 2.
        Uses early termination for speed."""
        # Check 2*x
        if cnt[2*x] > 1:
            return False
            
        # Check x+y for all y in S
        for y in S:
            if cnt[x + y] > 1:
                return False
        return True

    def perform_add(S, cnt, x):
        S.add(x)
        cnt[2*x] += 1
        for y in S:
            if y != x:
                cnt[x + y] += 1

    def perform_remove(S, cnt, x):
        S.remove(x)
        cnt[2*x] -= 1
        for y in S:
            cnt[x + y] -= 1

    def greedy_ordered(order_type, seed):
        """Greedy with different ordering strategies."""
        random.seed(seed)
        S = set()
        cnt = [0] * (max_sum + 1)
        
        if order_type == 0:  # Random
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        elif order_type == 1:  # Ascending
            candidates = list(range(1, N + 1))
        elif order_type == 2:  # Descending
            candidates = list(range(N, 0, -1))
        elif order_type == 3:  # Weighted random (favor middle)
            candidates = list(range(1, N + 1))
            weights = [min(x, N + 1 - x) for x in candidates]
            # Shuffle with bias
            for i in range(len(candidates) - 1, 0, -1):
                j = random.randint(0, i)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        elif order_type == 4:  # Sparse first (multiples of small numbers)
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            # Put smaller numbers first with some randomness
            candidates.sort(key=lambda x: (x % 7, random.random()))
        else:  # Fully random with salt
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(S, cnt, x):
                perform_add(S, cnt, x)
        
        return S, cnt

    def count_conflicts(S, cnt, x):
        """Count how many sums would exceed limit if x were added."""
        conflicts = 0
        if cnt[2*x] > 1:
            conflicts += 1
        for y in S:
            if cnt[x + y] > 1:
                conflicts += 1
        return conflicts

    def local_search(S, cnt, deadline):
        """Improved local search with better swap strategy and shake."""
        best_size = len(S)
        best_S = set(S)
        best_cnt = cnt[:]
        
        no_improve_count = 0
        max_no_improve = 3000
        shake_threshold = 1500
        
        S_list = list(S)
        
        while time.time() < deadline:
            if time.time() > deadline - 3:
                break
                
            # Try to add a random element (more attempts)
            added = False
            for _ in range(15):
                v = random.randint(1, N)
                if v in S:
                    continue
                if is_valid_addition(S, cnt, v):
                    perform_add(S, cnt, v)
                    added = True
                    break
            
            if added:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    best_cnt = cnt[:]
                no_improve_count = 0
            else:
                # Try swaps - more aggressive
                swapped = False
                for _ in range(20):
                    u = random.choice(S_list)
                    v = random.randint(1, N)
                    if v in S or v == u:
                        continue
                    
                    # Check swap u -> v
                    perform_remove(S, cnt, u)
                    
                    if is_valid_addition(S, cnt, v):
                        perform_add(S, cnt, v)
                        S_list.remove(u)
                        S_list.append(v)
                        swapped = True
                        break
                    else:
                        perform_add(S, cnt, u)
                
                if swapped:
                    no_improve_count += 1
                else:
                    no_improve_count += 1
                
                if no_improve_count > shake_threshold:
                    # Smart shake: remove elements that block the most additions
                    # Compute for each element how many potential additions it blocks
                    remove_candidates = []
                    sample_size = min(20, len(S_list))
                    sample = random.sample(S_list, sample_size)
                    
                    for u in sample:
                        # Count how many v in 1..N would become valid if u removed
                        perform_remove(S, cnt, u)
                        unblocked = 0
                        for _ in range(5):
                            v = random.randint(1, N)
                            if v not in S and v != u:
                                if is_valid_addition(S, cnt, v):
                                    unblocked += 1
                        perform_add(S, cnt, u)
                        remove_candidates.append((unblocked, u))
                    
                    remove_candidates.sort(reverse=True)
                    num_remove = min(3, len(S_list))
                    for _, u in remove_candidates[:num_remove]:
                        perform_remove(S, cnt, u)
                        S_list.remove(u)
                    no_improve_count = 0

        return best_S, best_cnt

    # Phase 1: Multiple diverse greedy passes
    best_S = set()
    best_cnt = [0] * (max_sum + 1)
    best_size = 0
    
    # Run diverse greedy strategies
    for i in range(30):
        order_type = i % 5
        S, cnt = greedy_ordered(order_type, i * 12345 + 67890)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            best_cnt = cnt
            if time.time() > time_limit * 0.4:
                break
            
    # Phase 2: Local Search on the best greedy result
    if len(best_S) > 0:
        phase2_deadline = min(time_limit * 0.7, time.time() + 30)
        final_S, final_cnt = local_search(best_S, best_cnt, phase2_deadline)
        if len(final_S) > best_size:
            best_S = final_S
            best_size = len(final_S)
            best_cnt = final_cnt
            
    # Phase 3: Try to improve further with more local search if time permits
    if time.time() < time_limit - 10 and len(best_S) > 0:
        phase3_deadline = min(time_limit - 5, time.time() + 25)
        final_S2, final_cnt2 = local_search(best_S, best_cnt, phase3_deadline)
        if len(final_S2) > best_size:
            best_S = final_S2
            best_size = len(final_S2)
            best_cnt = final_cnt2

    # Final safety check
    if time.time() > time_limit - 2:
        pass  # Just return what we have

    return list(best_S)

# EVOLVE-BLOCK-END
