"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    chosen_set = set()
    for x in candidates:
        if x in chosen_set:
            continue
        # Check all new sums
        ok = True
        new_sums = []
        for y in chosen:
            s = x + y
            new_sums.append(s)
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            new_sums.append(s2)
            if counts[s2] + 1 > 2:
                ok = False
        if ok:
            for s in new_sums:
                counts[s] += 1
            chosen.append(x)
            chosen_set.add(x)
    return chosen, counts

def _is_valid_b2(chosen):
    """Verify B2[2] property."""
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    return all(v <= 2 for v in counts.values())

def _local_search(chosen, time_limit=10.0):
    """Try swaps and insertions to improve set size."""
    chosen = sorted(chosen)
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    
    start_time = time.time()
    improved = True
    max_iter = 5000
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # First try insertions: add a new element without removing
        chosen_set = set(chosen)
        # Try candidate values in a smart order
        candidates_to_try = []
        for y in range(1, 1001):
            if y not in chosen_set:
                candidates_to_try.append(y)
        
        random.Random().shuffle(candidates_to_try)
        
        for y in candidates_to_try:
            if time.time() - start_time > time_limit:
                break
            # Check if adding y is valid
            ok = True
            for j in range(len(chosen)):
                s = y + chosen[j]
                if counts[s] + 1 > 2:
                    ok = False
                    break
            if ok:
                s2 = 2 * y
                if counts[s2] + 1 > 2:
                    ok = False
            if ok:
                # Do the insertion
                for j in range(len(chosen)):
                    s = y + chosen[j]
                    counts[s] += 1
                counts[2 * y] += 1
                chosen.append(y)
                chosen_set.add(y)
                improved = True
                break
        
        if improved:
            continue
            
        # Try swaps: remove x, add y
        for i in range(len(chosen)):
            if time.time() - start_time > time_limit:
                break
            x = chosen[i]
            # Temporarily remove x
            for j in range(len(chosen)):
                if j != i:
                    s = x + chosen[j]
                    counts[s] -= 1
            counts[2 * x] -= 1
            
            # Try to add some y not in chosen
            chosen_set = set(chosen)
            # Focus on candidates that might fit: try values near existing gaps
            candidate_ys = []
            for y in range(1, 1001):
                if y not in chosen_set:
                    candidate_ys.append(y)
            
            random.Random().shuffle(candidate_ys)
            
            swap_done = False
            for y in candidate_ys:
                if time.time() - start_time > time_limit:
                    break
                # Check if adding y is valid
                ok = True
                for j in range(len(chosen)):
                    if j == i:
                        continue
                    s = y + chosen[j]
                    if counts[s] + 1 > 2:
                        ok = False
                        break
                if ok:
                    s2 = 2 * y
                    if counts[s2] + 1 > 2:
                        ok = False
                if ok:
                    # Do the swap
                    for j in range(len(chosen)):
                        if j == i:
                            continue
                        s = y + chosen[j]
                        counts[s] += 1
                    counts[2 * y] += 1
                    chosen[i] = y
                    chosen_set.discard(x)
                    chosen_set.add(y)
                    improved = True
                    swap_done = True
                    break
            
            if not swap_done:
                # Restore x
                for j in range(len(chosen)):
                    if j != i:
                        s = x + chosen[j]
                        counts[s] += 1
                counts[2 * x] += 1
    
    return sorted(chosen)

def _structured_greedy(seed_val=0):
    """Try different structured orderings."""
    rng = random.Random(seed_val)
    N = 1000
    
    # Strategy 1: Ascending
    cand1 = list(range(1, N + 1))
    
    # Strategy 2: Descending
    cand2 = list(range(N, 0, -1))
    
    # Strategy 3: Random
    cand3 = list(range(1, N + 1))
    rng.shuffle(cand3)
    
    # Strategy 4: Bias towards smaller numbers
    cand4 = []
    for i in range(1, N + 1):
        # Weight towards smaller numbers
        weight = 1.0 / (1 + i * 0.01)
        cand4.append((weight, i))
    rng.shuffle(cand4)
    cand4 = [x[1] for x in cand4]
    
    # Strategy 5: Sparse then fill
    cand5 = list(range(1, N + 1, 3)) + list(range(2, N + 1, 3))
    
    # Strategy 6: Random with restart bias
    cand6 = list(range(1, N + 1))
    for i in range(N - 1, 0, -1):
        j = rng.randint(0, i)
        cand6[i], cand6[j] = cand6[j], cand6[i]
    
    best = []
    best_size = 0
    
    for cand in [cand1, cand2, cand3, cand4, cand5, cand6]:
        chosen, _ = _greedy_build(cand)
        if len(chosen) > best_size:
            best = chosen
            best_size = len(chosen)
    
    return best

def _greedy_random_fast(seed, N=1000):
    """Fast randomized greedy using array-based counts."""
    rng = random.Random(seed)
    max_sum = 2 * N
    cnt = [0] * (max_sum + 1)
    S = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        # Check if adding x is valid
        if cnt[2 * x] > 1:
            continue
        ok = True
        for y in S:
            if cnt[x + y] > 1:
                ok = False
                break
        if not ok:
            continue
        # Add x
        S.add(x)
        cnt[2 * x] += 1
        for y in S:
            if y != x:
                cnt[x + y] += 1
    
    return S, cnt

def _local_search_array(S, cnt, start_time, time_limit, N=1000, max_sum=2000):
    """Array-based local search for speed."""
    S_list = list(S)
    best_size = len(S_list)
    best_S = set(S_list)
    
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit - 2:
            break
        
        # Try insertions
        added = False
        for _ in range(20):
            v = random.randint(1, N)
            if v in S:
                continue
            if cnt[2 * v] > 1:
                continue
            ok = True
            for y in S:
                if cnt[v + y] > 1:
                    ok = False
                    break
            if not ok:
                continue
            # Add v
            S.add(v)
            cnt[2 * v] += 1
            for y in S:
                if y != v:
                    cnt[v + y] += 1
            added = True
            break
        
        if added:
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            no_improve = 0
        else:
            # Try swaps
            swapped = False
            for _ in range(20):
                if not S:
                    break
                u = random.choice(list(S))
                v = random.randint(1, N)
                if v in S or v == u:
                    continue
                
                # Remove u
                S.remove(u)
                cnt[2 * u] -= 1
                for y in S:
                    cnt[u + y] -= 1
                
                # Check if v can be added
                if cnt[2 * v] > 1:
                    # Revert
                    S.add(u)
                    cnt[2 * u] += 1
                    for y in S:
                        if y != u:
                            cnt[u + y] += 1
                    continue
                ok = True
                for y in S:
                    if cnt[v + y] > 1:
                        ok = False
                        break
                if not ok:
                    # Revert
                    S.add(u)
                    cnt[2 * u] += 1
                    for y in S:
                        if y != u:
                            cnt[u + y] += 1
                    continue
                
                # Add v
                S.add(v)
                cnt[2 * v] += 1
                for y in S:
                    if y != v:
                        cnt[v + y] += 1
                swapped = True
                break
            
            if swapped:
                no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                # Shake: remove 2 random elements
                if len(S) > 2:
                    to_remove = random.sample(list(S), 2)
                    for u in to_remove:
                        S.remove(u)
                        cnt[2 * u] -= 1
                        for y in S:
                            cnt[u + y] -= 1
                no_improve = 0
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    N = 1000
    max_sum = 2 * N
    
    best = []
    best_size = 0
    best_S = set()
    best_cnt = [0] * (max_sum + 1)
    
    # Strategy 1: Standard greedy (increasing order)
    candidates = list(range(1, N + 1))
    chosen1, _ = _greedy_build(candidates)
    best = chosen1[:]
    best_size = len(best)
    best_S = set(best)
    # Rebuild counts
    for i in range(len(best)):
        for j in range(i, len(best)):
            best_cnt[best[i] + best[j]] += 1
    
    # Strategy 2: Try structured approaches
    if time.time() < deadline - 10.0:
        for seed in range(10):
            if time.time() > deadline - 10.0:
                break
            chosen_s = _structured_greedy(seed)
            if len(chosen_s) > best_size:
                best = chosen_s[:]
                best_size = len(best)
                best_S = set(best)
                best_cnt = [0] * (max_sum + 1)
                for i in range(len(best)):
                    for j in range(i, len(best)):
                        best_cnt[best[i] + best[j]] += 1
    
    # Strategy 3: Fast randomized greedy using arrays
    for seed in range(30):
        if time.time() > deadline - 20.0:
            break
        S, cnt = _greedy_random_fast(seed, N)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            best_cnt = cnt
            best = list(S)
    
    # Strategy 4: Try some random orderings with original greedy
    rng = random.Random(42)
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        # Create a shuffled candidate list with some structure
        cand = list(range(1, N + 1))
        # Mix of random and structured
        if restart % 3 == 0:
            rng.shuffle(cand)
        elif restart % 3 == 1:
            # Bias towards smaller
            weights = [1.0 / (1 + i * 0.02) for i in range(N)]
            indexed = list(zip(weights, range(1, N + 1)))
            rng.shuffle(indexed)
            cand = [x[1] for x in indexed]
        else:
            # Random with some ordering
            half = N // 2
            first_half = list(range(1, half + 1))
            second_half = list(range(half + 1, N + 1))
            rng.shuffle(first_half)
            rng.shuffle(second_half)
            cand = first_half + second_half
        
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
            best_S = set(best)
            best_cnt = [0] * (max_sum + 1)
            for i in range(len(best)):
                for j in range(i, len(best)):
                    best_cnt[best[i] + best[j]] += 1
    
    # Local search on the best solution using array-based method
    if time.time() < deadline - 5.0:
        ls_time = min(30.0, deadline - time.time() - 2.0)
        if ls_time > 1.0:
            final_S = _local_search_array(best_S, best_cnt, start_time, ls_time, N, max_sum)
            if len(final_S) > best_size:
                best_S = final_S
                best_size = len(best_S)
                best = list(best_S)
                best_cnt = [0] * (max_sum + 1)
                for i in range(len(best)):
                    for j in range(i, len(best)):
                        best_cnt[best[i] + best[j]] += 1
    
    # Final verification and cleanup
    if not _is_valid_b2(best):
        # Fallback to standard greedy result
        best = list(range(1, 1001))
        chosen1, _ = _greedy_build(best)
        best = chosen1[:]
    
    return best
# EVOLVE-BLOCK-END
