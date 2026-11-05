import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    N = 512
    
    # 1. Precomputation
    # Sieve for primes up to 511
    is_prime = [True] * (N)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
    
    # List of primes
    primes = [i for i in range(2, N) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Bitmask for prime factors
    # Bit i corresponds to primes[i]
    masks = [0] * N
    for a in range(2, N):
        m = 0
        temp = a
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                m |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            m |= (1 << prime_to_idx[temp])
        masks[a] = m
    # Note: mask[1] = 0, which is correct (1 is coprime to everything)
    
    # Scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
        
    # Candidates sorted by score descending
    candidates = list(range(1, N))
    candidates.sort(key=lambda x: scores[x], reverse=True)
    
    def get_score(S):
        return sum(scores[a] for a in S)
        
    def greedy_init():
        S = []
        used_mask = 0
        for a in candidates:
            if masks[a] & used_mask:
                continue
            S.append(a)
            used_mask |= masks[a]
        return S, used_mask
        
    def local_search(S, used_mask, max_iters=20000):
        current_score = get_score(S)
        S_set = set(S)
        
        # Precompute list of elements for fast iteration
        S_list = list(S)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 5:
                break
            improved = False
            iters += 1
            
            # 1. Try Add
            # Find best a not in S such that masks[a] & used_mask == 0
            best_add = -1
            best_add_score = 0.0
            for a in candidates:
                if a in S_set:
                    continue
                if masks[a] & used_mask:
                    continue
                if scores[a] > best_add_score:
                    best_add_score = scores[a]
                    best_add = a
                # Since candidates are sorted by score desc, we can break early?
                # No, because the first valid one is the best valid one.
                # So we can break on first valid one.
                break 
                
            if best_add != -1:
                S_list.append(best_add)
                S_set.add(best_add)
                used_mask |= masks[best_add]
                current_score += best_add_score
                improved = True
                continue
                
            # 2. Try Swap (1-for-1)
            # For each a in S, try to replace with b
            # b must be coprime to S \ {a}
            # i.e., masks[b] & (used_mask ^ masks[a]) == 0
            swap_found = False
            best_swap_delta = 0
            best_swap_a = -1
            best_swap_b = -1
            
            # We iterate through S. To speed up, we can sort S by score ascending (worst first)
            # and only check swaps for the worst few elements?
            # But a swap might free up a prime that allows a much better number.
            # Let's check all, but optimize the inner loop.
            
            for a in S_list:
                if time.time() > deadline - 5:
                    break
                mask_without_a = used_mask ^ masks[a]
                
                # Find best b not in S, coprime to mask_without_a
                best_b = -1
                best_b_score = 0.0
                for b in candidates:
                    if b in S_set:
                        continue
                    if masks[b] & mask_without_a:
                        continue
                    # Since candidates are sorted by score, the first valid b is the best
                    best_b = b
                    best_b_score = scores[b]
                    break
                    
                if best_b != -1:
                    delta = best_b_score - scores[a]
                    if delta > best_swap_delta + 1e-12:
                        best_swap_delta = delta
                        best_swap_a = a
                        best_swap_b = best_b
                        # We don't break here because we want the best swap overall, 
                        # but usually the first significant improvement is good enough.
                        # However, to be safe, we continue to find the global best swap in this iteration.
                        # Optimization: If delta is large, we might break.
            
            if best_swap_a != -1:
                # Perform swap
                idx = S_list.index(best_swap_a)
                S_list[idx] = best_swap_b
                S_set.remove(best_swap_a)
                S_set.add(best_swap_b)
                used_mask = (used_mask ^ masks[best_swap_a]) | masks[best_swap_b]
                current_score += best_swap_delta
                improved = True
                continue
                
            # 3. Try 2-for-2 Swap (limited)
            # This is expensive. We try to replace the two worst elements in S
            # with the two best available elements that fit.
            if len(S_list) >= 2:
                # Sort S by score ascending to find worst two
                S_sorted = sorted(S_list, key=lambda x: scores[x])
                a1 = S_sorted[0]
                a2 = S_sorted[1]
                
                mask_without_a1a2 = used_mask ^ masks[a1] ^ masks[a2]
                
                # Find best pair b1, b2 not in S, coprime to mask_without_a1a2 and each other
                # This is O(N^2) in worst case, but we can prune.
                # Candidates available:
                avail = [b for b in candidates if b not in S_set and (masks[b] & mask_without_a1a2) == 0]
                
                if len(avail) >= 2:
                    # Sort avail by score desc
                    avail.sort(key=lambda x: scores[x], reverse=True)
                    
                    best_pair_score = 0
                    best_pair = None
                    # Only check top K candidates to keep it fast
                    K = min(20, len(avail))
                    for i in range(K):
                        b1 = avail[i]
                        for j in range(i+1, K):
                            b2 = avail[j]
                            if masks[b1] & masks[b2]:
                                continue
                            s = scores[b1] + scores[b2]
                            if s > best_pair_score:
                                best_pair_score = s
                                best_pair = (b1, b2)
                    
                    if best_pair:
                        delta = best_pair_score - (scores[a1] + scores[a2])
                        if delta > 1e-12:
                            # Perform swap
                            S_list.remove(a1)
                            S_list.remove(a2)
                            S_set.remove(a1)
                            S_set.remove(a2)
                            
                            b1, b2 = best_pair
                            S_list.append(b1)
                            S_list.append(b2)
                            S_set.add(b1)
                            S_set.add(b2)
                            
                            used_mask = (used_mask ^ masks[a1] ^ masks[a2]) | masks[b1] | masks[b2]
                            current_score += delta
                            improved = True
                            continue

        return S_list, used_mask, current_score

    # Main Loop
    best_S, best_mask, best_score = greedy_init()
    best_S, best_mask, best_score = local_search(best_S, best_mask, max_iters=5000)
    
    # Random Restarts / Perturbations
    rng = random.Random(42)
    
    # We can also try starting from a set of just the largest primes
    # Or random valid sets
    
    for restart in range(500):
        if time.time() > deadline - 10:
            break
            
        # Perturb best solution
        S_pert = list(best_S)
        # Remove k random elements
        k = rng.randint(1, min(5, len(S_pert)))
        for _ in range(k):
            if S_pert:
                idx = rng.randrange(len(S_pert))
                S_pert.pop(idx)
        
        # Re-run greedy to fill
        S_new, mask_new = greedy_init()
        # Wait, greedy_init starts from scratch. 
        # We should merge the perturbed set into a greedy process or just run local search on the perturbed set.
        # Let's just run local search on the perturbed set.
        # But we need to ensure validity. The perturbed set is valid.
        
        # Re-calculate mask
        mask_curr = 0
        for a in S_pert:
            mask_curr |= masks[a]
            
        S_res, mask_res, score_res = local_search(S_pert, mask_curr, max_iters=2000)
        
        if score_res > best_score + 1e-12:
            best_S = S_res
            best_mask = mask_res
            best_score = score_res
            
    # Final Polish
    if time.time() < deadline - 5:
        best_S, best_mask, best_score = local_search(best_S, best_mask, max_iters=5000)
        
    return best_S

