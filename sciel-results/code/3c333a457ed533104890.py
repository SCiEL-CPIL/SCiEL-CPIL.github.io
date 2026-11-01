import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 145.0

    space = q ** n

    def w2i(w):
        i = 0
        for v in w:
            i = i * q + v
        return i

    def i2w(i):
        w = [0] * n
        for j in range(n - 1, -1, -1):
            w[j] = i % q
            i //= q
        return w

    # Precompute neighbor index lists for every word in the space (distance <= radius).
    # For radius 1: the word itself plus all single-position changes.
    if radius >= 1:
        nbrs = []
        for i in range(space):
            w = i2w(i)
            lst = [i]
            for p in range(n):
                ov = w[p]
                for nv in range(q):
                    if nv != ov:
                        w[p] = nv
                        lst.append(w2i(w))
                w[p] = ov
            nbrs.append(lst)
    else:
        nbrs = [[i] for i in range(space)]

    # Build a good initial solution using greedy set cover
    # Start with all possible words, greedily pick the one covering most uncovered
    def greedy_set_cover():
        uncovered = np.ones(space, dtype=bool)
        selected = []
        remaining = list(range(space))
        # Precompute coverage counts for efficiency
        # For each word, how many uncovered words it covers
        while uncovered.any() and time.time() < deadline - 5.0:
            best_idx = -1
            best_count = -1
            # Sample a subset for speed if too many remain
            if len(remaining) > 100:
                sample = random.sample(remaining, 100)
            else:
                sample = remaining
            
            for idx in sample:
                nb = nbrs[idx]
                count = uncovered[nb].sum()
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            uncovered[nbrs[best_idx]] = False
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected

    # Better initial: use structured construction + greedy refinement
    # Tetracode-like structure for q=3, n=6
    if q == 3 and n == 6:
        # Use a known good structure: partition into two halves
        # Left half (3 positions): use a (3,3,1) covering code - all 27 words or a subset
        # Right half (3 positions): use a (3,3,1) covering code
        # Actually, let's use a smarter approach:
        # For K_3(3,1), we need 9 words (all of {0,1,2}^3)
        # But we can do better with a covering code structure
        
        # Use greedy set cover to find a good initial solution
        initial = greedy_set_cover()
    else:
        # Generic product construction
        if n >= 4:
            # Simple product: use all words for first n-4 positions, 
            # and a covering code for last 4 positions
            if n - 4 >= 0:
                prefix_space = q ** (n - 4)
                suffix_space = q ** 4
                # For suffix, use greedy on 4-position space
                suffix_nbrs = []
                for i in range(suffix_space):
                    w = [0] * 4
                    for j in range(3, -1, -1):
                        w[j] = i % q
                        i //= q
                    lst = [i * q**(n-4) + w2i(w)]
                    for p in range(4):
                        ov = w[p]
                        for nv in range(q):
                            if nv != ov:
                                w[p] = nv
                                lst.append(w2i(w))
                        w[p] = ov
                    suffix_nbrs.append(lst)
                
                # Greedy on suffix
                uncovered_suffix = np.ones(suffix_space, dtype=bool)
                suffix_selected = []
                remaining_suffix = list(range(suffix_space))
                while uncovered_suffix.any() and time.time() < deadline - 10.0:
                    best_idx = -1
                    best_count = -1
                    if len(remaining_suffix) > 50:
                        sample = random.sample(remaining_suffix, 50)
                    else:
                        sample = remaining_suffix
                    
                    for idx in sample:
                        count = uncovered_suffix[suffix_nbrs[idx]].sum()
                        if count > best_count:
                            best_count = count
                            best_idx = idx
                    
                    if best_idx == -1:
                        break
                    
                    uncovered_suffix[suffix_nbrs[best_idx]] = False
                    suffix_selected.append(best_idx)
                    remaining_suffix.remove(best_idx)
                
                # Combine: for each prefix, add all suffix selections
                initial = []
                for pref in range(prefix_space):
                    for suff in suffix_selected:
                        initial.append(pref * suffix_space + suff)
            else:
                initial = greedy_set_cover()
        else:
            initial = greedy_set_cover()
    
    initial_idx = list(initial)
    
    def check_valid(idx_set):
        cnt = np.zeros(space, dtype=np.int32)
        for i in idx_set:
            cnt[nbrs[i]] += 1
        return bool((cnt > 0).all())

    best = None
    best_len = 10 ** 9
    if check_valid(initial_idx):
        best = list(initial_idx)
        best_len = len(initial_idx)
    
    # Greedy deletion: remove codewords that are not essential
    def greedy_delete(idx_set, rng):
        cur = list(idx_set)
        cnt = np.zeros(space, dtype=np.int32)
        for i in cur:
            cnt[nbrs[i]] += 1
        
        # Sort by coverage count (least covered first)
        coverage_counts = {}
        for i in cur:
            coverage_counts[i] = cnt[nbrs[i]].sum()
        
        # Try to delete in random order, but prioritize low-coverage codewords
        order = sorted(cur, key=lambda x: coverage_counts[x])
        rng.shuffle(order)
        
        for c in order:
            if c not in cur:
                continue
            nb = nbrs[c]
            if (cnt[nb] <= 1).any():
                continue  # essential
            cur.remove(c)
            cnt[nb] -= 1
        
        return cur

    # Local search: try to replace codewords to enable more deletions
    def local_search(cur, rng, max_iters=200):
        if time.time() > deadline - 5.0:
            return cur
        
        cnt = np.zeros(space, dtype=np.int32)
        for i in cur:
            cnt[nbrs[i]] += 1
        cur_set = set(cur)
        
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            if time.time() > deadline - 3.0:
                break
            
            # Find non-essential codewords
            non_essential = [c for c in cur if not (cnt[nbrs[c]] <= 1).any()]
            if not non_essential:
                break
            
            rng.shuffle(non_essential)
            
            for c in non_essential[:5]:  # Try a few
                if time.time() > deadline - 3.0:
                    break
                
                nb = nbrs[c]
                
                # Try moving c to a random location
                # Focus on locations that cover many currently under-covered words
                under_covered = np.where(cnt <= 1)[0]
                
                # Sample candidate locations
                candidates = []
                if len(under_covered) > 0:
                    # Pick from under-covered regions
                    sample_size = min(30, len(under_covered))
                    sample_under = random.sample(under_covered.tolist(), sample_size)
                    for u in sample_under:
                        # Find codewords that cover u
                        for x in range(space):
                            if u in nbrs[x] and x not in cur_set:
                                candidates.append(x)
                                if len(candidates) > 50:
                                    break
                        if len(candidates) > 50:
                            break
                
                if not candidates:
                    candidates = [x for x in range(space) if x not in cur_set][:50]
                
                rng.shuffle(candidates)
                
                for x in candidates[:20]:
                    if time.time() > deadline - 3.0:
                        break
                    
                    # Simulate: remove c, add x
                    cnt[nb] -= 1
                    cnt[nbrs[x]] += 1
                    
                    if not (cnt > 0).all():
                        cnt[nb] += 1
                        cnt[nbrs[x]] -= 1
                        continue
                    
                    # Try to delete more
                    new_cur = [y for y in cur if y != c] + [x]
                    new_cnt = cnt.copy()
                    
                    deletions = 0
                    for y in list(new_cur):
                        if y == x:
                            continue
                        nby = nbrs[y]
                        if (new_cnt[nby] <= 1).any():
                            continue
                        new_cnt[nby] -= 1
                        new_cur.remove(y)
                        deletions += 1
                    
                    if len(new_cur) < len(cur):
                        # Commit
                        cur = new_cur
                        cur_set = set(cur)
                        cnt = np.zeros(space, dtype=np.int32)
                        for y in cur:
                            cnt[nbrs[y]] += 1
                        improved = True
                        break
                    else:
                        # Revert
                        cnt[nb] += 1
                        cnt[nbrs[x]] -= 1
        
        return cur

    # Two-opt style: swap two codewords
    def two_opt(cur, rng, max_iters=100):
        if time.time() > deadline - 5.0:
            return cur
        
        cnt = np.zeros(space, dtype=np.int32)
        for i in cur:
            cnt[nbrs[i]] += 1
        cur_set = set(cur)
        
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            if time.time() > deadline - 3.0:
                break
            
            non_essential = [c for c in cur if not (cnt[nbrs[c]] <= 1).any()]
            if len(non_essential) < 2:
                break
            
            # Try swapping pairs
            rng.shuffle(non_essential)
            for i in range(min(3, len(non_essential))):
                for j in range(i+1, min(len(non_essential), i+3)):
                    c1 = non_essential[i]
                    c2 = non_essential[j]
                    
                    if time.time() > deadline - 3.0:
                        break
                    
                    # Try replacing c1 with a random word
                    candidates = [x for x in range(space) if x not in cur_set][:30]
                    rng.shuffle(candidates)
                    
                    for x in candidates[:10]:
                        # Simulate: remove c1, add x
                        cnt[nbrs[c1]] -= 1
                        cnt[nbrs[x]] += 1
                        
                        if not (cnt > 0).all():
                            cnt[nbrs[c1]] += 1
                            cnt[nbrs[x]] -= 1
                            continue
                        
                        # Try to delete c2 or others
                        new_cur = [y for y in cur if y != c1] + [x]
                        new_cnt = cnt.copy()
                        
                        deletions = 0
                        for y in list(new_cur):
                            if y == x:
                                continue
                            nby = nbrs[y]
                            if (new_cnt[nby] <= 1).any():
                                continue
                            new_cnt[nby] -= 1
                            new_cur.remove(y)
                            deletions += 1
                        
                        if len(new_cur) < len(cur):
                            cur = new_cur
                            cur_set = set(cur)
                            cnt = np.zeros(space, dtype=np.int32)
                            for y in cur:
                                cnt[nbrs[y]] += 1
                            improved = True
                            break
                        else:
                            cnt[nbrs[c1]] += 1
                            cnt[nbrs[x]] -= 1
                    
                    if improved:
                        break
            
            if improved:
                break
        
        return cur

    # Multiple restarts with different strategies
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    for s in seeds:
        if time.time() > deadline - 15.0:
            break
        
        rng = random.Random(s)
        
        # Start from current best or initial
        if best is not None and s > 0:
            start_set = list(best)
        else:
            start_set = list(initial_idx)
        
        # Greedy delete
        res = greedy_delete(start_set, rng)
        
        # Local search
        res = local_search(res, rng, max_iters=150)
        
        # Two-opt
        res = two_opt(res, rng, max_iters=80)
        
        # Final greedy delete
        res = greedy_delete(res, rng)
        
        if check_valid(res) and len(res) < best_len:
            best = res
            best_len = len(res)
    
    # Final polish on best solution
    if best is not None and time.time() < deadline - 5.0:
        rng = random.Random(999)
        best = greedy_delete(best, rng)
        best = local_search(best, rng, max_iters=100)
        best = greedy_delete(best, rng)
        if check_valid(best):
            best_len = len(best)
    
    if best is None:
        return [i2w(i) for i in initial_idx]
    return [i2w(i) for i in best]
