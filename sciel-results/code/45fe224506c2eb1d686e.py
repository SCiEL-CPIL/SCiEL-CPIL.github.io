import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 148.0

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

    def check_valid(idx_set):
        cnt = np.zeros(space, dtype=np.int32)
        for i in idx_set:
            cnt[nbrs[i]] += 1
        return bool((cnt > 0).all())

    # Greedy set cover to find initial solution
    def greedy_set_cover():
        uncovered = np.ones(space, dtype=bool)
        selected = []
        remaining = list(range(space))
        while uncovered.any() and time.time() < deadline - 10.0:
            best_idx = -1
            best_count = -1
            if len(remaining) > 150:
                sample = random.sample(remaining, 150)
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
            if best_idx in remaining:
                remaining.remove(best_idx)
        
        return selected

    # Greedy deletion: remove codewords that are not essential
    def greedy_delete(idx_set, rng):
        cur = list(idx_set)
        cnt = np.zeros(space, dtype=np.int32)
        for i in cur:
            cnt[nbrs[i]] += 1
        
        # Sort by coverage count (least covered first), then shuffle
        coverage_counts = {}
        for i in cur:
            coverage_counts[i] = cnt[nbrs[i]].sum()
        
        order = sorted(cur, key=lambda x: coverage_counts[x])
        rng.shuffle(order)
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for c in order:
                if c not in cur:
                    continue
                nb = nbrs[c]
                if (cnt[nb] <= 1).any():
                    continue  # essential
                cur.remove(c)
                cnt[nb] -= 1
                changed = True
        
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
            
            for c in non_essential[:3]:  # Try a few
                if time.time() > deadline - 3.0:
                    break
                
                nb = nbrs[c]
                
                # Try moving c to a random location
                under_covered = np.where(cnt <= 1)[0]
                
                candidates = []
                if len(under_covered) > 0:
                    sample_size = min(40, len(under_covered))
                    sample_under = random.sample(under_covered.tolist(), sample_size)
                    for u in sample_under:
                        for x in range(space):
                            if u in nbrs[x] and x not in cur_set:
                                candidates.append(x)
                                if len(candidates) > 60:
                                    break
                        if len(candidates) > 60:
                            break
                
                if not candidates:
                    candidates = [x for x in range(space) if x not in cur_set][:40]
                
                rng.shuffle(candidates)
                
                for x in candidates[:25]:
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
            
            rng.shuffle(non_essential)
            for i in range(min(3, len(non_essential))):
                for j in range(i+1, min(len(non_essential), i+3)):
                    c1 = non_essential[i]
                    c2 = non_essential[j]
                    
                    if time.time() > deadline - 3.0:
                        break
                    
                    candidates = [x for x in range(space) if x not in cur_set][:40]
                    rng.shuffle(candidates)
                    
                    for x in candidates[:15]:
                        # Simulate: remove c1, add x
                        cnt[nbrs[c1]] -= 1
                        cnt[nbrs[x]] += 1
                        
                        if not (cnt > 0).all():
                            cnt[nbrs[c1]] += 1
                            cnt[nbrs[x]] -= 1
                            continue
                        
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

    # Initial solution
    initial = greedy_set_cover()
    
    best = None
    best_len = 10 ** 9
    if check_valid(initial):
        best = list(initial)
        best_len = len(initial)
    
    # Multiple restarts with different strategies
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    for s in seeds:
        if time.time() > deadline - 15.0:
            break
        
        rng = random.Random(s)
        
        if best is not None and s > 0:
            start_set = list(best)
        else:
            start_set = list(initial)
        
        res = greedy_delete(start_set, rng)
        res = local_search(res, rng, max_iters=150)
        res = two_opt(res, rng, max_iters=80)
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
        return [i2w(i) for i in initial]
    return [i2w(i) for i in best]
