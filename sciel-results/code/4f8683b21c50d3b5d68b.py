import numpy as np
import itertools
import time
import random

Q = 3
N = 6
NUM_WORDS = Q ** N

all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute ball masks (boolean) and ball index lists
ball_masks = []
ball_idx_lists = []
for i in range(NUM_WORDS):
    c = all_words[i]
    mask = np.zeros(NUM_WORDS, dtype=bool)
    mask[i] = True
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            nb = c.copy()
            nb[pos] = val
            idx = word_to_idx[tuple(nb)]
            mask[idx] = True
    ball_masks.append(mask)
    ball_idx_lists.append(np.where(mask)[0])

# Precompute ball_masks as a 2D array for vectorized operations
ball_matrix = np.array(ball_masks, dtype=np.int8)  # shape (729, 729)

def get_tetracode_code():
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 115

    best_code = get_tetracode_code()
    best_size = len(best_code)

    def greedy_cover():
        """Vectorized greedy: pick codeword covering most uncovered words."""
        uncovered = np.ones(NUM_WORDS, dtype=np.int32)
        code = []
        for _ in range(NUM_WORDS):
            if uncovered.sum() == 0:
                break
            scores = ball_matrix.dot(uncovered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            uncovered[ball_masks[c]] = 0
        return code

    def greedy_randomized():
        """Randomized greedy: pick from top candidates."""
        uncovered = np.ones(NUM_WORDS, dtype=np.int32)
        code = []
        for _ in range(NUM_WORDS):
            if uncovered.sum() == 0:
                break
            scores = ball_matrix.dot(uncovered)
            max_s = scores.max()
            if max_s == 0:
                break
            threshold = max_s * 0.7
            top = np.where(scores >= threshold)[0]
            c = top[np.random.randint(len(top))]
            code.append(c)
            uncovered[ball_masks[c]] = 0
        return code

    def reduction(code):
        """Remove redundant codewords (those whose ball is fully covered by others)."""
        code = list(code)
        if not code:
            return code
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_masks[c]] += 1
        
        changed = True
        while changed:
            changed = False
            indices = list(range(len(code)))
            np.random.shuffle(indices)
            for idx in indices:
                c = code[idx]
                if np.all(cov_counts[ball_masks[c]] > 1):
                    cov_counts[ball_masks[c]] -= 1
                    code.pop(idx)
                    changed = True
                    break
        return code

    def local_search(code, max_iters=20000):
        """Local search with remove, add, swap, and pair-exchange moves."""
        code = list(code)
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_masks[c]] += 1
        
        best_code = list(code)
        best_unc = (cov_counts == 0).sum()
        best_size = len(code)
        
        T = 15.0
        cool = (0.05 / 15.0) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            mt = np.random.randint(4)
            
            if mt == 0 and len(code) > 0:
                # Remove a codeword
                idx = np.random.randint(len(code))
                r = code[idx]
                would_lose = (cov_counts[ball_masks[r]] == 1).sum()
                delta_unc = would_lose
                delta_size = -1
                if delta_unc < 0 or (delta_unc == 0 and delta_size < 0) or np.random.rand() < np.exp(-(1000.0 * delta_unc + delta_size) / T):
                    code.pop(idx)
                    cov_counts[ball_masks[r]] -= 1
                    unc = (cov_counts == 0).sum()
                    if unc < best_unc or (unc == best_unc and len(code) < best_size):
                        best_unc = unc
                        best_size = len(code)
                        best_code = list(code)
            
            elif mt == 1:
                # Add a codeword
                a = np.random.randint(NUM_WORDS)
                if a in code:
                    continue
                new_cov = (cov_counts[ball_masks[a]] == 0).sum()
                delta_unc = -new_cov
                delta_size = 1
                if delta_unc < 0 or (delta_unc == 0 and delta_size < 0) or np.random.rand() < np.exp(-(1000.0 * delta_unc + delta_size) / T):
                    code.append(a)
                    cov_counts[ball_masks[a]] += 1
                    unc = (cov_counts == 0).sum()
                    if unc < best_unc or (unc == best_unc and len(code) < best_size):
                        best_unc = unc
                        best_size = len(code)
                        best_code = list(code)
            
            elif mt == 2 and len(code) > 0:
                # Swap: remove r, add a
                idx = np.random.randint(len(code))
                r = code[idx]
                a = np.random.randint(NUM_WORDS)
                if a == r or a in code:
                    continue
                
                unique_r = ball_masks[r] & ~ball_masks[a]
                unique_a = ball_masks[a] & ~ball_masks[r]
                
                L = (cov_counts[unique_r] == 1).sum()
                Nv = (cov_counts[unique_a] == 0).sum()
                
                delta_unc = L - Nv
                if delta_unc < 0 or np.random.rand() < np.exp(-1000.0 * delta_unc / T):
                    code[idx] = a
                    cov_counts[ball_masks[r]] -= 1
                    cov_counts[ball_masks[a]] += 1
                    unc = (cov_counts == 0).sum()
                    if unc < best_unc or (unc == best_unc and len(code) < best_size):
                        best_unc = unc
                        best_size = len(code)
                        best_code = list(code)
            
            elif mt == 3 and len(code) >= 2:
                # Pair exchange: remove r1, r2; add a1, a2
                idx1 = np.random.randint(len(code))
                idx2 = np.random.randint(len(code) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                r1 = code[idx1]
                r2 = code[idx2]
                
                a1 = np.random.randint(NUM_WORDS)
                a2 = np.random.randint(NUM_WORDS)
                if a1 == r1 or a1 == r2 or a2 == r1 or a2 == r2 or a1 in code or a2 in code or a1 == a2:
                    continue
                
                ball_r1_r2 = ball_masks[r1] | ball_masks[r2]
                ball_a1_a2 = ball_masks[a1] | ball_masks[a2]
                
                lose_mask = ball_r1_r2 & ~ball_a1_a2
                gain_mask = ball_a1_a2 & ~ball_r1_r2
                
                L = (cov_counts[lose_mask] == 1).sum()
                Nv = (cov_counts[gain_mask] == 0).sum()
                
                delta_unc = L - Nv
                if delta_unc < 0 or np.random.rand() < np.exp(-1000.0 * delta_unc / T):
                    code[idx1] = a1
                    code[idx2] = a2
                    cov_counts[ball_masks[r1]] -= 1
                    cov_counts[ball_masks[r2]] -= 1
                    cov_counts[ball_masks[a1]] += 1
                    cov_counts[ball_masks[a2]] += 1
                    unc = (cov_counts == 0).sum()
                    if unc < best_unc or (unc == best_unc and len(code) < best_size):
                        best_unc = unc
                        best_size = len(code)
                        best_code = list(code)
            
            T *= cool
        
        return best_code

    def entropy_guided_reduction(code):
        """
        Use coverage entropy to identify redundant codewords.
        Codewords covering points with low marginal importance (high overlap)
        are more likely redundant.
        """
        code = list(code)
        if len(code) < 2:
            return code
        
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_masks[c]] += 1
        
        # Compute importance: for each point, how "critical" is it?
        # Points covered by exactly 1 codeword are most critical
        # Points covered by many are less critical
        # For each codeword, compute sum of "criticality" of points it uniquely covers
        
        # Marginal importance of removing codeword i:
        # Number of points that would become uncovered (cov_count becomes 0)
        # Plus a penalty for points that would go from 2 to 1 (reduced redundancy)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle to break symmetry
            order = list(range(len(code)))
            np.random.shuffle(order)
            
            for idx in order:
                if time.time() > deadline - 5:
                    break
                c = code[idx]
                # Compute delta: how many points lose their only cover
                would_uncover = (cov_counts[ball_masks[c]] == 1).sum()
                if would_uncover == 0:
                    # Safe to remove
                    cov_counts[ball_masks[c]] -= 1
                    code.pop(idx)
                    improved = True
                    break
        
        return code

    def smooth_local_search(code, max_iters=15000):
        """
        Local search with smooth surrogate: instead of hard 0/1 coverage,
        use a soft penalty for under-coverage. This allows the search to
        explore configurations where some points are temporarily under-covered
        but the overall "energy" is lower.
        """
        code = list(code)
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_masks[c]] += 1
        
        best_code = list(code)
        best_energy = float('inf')
        
        # Energy: sum over points of max(0, 1 - cov_count)^2 + 0.1 * len(code)
        # This smoothly penalizes uncovered points and large code size
        
        def compute_energy(cc, size):
            uncovered = np.maximum(0, 1 - cc)
            return float(np.sum(uncovered ** 2) + 0.1 * size)
        
        best_energy = compute_energy(cov_counts, len(code))
        
        T = 5.0
        cool = (0.1 / 5.0) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            mt = np.random.randint(3)
            
            if mt == 0 and len(code) > 0:
                # Remove
                idx = np.random.randint(len(code))
                r = code[idx]
                
                # Compute delta energy
                ball_r = ball_masks[r]
                old_cc = cov_counts[ball_r]
                new_cc = old_cc - 1
                
                old_pen = np.sum(np.maximum(0, 1 - old_cc) ** 2)
                new_pen = np.sum(np.maximum(0, 1 - new_cc) ** 2)
                delta_e = (new_pen - old_pen) + 0.1 * (-1)
                
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                    code.pop(idx)
                    cov_counts[ball_r] -= 1
                    e = compute_energy(cov_counts, len(code))
                    if e < best_energy:
                        best_energy = e
                        best_code = list(code)
            
            elif mt == 1:
                # Add
                a = np.random.randint(NUM_WORDS)
                if a in code:
                    continue
                
                ball_a = ball_masks[a]
                old_cc = cov_counts[ball_a]
                new_cc = old_cc + 1
                
                old_pen = np.sum(np.maximum(0, 1 - old_cc) ** 2)
                new_pen = np.sum(np.maximum(0, 1 - new_cc) ** 2)
                delta_e = (new_pen - old_pen) + 0.1 * (1)
                
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                    code.append(a)
                    cov_counts[ball_a] += 1
                    e = compute_energy(cov_counts, len(code))
                    if e < best_energy:
                        best_energy = e
                        best_code = list(code)
            
            elif mt == 2 and len(code) > 0:
                # Swap
                idx = np.random.randint(len(code))
                r = code[idx]
                a = np.random.randint(NUM_WORDS)
                if a == r or a in code:
                    continue
                
                # Compute delta for swap
                ball_r = ball_masks[r]
                ball_a = ball_masks[a]
                
                # Points affected: union of ball_r and ball_a
                affected = ball_r | ball_a
                
                old_cc = cov_counts[affected]
                # After removing r and adding a:
                new_cc = old_cc.copy()
                new_cc[ball_r] -= 1
                new_cc[ball_a] += 1
                
                old_pen = np.sum(np.maximum(0, 1 - old_cc) ** 2)
                new_pen = np.sum(np.maximum(0, 1 - new_cc) ** 2)
                delta_e = new_pen - old_pen  # size unchanged
                
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                    code[idx] = a
                    cov_counts[ball_r] -= 1
                    cov_counts[ball_a] += 1
                    e = compute_energy(cov_counts, len(code))
                    if e < best_energy:
                        best_energy = e
                        best_code = list(code)
            
            T *= cool
        
        # Ensure final result is valid (all points covered)
        final_cov = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in best_code:
            final_cov[ball_masks[c]] += 1
        if np.all(final_cov > 0):
            return best_code
        else:
            # Fallback: return original if smooth search produced invalid result
            return code

    # Phase 1: Generate diverse initial solutions
    inits = []
    for i in range(20):
        if time.time() > deadline - 30:
            break
        if i < 8:
            sol = greedy_cover()
        else:
            sol = greedy_randomized()
        sol = reduction(sol)
        inits.append(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_code = [list(all_words[c]) for c in sol]
    
    # Phase 2: Local search from best initial solutions
    inits.sort(key=len)
    for sol in inits[:12]:
        if time.time() > deadline - 20:
            break
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            res = local_search(sol, max_iters=8000)
            res = reduction(res)
            if len(res) < best_size:
                best_size = len(res)
                best_code = [list(all_words[c]) for c in res]
    
    # Phase 3: Smooth local search for refinement
    if best_size <= 75 and time.time() < deadline - 10:
        # Convert best_code to indices
        code_idx = []
        for c in best_code:
            c_t = tuple(c)
            if c_t in word_to_idx:
                code_idx.append(word_to_idx[c_t])
        
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            res = smooth_local_search(code_idx, max_iters=12000)
            res = reduction(res)
            if len(res) < best_size:
                best_size = len(res)
                best_code = [list(all_words[c]) for c in res]
    
    # Phase 4: Entropy-guided final reduction
    if time.time() < deadline - 3:
        code_idx = []
        for c in best_code:
            c_t = tuple(c)
            if c_t in word_to_idx:
                code_idx.append(word_to_idx[c_t])
        
        res = entropy_guided_reduction(code_idx)
        if len(res) < best_size:
            best_size = len(res)
            best_code = [list(all_words[c]) for c in res]
    
    return best_code
