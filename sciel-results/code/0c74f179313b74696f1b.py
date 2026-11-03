import numpy as np
import itertools
import time
import random

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls: ball_indices[c] = list of indices covered by codeword c
ball_indices = []
for i in range(NUM_WORDS):
    c = all_words[i]
    bi = [i]
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            bi.append(idx)
    ball_indices.append(np.array(bi, dtype=np.int32))

# Build ball_matrix: ball_matrix[c, w] = 1 if w is in ball(c)
ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
for c in range(NUM_WORDS):
    ball_matrix[c, ball_indices[c]] = 1

# cover_matrix[w, c] = 1 if w is in ball(c) = ball_matrix[c, w]
cover_matrix = ball_matrix.T

# Precompute distance-2 adjacency for independent set construction
# Two codewords at Hamming distance <= 2 can potentially be "close"
# For radius-1 covering, we want to avoid having two codewords too close
dist2_adj = [set() for _ in range(NUM_WORDS)]
for idx in range(NUM_WORDS):
    w = all_words[idx]
    for i in range(N):
        for j in range(i+1, N):
            for vi in range(Q):
                if vi == w[i]:
                    continue
                for vj in range(Q):
                    if vj == w[j]:
                        continue
                    nw = w.copy()
                    nw[i] = vi
                    nw[j] = vj
                    nidx = word_to_idx[tuple(nw)]
                    dist2_adj[idx].add(nidx)
                    dist2_adj[nidx].add(idx)

# Also include distance-1 neighbors in the "conflict" graph
dist_conflict = [set() for _ in range(NUM_WORDS)]
for idx in range(NUM_WORDS):
    dist_conflict[idx] = set(dist2_adj[idx])
    for nb in ball_indices[idx]:
        if nb != idx:
            dist_conflict[idx].add(nb)

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def _reduce_code(code, ball_indices):
    """Remove redundant codewords from code. Returns reduced code."""
    if not code:
        return code
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        cov_counts[ball_indices[c]] += 1
    changed = True
    while changed:
        changed = False
        for idx in range(len(code)):
            c = code[idx]
            if np.all(cov_counts[ball_indices[c]] > 1):
                cov_counts[ball_indices[c]] -= 1
                code.pop(idx)
                changed = True
                break
    return code

def _greedy_independent_set(dist_conflict, rng, deadline):
    """Greedy minimum-degree independent set on the conflict graph."""
    remaining = set(range(NUM_WORDS))
    code = []
    while remaining:
        if time.monotonic() > deadline:
            break
        # Pick vertex with minimum degree in remaining
        best_v = min(remaining, key=lambda v: len(dist_conflict[v] & remaining))
        code.append(best_v)
        remaining.discard(best_v)
        for u in dist_conflict[best_v]:
            if u in remaining:
                remaining.discard(u)
    return code

def _greedy_cover(code, ball_indices, deadline):
    """Fill gaps in coverage greedily."""
    covered = np.zeros(NUM_WORDS, dtype=bool)
    for c in code:
        covered[ball_indices[c]] = True
    
    in_code = set(code)
    while not np.all(covered):
        if time.monotonic() > deadline:
            break
        scores = ball_matrix.dot(~covered)
        scores[list(in_code)] = -1  # Don't pick already in code
        c = int(np.argmax(scores))
        if scores[c] <= 0:
            break
        code.append(c)
        in_code.add(c)
        covered[ball_indices[c]] = True
    return code

def _local_search_swap(code, ball_indices, cover_matrix, rng, deadline, max_iters=500):
    """Swap-based local search to reduce code size."""
    if not code:
        return code
    
    in_code = set(code)
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        cov_counts[ball_indices[c]] += 1
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.monotonic() > deadline:
            break
        
        # Recompute coverage counts
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
        
        # Try removing redundant words
        for idx in range(len(code)):
            c = code[idx]
            if np.all(cov_counts[ball_indices[c]] > 1):
                cov_counts[ball_indices[c]] -= 1
                code.pop(idx)
                in_code.discard(c)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove c_i, add c_j that covers all uniquely-covered words
        for idx in range(len(code)):
            c_i = code[idx]
            ball_i = ball_indices[c_i]
            unique_words = ball_i[cov_counts[ball_i] == 1]
            
            if len(unique_words) == 0:
                continue
            
            target = unique_words[0]
            cand_mask = cover_matrix[target] == 1
            cand_mask[list(in_code)] = False
            candidates = np.where(cand_mask)[0]
            
            if len(candidates) == 0:
                continue
            
            num_tries = min(20, len(candidates))
            tried = rng.choice(candidates, size=num_tries, replace=False)
            
            for c_j in tried:
                if not np.all(cover_matrix[unique_words, c_j]):
                    continue
                
                # Perform swap
                code[idx] = c_j
                in_code.discard(c_i)
                in_code.add(c_j)
                
                # Recompute and reduce
                cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
                for c in code:
                    cov_counts[ball_indices[c]] += 1
                
                # Try to remove more
                for ridx in range(len(code)):
                    rc = code[ridx]
                    if np.all(cov_counts[ball_indices[rc]] > 1):
                        cov_counts[ball_indices[rc]] -= 1
                        code.pop(ridx)
                        in_code.discard(rc)
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
    
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
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

    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin
    rng = random.Random(42)
    
    best_code = get_tetracode_code()
    best_size = len(best_code)
    
    # Phase 1: Independent set based construction
    iso_code = _greedy_independent_set(dist_conflict, rng, deadline - 30.0)
    iso_code = _greedy_cover(iso_code, ball_indices, deadline - 20.0)
    iso_code = _reduce_code(iso_code, ball_indices)
    iso_code = _local_search_swap(iso_code, ball_indices, cover_matrix, rng, deadline - 10.0)
    
    if len(iso_code) < best_size:
        best_size = len(iso_code)
        best_code = [list(all_words[c]) for c in iso_code]
    
    # Phase 2: Multiple randomized greedy + local search restarts
    restart = 0
    while time.monotonic() < deadline - 5.0:
        restart += 1
        if time.monotonic() > deadline:
            break
        
        # Temperature annealing: start broad, narrow over restarts
        temp = max(0.3, 1.0 - restart * 0.02)
        
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        
        # Greedy Phase with randomized selection
        while not np.all(covered):
            if time.monotonic() > deadline:
                break
            scores = ball_matrix.dot(~covered)
            max_score = np.max(scores)
            if max_score == 0:
                break
                
            threshold = max_score * temp
            candidates = np.where(scores >= threshold)[0]
            c = candidates[rng.randint(0, len(candidates) - 1)]
            
            code.append(c)
            covered[ball_indices[c]] = True
        
        if time.monotonic() > deadline:
            break
        
        # Reduction Phase
        code = _reduce_code(code, ball_indices)
        
        # Local Search Phase
        code = _local_search_swap(code, ball_indices, cover_matrix, rng, deadline - 5.0, max_iters=100)
        
        current_size = len(code)
        if current_size < best_size:
            best_size = current_size
            best_code = [list(all_words[c]) for c in code]
            if best_size <= 73:
                # Good improvement, continue searching
                pass
    
    return best_code
