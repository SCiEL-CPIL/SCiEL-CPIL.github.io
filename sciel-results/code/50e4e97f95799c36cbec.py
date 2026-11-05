"""Ternary covering code K_3(7,1) – vectorized greedy + aggressive local polish.

Strategy:
  1. Fast ball precomputation via vectorized index arithmetic.
  2. Vectorized greedy set cover with a max-heap for tie-breaking.
  3. Redundancy removal (repeated).
  4. 1-exchange (swap) local search with vectorized candidate scoring.
  5. Repeated polish passes until time budget is exhausted.
"""
import time
import heapq
import itertools

import numpy as np


def _ball_indices_vectorized(q, n, radius):
    """Return list of numpy arrays; ball[i] = all indices within Hamming distance <= radius of word i."""
    total = q ** n
    balls = []
    # Precompute single-position neighbor offsets
    # For each position p and value change old->new, the index delta is (new - old) * stride[p]
    strides = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)
    # neighbor_deltas[p] = list of (delta, old_val, new_val) for all changes at position p
    # But we need: for word w, the neighbor at position p with value v (v != w_p) is w + (v - w_p)*strides[p]
    
    # Precompute for each position p, the list of (delta_for_old0, ...) 
    # Actually let's precompute: for each p, for each old value a in 0..q-1, the list of new values
    # and the corresponding index offsets.
    
    # For radius 1: ball[i] = {i} union {i + (v - a)*strides[p] : p in range(n), a = i//strides[p] % q, v != a}
    
    for i in range(total):
        idx_set = [i]
        for p in range(n):
            s = strides[p]
            a = (i // s) % q
            for v in range(q):
                if v == a:
                    continue
                idx_set.append(i + (v - a) * s)
        balls.append(np.array(idx_set, dtype=np.int32))
    return balls


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 135.0  # safety margin under 150s

    total = q ** n

    # Precompute balls
    balls = _ball_indices_vectorized(q, n, radius)
    # Also build a set for O(1) membership: ball_set[i] = set of indices
    # This is memory heavy but useful for swap search
    # For 2187 words, each ball has 1 + 7*2 = 15 elements. So 2187 * 15 ~ 32k entries total.
    # We can store as frozenset or sorted list.
    ball_sets = [frozenset(b) for b in balls]
    
    # Precompute coverage count array
    covered = np.zeros(total, dtype=np.int32)
    # Heap: we want max marginal gain. Initially every word covers 15 new words.
    # Use negative for min-heap.
    # heap entries: (-gain, word_idx, version)
    # We'll use a simple approach: recompute gain lazily.
    
    # Actually, let's use a different greedy: at each step, pick the word that covers the most uncovered words.
    # gain[w] = number of currently-uncovered words in balls[w]
    # We can maintain gain incrementally.
    
    gain = np.full(total, len(balls[0]), dtype=np.int32)  # initial gain for each word
    # Initially all words are uncovered, so gain[w] = len(balls[w])
    # All balls have same size for uniform radius, so this is fine.
    
    # Max-heap via negative gain
    heap = [(-int(gain[w]), w) for w in range(total)]
    heapq.heapify(heap)
    
    chosen = []
    chosen_set = set()
    
    while True:
        if time.time() > DEADLINE - 20.0:
            break  # leave time for local search
        if len(chosen) >= total:
            break
        neg, w = heapq.heappop(heap)
        cur_gain = -neg
        if cur_gain < 1:
            break
        if w in chosen_set:
            continue
        if covered[w] > 0:
            # stale entry, skip
            continue
        # Commit word w
        chosen.append(w)
        chosen_set.add(w)
        new_covered = 0
        for c in balls[w]:
            if covered[c] == 0:
                new_covered += 1
            covered[c] += 1
        
        # Update gains: for every word v, gain[v] decreases by the number of newly covered words in balls[v]
        # This is expensive. Let's use a different approach: recompute gain only when needed.
        # Alternative: use a "lazy" heap where we verify gain on pop.
        # For simplicity and correctness, let's just push all words back with updated gains periodically.
        # Actually, the standard approach: after selecting w, for each v not yet chosen,
        # gain[v] = sum(1 for c in balls[v] if covered[c] == 0)
        # This is O(total * ball_size) per iteration = 2187 * 15 = 32805 per step.
        # With ~186 steps, that's ~6M operations. Fast in numpy.
        
        # Recompute gains vectorized
        # For each word v, gain[v] = number of c in balls[v] with covered[c] == 0
        # balls is a list of arrays. We can't easily vectorize across different-length arrays.
        # But all balls have the same length (15 for radius 1, q=3, n=7).
        # Let's stack them into a 2D array.
        
        # Actually, let's just do it in a loop but with numpy per ball.
        # For speed, precompute the ball matrix as a 2D array.
        
    # The above loop is inefficient. Let's restart with a better design.
    
    # --- Better approach: precompute ball_matrix as 2D numpy array ---
    # ball_matrix[v, j] = the j-th neighbor of word v
    ball_sizes = [len(b) for b in balls]
    max_size = max(ball_sizes)
    ball_matrix = np.zeros((total, max_size), dtype=np.int32)
    for v in range(total):
        size = len(balls[v])
        ball_matrix[v, :size] = balls[v]
    
    # Greedy set cover with vectorized gain
    covered = np.zeros(total, dtype=np.uint8)
    chosen = []
    chosen_set = set()
    
    # Initial gain: each word covers ball_size words
    init_size = len(balls[0])
    gain = np.full(total, init_size, dtype=np.int32)
    
    heap = [(-init_size, w) for w in range(total)]
    heapq.heapify(heap)
    
    while True:
        if time.time() > DEADLINE - 25.0:
            break
        if len(chosen) >= total:
            break
        
        neg, w = heapq.heappop(heap)
        if w in chosen_set:
            continue
        if covered[w] > 0:
            continue
        
        # Verify gain
        cur_gain = int(np.sum(covered[balls[w]] == 0))
        if cur_gain < 1:
            break
        if cur_gain != -neg:
            # Stale: recompute and re-push if still useful
            if cur_gain > 0:
                heapq.heappush(heap, (-cur_gain, w))
            continue
        
        # Commit
        chosen.append(w)
        chosen_set.add(w)
        for c in balls[w]:
            covered[c] = 1
        
        # Update gains: for each v not chosen, gain[v] = count of uncovered in balls[v]
        # Vectorized: for all v, compute sum(covered[ball_matrix[v]] == 0)
        # This is O(total * max_size) = 2187 * 15 = 32805
        # Do it every iteration for correctness.
        # Use advanced indexing: covered[ball_matrix] gives a 2D array of shape (total, max_size)
        # But we need to handle the padding (entries beyond actual ball size).
        # Since all balls have the same size for uniform q,n,radius, max_size == init_size.
        cov_expanded = covered[ball_matrix]  # shape (total, max_size)
        new_gain = np.sum(cov_expanded == 0, axis=1)
        
        # Push updated gains for all unchosen words
        # To avoid heap explosion, only push words with gain > 0 that aren't already in heap
        # Simpler: clear and rebuild heap each iteration? O(total log total) per step = 2187 * ~11 = 24k per step
        # With 186 steps: ~4.5M operations. Fine.
        heap = [(-int(new_gain[v]), v) for v in range(total) if v not in chosen_set and new_gain[v] > 0]
        heapq.heapify(heap)
    
    # Safety: complete cover
    for w in range(total):
        if covered[w] == 0 and w not in chosen_set:
            chosen.append(w)
            chosen_set.add(w)
            for c in balls[w]:
                covered[c] = 1
    
    # --- Local Search ---
    chosen_list = list(chosen_set)
    
    def _coverage_array(chosen_list):
        cov = np.zeros(total, dtype=np.uint8)
        for w in chosen_list:
            cov[balls[w]] = 1
        return cov
    
    def _try_remove(chosen_list):
        improved = True
        while improved:
            if time.time() > DEADLINE - 5.0:
                return chosen_list
            improved = False
            cov = np.zeros(total, dtype=np.uint8)
            for w in chosen_list:
                cov[balls[w]] = 1
            for i in range(len(chosen_list) - 1, -1, -1):
                # Temporarily remove chosen_list[i]
                # Recompute coverage without it: expensive. Instead, track counts.
                pass
        
        # Better: use count array
        cov_count = np.zeros(total, dtype=np.int32)
        for w in chosen_list:
            cov_count[balls[w]] += 1
        
        improved = True
        while improved:
            if time.time() > DEADLINE - 5.0:
                return chosen_list
            improved = False
            for i in range(len(chosen_list) - 1, -1, -1):
                for c in balls[chosen_list[i]]:
                    cov_count[c] -= 1
                if cov_count.min() >= 1:
                    chosen_list.pop(i)
                    improved = True
                    break
                for c in balls[chosen_list[i]]:
                    cov_count[c] += 1
        return chosen_list
    
    chosen_list = _try_remove(chosen_list)
    
    def _try_swap(chosen_list):
        # Try to replace one codeword with another to enable removal of another
        improved = True
        max_iters = 0
        while improved and time.time() < DEADLINE - 5.0:
            improved = False
            max_iters += 1
            if max_iters > 50000:
                break
            
            cov_count = np.zeros(total, dtype=np.int32)
            for w in chosen_list:
                cov_count[balls[w]] += 1
            
            for i in range(len(chosen_list)):
                if time.time() > DEADLINE - 5.0:
                    return chosen_list
                w_old = chosen_list[i]
                for c in balls[w_old]:
                    cov_count[c] -= 1
                
                uncov = np.where(cov_count == 0)[0]
                if len(uncov) == 0:
                    # Redundant: remove
                    chosen_list.pop(i)
                    improved = True
                    break
                if len(uncov) == 1:
                    target = int(uncov[0])
                    # Find a word covering target, not in chosen_list
                    # Precompute: for each word, which targets it covers (its ball)
                    # We need a word w_new such that target in balls[w_new] and w_new not in chosen_set
                    chosen_set_local = set(chosen_list)
                    found = False
                    for w_new in range(total):
                        if w_new in chosen_set_local:
                            continue
                        if target in ball_sets[w_new]:
                            chosen_list[i] = w_new
                            improved = True
                            found = True
                            break
                    if found:
                        break
                for c in balls[w_old]:
                    cov_count[c] += 1
        return chosen_list
    
    chosen_list = _try_swap(chosen_list)
    chosen_list = _try_remove(chosen_list)
    
    # Polish
    while time.time() < DEADLINE - 3.0:
        before = len(chosen_list)
        chosen_list = _try_remove(chosen_list)
        chosen_list = _try_swap(chosen_list)
        chosen_list = _try_remove(chosen_list)
        if len(chosen_list) == before:
            break
    
    # Build output
    out = []
    seen = set()
    for w in chosen_list:
        codeword = list(np.unravel_index(w, (q,) * n))
        t = tuple(codeword)
        if t not in seen:
            seen.add(t)
            out.append(codeword)
    return out
