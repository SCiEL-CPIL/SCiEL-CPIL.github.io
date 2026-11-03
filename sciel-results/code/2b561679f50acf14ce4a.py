import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill

    N = q ** n

    # Helper functions
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        base = 1
        for i in range(n):
            idx += w[i] * base
            base *= q
        return idx

    # Precompute neighbors for each word index (radius=1 case is most common)
    if radius == 1:
        ball_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, ball_size), dtype=np.int32)
        for idx in range(N):
            w = idx_to_word(idx)
            neighbors[idx, 0] = idx
            pos = 1
            for p in range(n):
                orig = w[p]
                for val in range(q):
                    if val == orig:
                        continue
                    w[p] = val
                    neighbors[idx, pos] = word_to_idx(w)
                    pos += 1
                w[p] = orig
    elif radius == 0:
        ball_size = 1
        neighbors = np.zeros((N, 1), dtype=np.int32)
        for idx in range(N):
            neighbors[idx, 0] = idx
    else:
        # Fallback: brute force
        ball_size = N
        neighbors = np.zeros((N, N), dtype=np.int32)
        for idx in range(N):
            w1 = idx_to_word(idx)
            pos = 0
            for jdx in range(N):
                w2 = idx_to_word(jdx)
                dist = sum(1 for a, b in zip(w1, w2) if a != b)
                if dist <= radius:
                    neighbors[idx, pos] = jdx
                    pos += 1

    def is_covered(code_indices):
        """Check if all words are covered by the given code."""
        covered = np.zeros(N, dtype=np.uint8)
        for c in code_indices:
            covered[neighbors[c, :ball_size]] = 1
        return covered.all()

    def remove_redundant(code_list, max_iters=200):
        """Iteratively remove codewords that don't reduce coverage."""
        code_arr = list(code_list)
        changed = True
        iters = 0
        while changed and iters < max_iters:
            if time.time() > deadline:
                break
            changed = False
            iters += 1
            for i in range(len(code_arr)):
                if time.time() > deadline:
                    break
                # Check if removing code_arr[i] still covers all
                covered = np.zeros(N, dtype=np.uint8)
                for j, c in enumerate(code_arr):
                    if j == i:
                        continue
                    covered[neighbors[c, :ball_size]] = 1
                if covered.all():
                    code_arr.pop(i)
                    changed = True
                    break
        return code_arr

    def greedy_init_fast():
        """Fast greedy set cover initialization using uncovered-word neighborhood sampling."""
        coverage = np.zeros(N, dtype=np.int8)
        uncovered_count = N
        code_indices = []

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            # Find an uncovered word
            uncovered_mask = coverage == 0
            if not uncovered_mask.any():
                break
            uncovered_indices = np.where(uncovered_mask)[0]
            # Sample a random uncovered word (or first one for determinism)
            target = random.choice(uncovered_indices.tolist()) if len(uncovered_indices) > 10 else uncovered_indices[0]

            # Find candidates that cover the target word
            # A codeword c covers target if target is in neighbors[c, :ball_size]
            # Equivalently, c is in neighbors[target, :ball_size] (by symmetry for radius 1)
            candidates = neighbors[target, :ball_size].tolist()
            
            best_c = -1
            max_new = -1
            
            for c in candidates:
                new_covers = np.sum(coverage[neighbors[c, :ball_size]] == 0)
                if new_covers > max_new:
                    max_new = new_covers
                    best_c = c

            if best_c == -1 or max_new == 0:
                break

            code_indices.append(best_c)
            prev_uncovered = coverage[neighbors[best_c, :ball_size]] == 0
            uncovered_count -= np.sum(prev_uncovered)
            coverage[neighbors[best_c, :ball_size]] += 1

        return code_indices

    def randomized_local_search(code_list, time_budget=20.0):
        """Randomized local search: swap codewords to improve coverage, then remove redundant."""
        code_arr = list(code_list)
        start = time.time()
        best_arr = list(code_arr)
        best_len = len(code_arr)
        
        # First pass: remove redundant
        code_arr = remove_redundant(code_arr, max_iters=100)
        best_arr = list(code_arr)
        best_len = len(code_arr)
        
        while time.time() - start < time_budget:
            if time.time() > deadline:
                break
            
            improved = False
            # Try random swaps
            for _ in range(50):
                if time.time() > deadline:
                    break
                if len(code_arr) < 2:
                    break
                    
                i = random.randrange(len(code_arr))
                # Pick a random candidate to swap with
                c = random.randrange(N)
                if c in code_arr:
                    continue
                    
                old_c = code_arr[i]
                code_arr[i] = c
                
                # Check coverage
                covered = np.zeros(N, dtype=np.uint8)
                for cc in code_arr:
                    covered[neighbors[cc, :ball_size]] = 1
                
                if covered.all():
                    # Try to remove redundant
                    new_arr = remove_redundant(code_arr, max_iters=50)
                    if len(new_arr) < best_len:
                        best_arr = new_arr
                        best_len = len(new_arr)
                        code_arr = new_arr
                        improved = True
                        break
                    else:
                        # Keep the swap if it doesn't hurt, revert if it does
                        code_arr[i] = old_c
                else:
                    # Revert
                    code_arr[i] = old_c
            
            if not improved:
                # Try more aggressive: remove a random codeword and try to fix
                if len(code_arr) > 10:
                    i = random.randrange(len(code_arr))
                    removed = code_arr.pop(i)
                    # Try to find a replacement that covers the lost words
                    # This is complex, so just revert for now
                    code_arr.insert(i, removed)
        
        return best_arr

    best_code = None
    best_size = N + 1

    # Multiple restarts with different strategies
    for seed in range(10):
        if time.time() > deadline - 5:
            break

        random.seed(seed)
        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        
        if remaining < 5:
            break
        
        # Alternate between greedy and structured init
        if seed % 2 == 0:
            code_indices = greedy_init_fast()
        else:
            # Structured init
            if n == 8 and q == 3:
                G = np.array([
                    [1, 0, 0, 0, 1, 1, 2, 1],
                    [0, 1, 0, 0, 1, 2, 1, 2],
                    [0, 0, 1, 0, 2, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 2]
                ], dtype=np.int32) % 3

                code_indices = []
                for msg in itertools.product(range(3), repeat=4):
                    msg_arr = np.array(msg, dtype=np.int32)
                    codeword = (msg_arr @ G) % 3
                    idx = word_to_idx(codeword.tolist())
                    code_indices.append(idx)
            else:
                tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
                code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
                code_indices = [word_to_idx(w) for w in code]

        # Randomized local search with time budget
        time_budget = min(15.0, remaining / 2.0)
        code_indices = randomized_local_search(code_indices, time_budget)

        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code = [idx_to_word(idx) for idx in code_indices]

    if best_code is None:
        # Final fallback
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    return best_code
