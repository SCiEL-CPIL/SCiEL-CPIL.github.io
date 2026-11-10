import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Construct a ternary covering code of length n with radius r.
    For q=3, n=8, r=1, we aim to find a code with fewer than 486 codewords.
    Uses a combination of structured construction, greedy seed, and local search.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback: greedy set cover
        N = q ** n
        powers = [q ** i for i in range(n)]
        neighbors = np.zeros((N, 1 + (q - 1) * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            temp = w
            for i in range(n):
                d = temp % q
                temp //= q
                for new_d in range(q):
                    if new_d != d:
                        w1 = w + (new_d - d) * powers[i]
                        neighbors[w, idx] = w1
                        idx += 1
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = candidates[np.random.randint(len(candidates))]
            code.append(w)
            uncovered[neighbors[w]] = 0
        return [[int(c) for c in np.base_repr(w, base=q).zfill(n)][::-1] for w in code]

    N = 3 ** 8  # 6561
    deadline = time.time() + 115.0

    # Precompute neighbors
    neighbors = np.zeros((N, 17), dtype=np.int32)
    powers = [3 ** i for i in range(n)]
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for i in range(n):
            d = temp % 3
            temp //= 3
            w1 = w + ((d + 1) % 3 - d) * powers[i]
            neighbors[w, idx] = w1
            idx += 1
            w2 = w + ((d + 2) % 3 - d) * powers[i]
            neighbors[w, idx] = w2
            idx += 1

    # Tetracode seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code_ints = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            w_int = 0
            for i in range(n):
                w_int += word[i] * powers[i]
            seed_code_ints.append(w_int)
    best_code = seed_code_ints[:]
    best_size = len(best_code)

    rng = np.random.default_rng(42)

    # Greedy restarts
    while time.time() < deadline - 30.0:
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.append(w)
            uncovered[neighbors[w]] = 0
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            if best_size <= 403:
                break

    # Local search: try to reduce code size by swapping
    if time.time() < deadline - 10.0:
        # Try to find a code of size best_size - 1, best_size - 2, ...
        for target_size in range(best_size - 1, 402, -1):
            if time.time() > deadline - 15.0:
                break
            found = False
            for attempt in range(20):
                if time.time() > deadline - 15.0:
                    break
                # Start from best_code, remove one, then use local search
                init_code = best_code[:]
                if len(init_code) > target_size:
                    # Randomly remove extra codewords
                    remove_idx = rng.choice(len(init_code), len(init_code) - target_size, replace=False)
                    init_code = [init_code[i] for i in range(len(init_code)) if i not in remove_idx]
                
                # Local search: try to fix by swapping
                uncovered = np.ones(N, dtype=np.uint8)
                for w in init_code:
                    uncovered[neighbors[w]] = 0
                unc_count = uncovered.sum()
                
                if unc_count == 0:
                    best_size = target_size
                    best_code = init_code[:]
                    found = True
                    break
                
                # Local search iterations
                for it in range(500):
                    if time.time() > deadline - 15.0:
                        break
                    # Find uncovered words
                    unc_words = np.where(uncovered > 0)[0]
                    if len(unc_words) == 0:
                        best_size = target_size
                        best_code = init_code[:]
                        found = True
                        break
                    
                    # Try to add a codeword that covers the most uncovered
                    gains = uncovered[neighbors].sum(axis=1)
                    # Only consider adding if we can remove one
                    # First, try to find a swap: remove one, add one
                    # For simplicity, try adding the best
                    best_w = np.argmax(gains)
                    if gains[best_w] == 0:
                        break
                    
                    # Try swapping: for each codeword in init_code, remove it and see if we can cover more
                    best_swap = None
                    best_swap_gain = 0
                    for i, cw in enumerate(init_code):
                        # Remove cw
                        # Count how many of its neighbors are only covered by cw
                        # This is expensive, so approximate: count uncovered neighbors of best_w
                        # that are not neighbors of cw
                        # Simplified: just try the swap
                        new_uncovered = uncovered.copy()
                        # Re-add uncovered from removing cw
                        # Actually, let's just do a simple local search:
                        # Add best_w, then remove the one that covers the fewest unique
                        pass
                    
                    # Simpler local search: just add if it helps, then prune
                    if gains[best_w] > 0:
                        init_code.append(int(best_w))
                        uncovered[neighbors[best_w]] = 0
                        unc_count = uncovered.sum()
                        # Now prune: remove codewords that don't cover any unique uncovered
                        # Actually, remove codewords whose removal doesn't increase uncovered
                        if len(init_code) > target_size:
                            # Remove the one that covers the fewest
                            for i in range(len(init_code)):
                                # Count unique coverage
                                cw = init_code[i]
                                others = [j for j in range(len(init_code)) if j != i]
                                # Count neighbors of cw not in neighbors of others
                                # This is O(n^2 * 17)
                                if time.time() > deadline - 15.0:
                                    break
                    else:
                        break
                
                if found:
                    break
            
            if found:
                continue

    # Convert to final format
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % 3)
            temp //= 3
        final_code.append(word)
    return final_code
