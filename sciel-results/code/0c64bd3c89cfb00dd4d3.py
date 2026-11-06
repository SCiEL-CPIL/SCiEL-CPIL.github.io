import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 150s kill

    N = 1000

    def sum_counts_from_mask(mask):
        # mask: bool array length N+1 (index 0 unused)
        # returns counts of a+b with a<=b, for sums 2..2N
        # convolve gives counts of ordered pairs; we want unordered with a<=b.
        # Use convolution of mask with mask: c[s] = sum_{i+j=s} mask[i]*mask[j]
        # ordered. For a<=b: count = (c[s] + (s even ? mask[s//2] : 0)) // 2
        m = mask.astype(np.int32)
        c = np.convolve(m, m, mode='full')  # length 2N+1, index s
        s = np.arange(2 * N + 1)
        diag = np.zeros(2 * N + 1, dtype=np.int32)
        even = (s % 2 == 0)
        half = s // 2
        # mask index must be within [0,N]
        valid = even & (half <= N)
        diag[valid] = m[half[valid]]
        counts = (c + diag) // 2
        return counts

    def is_valid(mask):
        counts = sum_counts_from_mask(mask)
        return np.max(counts) <= 2

    best_size = 0
    best_mask = None

    def greedy(order):
        mask = np.zeros(N + 1, dtype=bool)
        for x in order:
            if x > N or x < 1:
                continue
            # test adding x
            mask[x] = True
            if not is_valid(mask):
                mask[x] = False
            # else keep
        return mask

    def local_search(mask, time_limit):
        # try swaps: remove a, add b
        m = mask.copy()
        size = int(np.sum(m))
        counts = sum_counts_from_mask(m)
        present = np.where(m)[0]
        present = present[present >= 1]
        rng = random.Random()
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_limit:
            improved = False
            # try a few random swap moves
            for _ in range(200):
                if len(present) == 0:
                    break
                a = present[rng.randrange(len(present))]
                # pick candidate b not present
                b = rng.randint(1, N)
                if m[b]:
                    continue
                # try swap
                m[a] = False
                m[b] = True
                if is_valid(m):
                    new_size = int(np.sum(m))
                    if new_size > size:
                        size = new_size
                        present = np.where(m)[0]
                        present = present[present >= 1]
                        counts = sum_counts_from_mask(m)
                        improved = True
                    else:
                        m[a] = True
                        m[b] = False
                else:
                    m[a] = True
                    m[b] = False
        return m

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Build an ordering: mix of random and a heuristic (prefer small numbers early sometimes)
        r = random.random()
        if r < 0.34:
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif r < 0.67:
            # bias toward smaller numbers
            order = list(range(1, N + 1))
            weights = [1.0 / (x + 1) for x in order]
            random.shuffle(order)
        else:
            # bias toward larger numbers
            order = list(range(1, N + 1))
            random.shuffle(order)

        mask = greedy(order)
        size = int(np.sum(mask))

        # local search with a small time slice
        if time.time() < deadline - 10:
            slice_time = min(5.0, max(0.5, (deadline - 10) - (time.time() - start_time)))
            mask = local_search(mask, slice_time)
            size = int(np.sum(mask))

        if best_mask is None or size > best_size:
            best_size = size
            best_mask = mask.copy()

    if best_mask is None:
        # fallback: empty or a trivial valid set
        return []
    result = [int(x) for x in np.where(best_mask)[0] if x >= 1]
    return result
