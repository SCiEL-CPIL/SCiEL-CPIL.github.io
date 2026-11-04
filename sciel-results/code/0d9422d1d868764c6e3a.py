import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 145.0

    N = 100
    target_sum = N / 2.0
    num_ones = int(target_sum)  # 50 ones

    def score_binary(h):
        """Compute max overlap of h with shifted 1-h, excluding lag 0."""
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude lag 0 (index N-1)
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return np.max(c_nonzero)

    def try_binary_rounding(h):
        """Round continuous h to binary, fixing count to exactly num_ones."""
        h_sorted_indices = np.argsort(-h)
        h_rounded = np.zeros(N)
        for i in range(num_ones):
            h_rounded[h_sorted_indices[i]] = 1
        return h_rounded

    def local_search_binary(h_bin, max_iter=10000, temperature=0.0):
        """
        Local search on binary sequences with swap moves.
        h_bin must be a binary array with exactly num_ones 1s.
        """
        h_curr = h_bin.copy()
        curr_score = score_binary(h_curr)
        best_h = h_curr.copy()
        best_score = curr_score

        no_improve = 0
        rng = random.Random(12345)

        for iteration in range(max_iter):
            if time.time() - start_time > time_limit - 2:
                break

            # Pick random 1 and random 0 to swap
            ones = np.where(h_curr == 1)[0]
            zeros = np.where(h_curr == 0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break

            i1 = rng.choice(ones)
            i0 = rng.choice(zeros)

            # Perform swap
            h_curr[i1] = 0
            h_curr[i0] = 1

            new_score = score_binary(h_curr)

            # Accept if better or with some probability (simulated annealing-like)
            if new_score < curr_score:
                curr_score = new_score
                no_improve = 0
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_curr.copy()
            else:
                # Undo swap
                h_curr[i1] = 1
                h_curr[i0] = 0
                no_improve += 1

            # If stuck, do a random perturbation
            if no_improve > 50:
                # Randomly flip 3-5 pairs
                for _ in range(3):
                    ones = np.where(h_curr == 1)[0]
                    zeros = np.where(h_curr == 0)[0]
                    if len(ones) > 0 and len(zeros) > 0:
                        i1 = rng.choice(ones)
                        i0 = rng.choice(zeros)
                        h_curr[i1] = 0
                        h_curr[i0] = 1
                curr_score = score_binary(h_curr)
                no_improve = 0

        return best_h, best_score

    def generate_random_binary():
        """Generate a random binary sequence with exactly num_ones 1s."""
        positions = list(range(N))
        random.shuffle(positions)
        h = np.zeros(N)
        for i in range(num_ones):
            h[positions[i]] = 1
        return h

    def generate_structured_binary(pattern_type):
        """Generate structured binary sequences."""
        h = np.zeros(N)
        if pattern_type == 0:
            # Alternating
            h[::2] = 1
            # Adjust count
            count = np.sum(h)
            if count > num_ones:
                # Remove excess ones
                ones = np.where(h == 1)[0]
                random.shuffle(ones.tolist())
                for i in range(int(count - num_ones)):
                    h[ones[i]] = 0
            elif count < num_ones:
                # Add more ones
                zeros = np.where(h == 0)[0]
                random.shuffle(zeros.tolist())
                for i in range(int(num_ones - count)):
                    h[zeros[i]] = 1
        elif pattern_type == 1:
            # Blocks of 2
            h[::4] = 1
            h[1::4] = 1
            count = np.sum(h)
            if count > num_ones:
                ones = np.where(h == 1)[0]
                random.shuffle(ones.tolist())
                for i in range(int(count - num_ones)):
                    h[ones[i]] = 0
            elif count < num_ones:
                zeros = np.where(h == 0)[0]
                random.shuffle(zeros.tolist())
                for i in range(int(num_ones - count)):
                    h[zeros[i]] = 1
        elif pattern_type == 2:
            # Fibonacci-like spacing
            positions = set()
            pos = 0
            while len(positions) < num_ones:
                positions.add(pos % N)
                pos += 2 if len(positions) % 2 == 0 else 3
            for p in positions:
                h[p] = 1
        elif pattern_type == 3:
            # Quadratic residues mod N
            for i in range(N):
                if (i * i) % N < N // 2:
                    h[i] = 1
            count = np.sum(h)
            if count > num_ones:
                ones = np.where(h == 1)[0]
                random.shuffle(ones.tolist())
                for i in range(int(count - num_ones)):
                    h[ones[i]] = 0
            elif count < num_ones:
                zeros = np.where(h == 0)[0]
                random.shuffle(zeros.tolist())
                for i in range(int(num_ones - count)):
                    h[zeros[i]] = 1
        elif pattern_type == 4:
            # Spread out evenly with jitter
            base_positions = np.linspace(0, N, num_ones + 1, endpoint=False).astype(int)
            for p in base_positions:
                h[p] = 1
            # Add jitter
            for i in range(num_ones):
                if i > 0:
                    shift = random.choice([-1, 1])
                    new_pos = (base_positions[i] + shift) % N
                    if h[new_pos] == 0:
                        h[base_positions[i]] = 0
                        h[new_pos] = 1
            # Ensure exact count
            count = np.sum(h)
            if count > num_ones:
                ones = np.where(h == 1)[0]
                random.shuffle(ones.tolist())
                for i in range(int(count - num_ones)):
                    h[ones[i]] = 0
            elif count < num_ones:
                zeros = np.where(h == 0)[0]
                random.shuffle(zeros.tolist())
                for i in range(int(num_ones - count)):
                    h[zeros[i]] = 1
        return h

    best_h = None
    best_score = np.inf

    # Try structured initializations
    for pattern_type in range(5):
        if time.time() - start_time > time_limit - 5:
            break
        h = generate_structured_binary(pattern_type)
        if np.sum(h) == num_ones:
            s = score_binary(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()

    # Random initializations
    for trial in range(20):
        if time.time() - start_time > time_limit - 5:
            break
        h = generate_random_binary()
        s = score_binary(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Run local search from multiple starting points
    start_points = []
    if best_h is not None:
        start_points.append(best_h.copy())

    for pattern_type in range(5):
        if time.time() - start_time > time_limit - 5:
            break
        h = generate_structured_binary(pattern_type)
        if np.sum(h) == num_ones:
            start_points.append(h)

    for trial in range(10):
        if time.time() - start_time > time_limit - 5:
            break
        h = generate_random_binary()
        start_points.append(h)

    for sp in start_points:
        if time.time() - start_time > time_limit - 3:
            break
        h_ls, score_ls = local_search_binary(sp, max_iter=10000)
        if score_ls < best_score:
            best_score = score_ls
            best_h = h_ls.copy()

    # More extensive local search from the best solution found
    if time.time() - start_time < time_limit - 5 and best_h is not None:
        h_ls, score_ls = local_search_binary(best_h, max_iter=20000)
        if score_ls < best_score:
            best_score = score_ls
            best_h = h_ls.copy()

    # Final check
    if best_h is None:
        best_h = generate_random_binary()

    # Ensure exact count
    if np.sum(best_h) != num_ones:
        best_h = try_binary_rounding(best_h)

    return best_h
