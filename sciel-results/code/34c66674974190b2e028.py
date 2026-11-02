import time
import numpy as np
import random
from scipy.cluster.vq import kmeans2

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 118.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as boolean masks
    # cov[c, w] = True if c covers w
    cov = np.zeros((total, total), dtype=np.bool_)

    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q

        cov[c, c] = True

        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                cov[c, w] = True

    # Greedy construction
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total

        while uncovered > 0:
            # Compute gains vectorized
            gains = cov @ (1 - covered)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_c = random.choice(candidates)
            else:
                best_c = candidates[0]

            code.append(best_c)
            covered[cov[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code
        
    def spectral_init(target_k):
        """
        Initialize code using spectral clustering on the Hamming graph.
        Uses Fourier basis (eigenvectors) of the Hamming graph.
        """
        # Generate all words
        words = np.zeros((total, n), dtype=int)
        for c in range(total):
            temp = c
            for i in range(n):
                words[c, i] = temp % q
                temp //= q
        
        # Compute embedding
        # Weight 1: cos/sin of 2*pi*x_i/3 for each i
        # Weight 2: cos/sin of 2*pi*(x_i+x_j)/3, 2*pi*(x_i+2*x_j)/3 etc?
        # Actually, the characters are $u \cdot x$.
        # For q=3, $u \in \{0,1,2\}^n$.
        # We need weight 1 and 2 characters.
        
        # Weight 1: 7 positions, 2 non-zero values (1, 2).
        # $\exp(2\pi i \cdot u \cdot x / 3)$.
        # Real part: $\cos(2\pi u \cdot x / 3)$
        # Imag part: $\sin(2\pi u \cdot x / 3)$
        
        # Note: $\cos(4\pi x/3) = \cos(2\pi x/3)$, $\sin(4\pi x/3) = -\sin(2\pi x/3)$.
        # So for each position i, we just need $\cos(2\pi x_i/3)$ and $\sin(2\pi x_i/3)$.
        # That's 14 features.
        
        # Weight 2: Pairs (i, j) with $i < j$.
        # Non-zero components can be (1,1), (1,2), (2,1), (2,2).
        # $u \cdot x = u_i x_i + u_j x_j$.
        # Features: $\cos(2\pi (u_i x_i + u_j x_j)/3)$ and $\sin(...)$.
        # There are 21 pairs. 4 combinations of (u_i, u_j).
        # Total 21 * 4 * 2 = 168 features.
        
        # Total 182 features.
        
        features = np.zeros((total, 182), dtype=float)
        
        # Weight 1
        col = 0
        for i in range(n):
            angles = (2 * np.pi / 3) * words[:, i]
            features[:, col] = np.cos(angles)
            features[:, col + 1] = np.sin(angles)
            col += 2
            
        # Weight 2
        for i in range(n):
            for j in range(i + 1, n):
                for u_i in [1, 2]:
                    for u_j in [1, 2]:
                        # u = (.., u_i, .., u_j, ..)
                        # dot = u_i * x_i + u_j * x_j
                        angles = (2 * np.pi / 3) * (u_i * words[:, i] + u_j * words[:, j])
                        features[:, col] = np.cos(angles)
                        features[:, col + 1] = np.sin(angles)
                        col += 2
                        
        # Run K-Means
        # scipy.cluster.vq.kmeans2(data, k, minit='points')
        # minit='points' picks random points as initial centroids.
        # We might want to run it a few times or use a good init.
        
        try:
            centroids, label = kmeans2(features, target_k, minit='random', iter=10, seed=42)
        except Exception:
            return greedy_build(seed=42)
            
        # For each cluster, pick the word closest to the centroid in feature space
        # Or just pick the first word in the cluster?
        # Better: pick the word in the cluster that minimizes distance to centroid.
        
        code = []
        for k in range(target_k):
            cluster_idx = np.where(label == k)[0]
            if len(cluster_idx) == 0:
                continue
            # Calculate distance to centroid
            dists = np.sum((features[cluster_idx] - centroids[k]) ** 2, axis=1)
            best_local = cluster_idx[np.argmin(dists)]
            code.append(best_local)
            
        return code

    # Local search with 1-for-1 swaps and 2-for-1 improvements
    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[cov[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    if np.min(cnt[cov[c]]) >= 2:
                        code_set.remove(c)
                        cnt[cov[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        try_delete()

        no_success = 0
        max_streak = 1000

        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            # 1. Try deletion
            deleted = False
            for c in list(code_set):
                if np.min(cnt[cov[c]]) >= 2:
                    code_set.remove(c)
                    cnt[cov[c]] -= 1
                    deleted = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break

            if deleted:
                continue

            # 2. Try 1-for-1 swap
            c_remove = random.choice(list(code_set))
            critical_mask = (cnt == 1) & cov[c_remove]

            if not np.any(critical_mask):
                code_set.remove(c_remove)
                cnt[cov[c_remove]] -= 1
                continue

            critical_words = np.where(critical_mask)[0]

            # Find candidates that cover all critical words
            candidates_mask = np.all(cov[:, critical_words], axis=1)
            in_set_mask = np.zeros(total, dtype=bool)
            for c in code_set:
                in_set_mask[c] = True
            candidates_mask &= ~in_set_mask

            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) > 0:
                c_add = random.choice(candidate_indices)
                code_set.remove(c_remove)
                code_set.add(c_add)
                cnt[cov[c_remove]] -= 1
                cnt[cov[c_add]] += 1
                no_success = 0
                try_delete()
            else:
                no_success += 1

            # 3. Try 2-for-1 improvement (remove 1, add 2)
            if no_success > 500 and time.time() - t_start < time_budget - 0.5:
                # Try to remove one codeword and add two
                c_remove = random.choice(list(code_set))
                critical_mask = (cnt == 1) & cov[c_remove]
                critical_words = np.where(critical_mask)[0]

                if len(critical_words) > 0:
                    # Find two codewords that together cover all critical words
                    # Sample random pairs
                    for _ in range(50):
                        c1 = random.randint(0, total - 1)
                        c2 = random.randint(0, total - 1)
                        if c1 in code_set or c2 in code_set or c1 == c2:
                            continue
                        if c1 == c_remove or c2 == c_remove:
                            continue

                        # Check if c1 and c2 together cover all critical words
                        if np.all(cov[c1, critical_words] | cov[c2, critical_words]):
                            code_set.remove(c_remove)
                            code_set.add(c1)
                            code_set.add(c2)
                            cnt[cov[c_remove]] -= 1
                            cnt[cov[c1]] += 1
                            cnt[cov[c2]] += 1
                            no_success = 0
                            try_delete()
                            break
                    else:
                        no_success += 1
                else:
                    no_success += 1

            if no_success > max_streak:
                # Perturb: add a random codeword
                rand_c = random.randint(0, total - 1)
                if rand_c not in code_set:
                    code_set.add(rand_c)
                    cnt[cov[rand_c]] += 1
                    try_delete()
                no_success = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=15.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Try Spectral Init
    if time.time() < deadline - 20.0:
        spec_code = spectral_init(186)
        if len(spec_code) > 0:
            best_code_spec, best_size_spec = local_search(spec_code, time_budget=min(15.0, deadline - 5.0 - time.time()))
            if best_size_spec < best_size_overall:
                best_size_overall = best_size_spec
                best_code_overall = best_code_spec

    # Multi-restart
    seed = 0
    while time.time() < deadline - 10.0 and best_size_overall > 170:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 10.0 - time.time()
        tb = min(10.0, max(2.0, time_left / 8.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Convert to list of lists
    result = []
    for idx in best_code_overall:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
