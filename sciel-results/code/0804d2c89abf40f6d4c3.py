import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    # Build standard superpermutation up to n-1
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new

    if n <= 2:
        return "".join(map(str, word))

    # Extract all (n-1)! permutations of 1..(n-1) in order of appearance
    m = n - 1
    perms, seen = [], set()
    for i in range(len(word) - m + 1):
        window = tuple(word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)

    N = len(perms)

    # Precompute overlap matrix: overlap[i][j] = longest suffix of perms[i] matching prefix of perms[j]
    ov_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            if i == j:
                ov_mat[i][j] = m
                continue
            pj = perms[j]
            ov = 0
            for k in range(m - 1, -1, -1):
                if pi[-k:] == pj[:k]:
                    ov = k
                    break
            ov_mat[i][j] = ov

    # Build word from a given path (list of permutation indices)
    def build_word(path):
        w = []
        for idx in path:
            perm = perms[idx]
            block = list(perm) + [n] + list(perm)
            overlap = 0
            min_len = min(len(w), len(block))
            for o in range(min_len, -1, -1):
                if w[len(w) - o:] == block[:o]:
                    overlap = o
                    break
            w += block[overlap:]
        return w

    # Compute total overlap for a path
    def total_overlap(path):
        s = 0
        for i in range(len(path) - 1):
            s += ov_mat[path[i]][path[i + 1]]
        return s

    # Start with standard order
    best_path = list(range(N))
    best_ov = total_overlap(best_path)

    # Greedy nearest neighbor to get a better starting path
    random.seed(42)
    unvisited = list(range(N))
    random.shuffle(unvisited)
    nn_path = [unvisited.pop(0)]
    while unvisited:
        last = nn_path[-1]
        best_idx = -1
        best_ov_val = -1
        for idx in unvisited:
            v = ov_mat[last][idx]
            if v > best_ov_val:
                best_ov_val = v
                best_idx = idx
        nn_path.append(best_idx)
        unvisited.remove(best_idx)

    nn_ov = total_overlap(nn_path)
    if nn_ov > best_ov:
        best_path = nn_path
        best_ov = nn_ov

    # 2-opt local search with O(1) delta using prefix sums
    # For a path p, prefix sum: PS[i] = sum(ov_mat[p[k]][p[k+1]] for k in 0..i-1), PS[0]=0
    # Reversal of segment [i, j] (inclusive):
    #   Old boundary edges: (p[i-1], p[i]) and (p[j], p[j+1]) if they exist
    #   New boundary edges: (p[i-1], p[j]) and (p[i], p[j+1]) if they exist
    #   Internal: old sum = PS[j] - PS[i], new sum = sum of ov_mat[p[k+1]][p[k]] for k in i..j-1
    #   We need prefix sums of the reverse direction too, or compute on the fly.
    #   Actually, let's precompute both forward and reverse prefix sums.

    def do_2opt(path, current_ov, time_limit):
        N_local = len(path)
        # Precompute forward prefix sums
        fwd_ps = [0] * (N_local)  # fwd_ps[i] = sum of ov_mat[path[k]][path[k+1]] for k=0..i-1
        for i in range(1, N_local):
            fwd_ps[i] = fwd_ps[i-1] + ov_mat[path[i-1]][path[i]]

        # Precompute reverse prefix sums: rev_ps[i] = sum of ov_mat[path[k]][path[k-1]] for k=i..N-2
        # i.e., reverse direction overlaps
        rev_ps = [0] * (N_local)
        for i in range(N_local - 2, -1, -1):
            rev_ps[i] = rev_ps[i+1] + ov_mat[path[i+1]][path[i]]

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(N_local - 1):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, N_local):
                    # Compute delta for reversing path[i..j]
                    old_score = 0
                    new_score = 0

                    # Left boundary
                    if i > 0:
                        old_score += ov_mat[path[i-1]][path[i]]
                        new_score += ov_mat[path[i-1]][path[j]]
                    # Right boundary
                    if j < N_local - 1:
                        old_score += ov_mat[path[j]][path[j+1]]
                        new_score += ov_mat[path[i]][path[j+1]]

                    # Internal: old = sum ov_mat[path[k]][path[k+1]] for k=i..j-1 = fwd_ps[j] - fwd_ps[i]
                    #           new = sum ov_mat[path[k]][path[k-1]] for k=i+1..j = rev_ps[i] - rev_ps[j]
                    old_internal = fwd_ps[j] - fwd_ps[i]
                    new_internal = rev_ps[i] - rev_ps[j]

                    delta = (new_score + new_internal) - (old_score + old_internal)
                    if delta > 0:
                        # Perform reversal
                        path[i:j+1] = reversed(path[i:j+1])
                        current_ov += delta
                        improved = True
                        # Update prefix sums after reversal
                        # Only need to update from position i onwards
                        for k in range(i, N_local):
                            if k == 0:
                                fwd_ps[k] = 0
                            else:
                                fwd_ps[k] = fwd_ps[k-1] + ov_mat[path[k-1]][path[k]]
                        for k in range(N_local - 1, -1, -1):
                            if k == N_local - 1:
                                rev_ps[k] = 0
                            else:
                                rev_ps[k] = rev_ps[k+1] + ov_mat[path[k+1]][path[k]]
                        break  # restart inner loop
                if improved:
                    continue  # restart outer loop

        return path, current_ov

    # Run 2-opt
    time_limit_2opt = deadline - 20
    best_path, best_ov = do_2opt(best_path[:], best_ov, time_limit_2opt)

    # Build final word
    word = build_word(best_path)

    # Trim from ends: remove leading/trailing characters that don't create unique permutations
    L = len(word)
    if L > n:
        # Count frequency of each n-window
        freq = {}
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            freq[p] = freq.get(p, 0) + 1

        # Trim from left
        while len(word) > n:
            if time.time() > deadline - 5:
                break
            first_perm = tuple(word[:n])
            if freq.get(first_perm, 0) > 1:
                freq[first_perm] -= 1
                word = word[1:]
            else:
                break

        # Trim from right
        while len(word) > n:
            if time.time() > deadline - 5:
                break
            last_perm = tuple(word[-n:])
            if freq.get(last_perm, 0) > 1:
                freq[last_perm] -= 1
                word = word[:-1]
            else:
                break

    return "".join(map(str, word))
