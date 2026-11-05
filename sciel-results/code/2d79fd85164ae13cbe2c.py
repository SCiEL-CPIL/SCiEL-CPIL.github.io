"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if n <= 6:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Build S_6 standard construction
        word = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new

        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)

        N = len(perms)

        # Precompute overlap matrix using numpy for speed
        import numpy as np
        # Convert perms to list of tuples for quick access
        # overlap[i][j] = longest suffix of perms[i] that is prefix of perms[j]
        # We can compute this efficiently
        overlap_mat = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_mat[i][j] = 6
                    continue
                pj = perms[j]
                ov = 0
                for k in range(5, -1, -1):
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat[i][j] = ov

        # Helper to calculate total overlap
        def calc_total_overlap(p):
            total = 0
            for i in range(len(p) - 1):
                total += overlap_mat[p[i]][p[i + 1]]
            return total

        # Helper to build word from path
        def build_word_from_path(p):
            w = []
            for idx in p:
                perm = perms[idx]
                block = list(perm) + [7] + list(perm)
                overlap = 0
                min_len = min(len(w), len(block))
                for o in range(min_len, -1, -1):
                    if w[len(w) - o:] == block[:o]:
                        overlap = o
                        break
                w += block[overlap:]
            return "".join(map(str, w))

        # 1. Generate Naive Word (standard order)
        naive_path = list(range(N))
        naive_word = build_word_from_path(naive_path)
        best_word = naive_word
        best_overlap = calc_total_overlap(naive_path)
        best_path = naive_path[:]

        # 2. Greedy construction (nearest neighbor) with multiple random starts
        random.seed(42)
        for start_seed in range(10):
            random.seed(start_seed)
            unvisited = list(range(N))
            random.shuffle(unvisited)
            path = [unvisited.pop()]

            while unvisited:
                last = path[-1]
                best_idx = -1
                best_ov = -1
                for idx in unvisited:
                    ov = overlap_mat[last][idx]
                    if ov > best_ov:
                        best_ov = ov
                        best_idx = idx
                path.append(best_idx)
                unvisited.remove(best_idx)

            current_overlap = calc_total_overlap(path)
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_path = path[:]

        # 3. Local improvement: 2-opt with limited segment length + Or-Opt (move 1)
        path = best_path[:]
        current_overlap = best_overlap
        time_limit = deadline - 15

        # 2-opt: try reversing segments of length up to max_seg
        max_seg = 50  # Limit segment length for speed
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(N - 1):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, min(i + max_seg, N)):
                    if j == N - 1:
                        old_ov = overlap_mat[path[i]][path[i + 1]]
                        new_ov = overlap_mat[path[i]][path[j]]
                        if new_ov > old_ov:
                            path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                    else:
                        old_ov1 = overlap_mat[path[i]][path[i + 1]]
                        old_ov2 = overlap_mat[path[j]][path[j + 1]]
                        new_ov1 = overlap_mat[path[i]][path[j]]
                        new_ov2 = overlap_mat[path[i + 1]][path[j + 1]]
                        if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                            path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                            current_overlap += (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
                            improved = True
                            break
                if improved:
                    break

            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_path = path[:]

        # Or-Opt: move single element
        if time.time() < time_limit:
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for i in range(N):
                    if time.time() > time_limit:
                        break
                    elem = path[i]
                    # Remove elem from position i
                    # Neighbors: left = path[i-1] if i > 0 else None
                    #           right = path[i+1] if i < N-1 else None
                    for j in range(N):
                        if j == i or j == i + 1:
                            continue
                        # Insert elem at position j (before current path[j])
                        # After removal, positions shift
                        # Left neighbor after removal: if i > 0, path[i-1], else None
                        # Right neighbor after removal: if i < N-1, path[i+1], else None
                        # After insertion at j: left of elem is path[j-1] if j > 0 else None
                        #                       right of elem is path[j] (after shift)
                        
                        # Compute delta
                        # Old connections involving i:
                        old_delta = 0
                        if i > 0 and i < N - 1:
                            old_delta = overlap_mat[path[i - 1]][path[i]] + overlap_mat[path[i]][path[i + 1]]
                        elif i > 0:
                            old_delta = overlap_mat[path[i - 1]][path[i]]
                        elif i < N - 1:
                            old_delta = overlap_mat[path[i]][path[i + 1]]
                        
                        # New connections:
                        # After removing i, the element at i+1 (if exists) shifts to i
                        # Insert at position j means:
                        # If j < i: insert before j, so after removal, position j is still j
                        #   left of elem = path[j-1] if j > 0 else None
                        #   right of elem = path[j]
                        # If j > i: insert before j, but after removal, position j-1 is original j
                        #   left of elem = path[j-1] (which is original j)
                        #   right of elem = path[j] (which is original j+1)
                        
                        new_delta = 0
                        if j < i:
                            # Insert at position j (before current j)
                            if j > 0:
                                new_delta += overlap_mat[path[j - 1]][elem]
                            if j < N - 1:
                                new_delta += overlap_mat[elem][path[j]]
                            # Also need to fix the connection that was broken
                            # After removal, path[i-1] connects to path[i+1] (if both exist)
                            # But we already accounted for that in old_delta
                        else:
                            # j > i, insert at position j (before current j, which is now at j-1 after removal)
                            # The element goes between path[j-1] and path[j]
                            if j > 0:
                                new_delta += overlap_mat[path[j - 1]][elem]
                            if j < N:
                                new_delta += overlap_mat[elem][path[j]]
                        
                        # The net change is new_delta - old_delta
                        # But we need to be more careful about what connections are broken/formed
                        # Let me just do the move and check
                        # For speed, only do this for promising moves
                        if new_delta > old_delta:
                            # Perform the move
                            path.pop(i)
                            # Adjust j if j > i
                            if j > i:
                                j -= 1
                            path.insert(j, elem)
                            current_overlap += new_delta - old_delta
                            improved = True
                            break
                    if improved:
                        break

                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = path[:]

        # Build final word from best_path
        final_word = build_word_from_path(best_path)
        
        # Compare with naive
        if len(final_word) < len(naive_word):
            return final_word
        else:
            return naive_word

    # Fallback for n > 7
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
