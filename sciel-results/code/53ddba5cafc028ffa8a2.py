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

        # Precompute overlap matrix: overlap[i][j] = longest suffix of perms[i] that is prefix of perms[j]
        # This is O(N^2 * 6) = ~3M operations, fast
        overlap_mat = [[0] * N for _ in range(N)]
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

        # 2. Heuristic Optimization
        # Nearest Neighbor
        random.seed(42)
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
        
        # If NN is already better, update best
        if current_overlap > best_overlap:
            best_overlap = current_overlap
            # We don't build the word yet, just track overlap to save time
            # But we need the word at the end. We can build it if it's better.
            # Actually, let's just keep the path and build at the end if it's the best.
            # Or build now if it's better? Building is O(6000). Cheap.
            # Let's just track the best path.
            best_path = path[:]
        else:
            best_path = naive_path

        # 2-opt + Or-Opt improvement
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            
            # 2-Opt
            for i in range(len(path) - 2):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, len(path) - 1):
                    if j == len(path) - 1:
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
            
            # Or-Opt (Move segment of length 1 or 2)
            if not improved and time.time() < deadline - 15:
                for L in [1, 2]:
                    for i in range(len(path) - L):
                        if time.time() > deadline - 15:
                            break
                        # Segment to move: path[i : i+L]
                        # Remove it
                        # Neighbors in original: 
                        # left: path[i-1] if i>0 else None
                        # right: path[i+L] if i+L < N else None
                        # 
                        # Try inserting at every position j in the remaining path
                        # This is O(N^2) per L. 720^2 * 2 = 1M. OK.
                        
                        # To simplify, we iterate j.
                        # If we move i..i+L-1 to position j:
                        # Case 1: j < i (insert before j)
                        # Case 2: j > i+L-1 (insert after j, effectively before j+1)
                        
                        # We can just generate the new path and check delta?
                        # That's O(N) per move. Total O(N^3) = 300M. Too slow.
                        # We need O(1) delta calculation.
                        
                        # Let's stick to 2-opt for now, it's safer and faster to implement correctly.
                        # Or-Opt is complex.
                        pass

            # Check if current is best
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
