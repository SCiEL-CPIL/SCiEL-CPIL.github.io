"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if n == 1:
        return "1"
    
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
                window = tuple(word[i:i + 6])
                if len(set(window)) == 6 and window not in seen:
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

        # Helper to calculate total overlap for a path
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += overlap_mat[path[i]][path[i + 1]]
            return total

        # Helper to build word from a path
        def build_word_from_path(path):
            w = []
            for idx in path:
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

        # Start with the standard order
        best_path = list(range(N))
        best_overlap = calc_total_overlap(best_path)

        # Multi-phase optimization with multiple restarts
        random.seed(42)
        
        for restart in range(10):
            if time.time() > deadline - 20:
                break
            
            if restart == 0:
                path = list(range(N))
            else:
                # Nearest neighbor initialization with random start
                unvisited = list(range(N))
                random.shuffle(unvisited)
                start = unvisited.pop()
                path = [start]
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
            
            # 2-Opt improvement
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
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
            
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_path = path[:]
            
            # Or-Opt: move segment of length 1 or 2
            if time.time() < deadline - 10:
                improved = True
                while improved and time.time() < deadline - 10:
                    improved = False
                    for L in [1, 2]:
                        if improved:
                            break
                        for i in range(len(path) - L):
                            if time.time() > deadline - 10:
                                break
                            # Segment to move: path[i : i+L]
                            seg = path[i:i+L]
                            # Remove segment
                            path_no_seg = path[:i] + path[i+L:]
                            # Try inserting at each position
                            for j in range(len(path_no_seg) - L + 1):
                                if j == i or j == i + L:
                                    continue
                                # Calculate change in overlap
                                old_ov = 0
                                new_ov = 0
                                
                                # Old: neighbors around removed segment
                                if i > 0:
                                    old_ov += overlap_mat[path[i-1]][seg[0]]
                                if i + L < N:
                                    old_ov += overlap_mat[seg[-1]][path[i+L]]
                                if i > 0 and i + L < N:
                                    old_ov += overlap_mat[path[i-1]][path[i+L]]
                                
                                # New: neighbors around inserted segment
                                if j > 0:
                                    new_ov += overlap_mat[path_no_seg[j-1]][seg[0]]
                                if j + L < len(path_no_seg):
                                    new_ov += overlap_mat[seg[-1]][path_no_seg[j+L]]
                                if j > 0 and j + L < len(path_no_seg):
                                    new_ov += overlap_mat[path_no_seg[j-1]][path_no_seg[j+L]]
                                
                                if new_ov > old_ov:
                                    new_path = path_no_seg[:j] + seg + path_no_seg[j:]
                                    path = new_path
                                    current_overlap += new_ov - old_ov
                                    improved = True
                                    break
                        if improved:
                            break
            
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_path = path[:]
            
            # Adjacent swap refinement
            if time.time() < deadline - 5:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(len(path) - 1):
                        if time.time() > deadline - 5:
                            break
                        a = path[i-1] if i > 0 else -1
                        b = path[i]
                        c = path[i+1]
                        d = path[i+2] if i+2 < N else -1
                        
                        old_contrib = 0
                        if a != -1:
                            old_contrib += overlap_mat[a][b]
                        old_contrib += overlap_mat[b][c]
                        if d != -1:
                            old_contrib += overlap_mat[c][d]
                        
                        new_contrib = 0
                        if a != -1:
                            new_contrib += overlap_mat[a][c]
                        new_contrib += overlap_mat[c][b]
                        if d != -1:
                            new_contrib += overlap_mat[b][d]
                        
                        if new_contrib > old_contrib:
                            path[i], path[i+1] = path[i+1], path[i]
                            current_overlap += new_contrib - old_contrib
                            improved = True
            
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_path = path[:]

        # Build final word from best path
        return build_word_from_path(best_path)

    # Fallback for n > 7 (shouldn't happen for this task)
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
