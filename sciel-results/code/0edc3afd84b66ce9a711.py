"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

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

        # Precompute overlap matrix
        # overlap[i][j] = longest suffix of perms[i] that is prefix of perms[j]
        overlap_mat = [[0] * N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_mat[i][j] = 6
                    continue
                pj = perms[j]
                ov = 0
                # Max possible overlap is 5 (since they are distinct, 6 is impossible)
                for k in range(5, 0, -1):
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat[i][j] = ov

        def calc_total_overlap(p):
            total = 0
            for i in range(len(p) - 1):
                total += overlap_mat[p[i]][p[i + 1]]
            return total

        # Simulated Annealing
        # Initial solution: Nearest Neighbor
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

        # Additional initial solution: try a few different random starts
        best_initial = path[:]
        best_initial_score = calc_total_overlap(best_initial)
        
        for seed in range(5):
            random.seed(42 + seed)
            unvisited = list(range(N))
            random.shuffle(unvisited)
            p = [unvisited.pop()]
            while unvisited:
                last = p[-1]
                best_idx = -1
                best_ov = -1
                for idx in unvisited:
                    ov = overlap_mat[last][idx]
                    if ov > best_ov:
                        best_ov = ov
                        best_idx = idx
                p.append(best_idx)
                unvisited.remove(best_idx)
            sc = calc_total_overlap(p)
            if sc > best_initial_score:
                best_initial_score = sc
                best_initial = p[:]
        
        path = best_initial

        current_score = calc_total_overlap(path)
        best_score = current_score
        best_path = path[:]

        # SA Parameters
        T = 3.0
        cooling = 0.9997
        min_T = 1e-8
        
        # Time management
        t_start = time.time()
        
        while time.time() < deadline - 30:
            if T < min_T:
                break
            
            # Choose move type: 60% Or-Opt 1, 30% Or-Opt 2, 10% Or-Opt 3
            move_type = random.random()
            
            if move_type < 0.6:
                # Or-Opt 1: Move single element k to position i
                k = random.randint(0, N - 1)
                i = random.randint(0, N - 2)
                
                if i == k or i == k - 1:
                    continue  # No-op
                
                # Remove element k, insert between i and i+1
                # We need to handle index shifts carefully
                # After removing k, elements after k shift left by 1
                # Insertion position i is in the ORIGINAL indexing
                
                if k == 0:
                    # Remove first element
                    if i == 0:
                        continue
                    old1 = overlap_mat[path[0]][path[1]]
                    old2 = overlap_mat[path[i]][path[i+1]]
                    new1 = overlap_mat[path[i]][path[0]]
                    new2 = overlap_mat[path[0]][path[i+1]]
                    delta = new1 + new2 - old1 - old2
                elif k == N - 1:
                    # Remove last element
                    if i == N - 2:
                        continue
                    old1 = overlap_mat[path[N-2]][path[N-1]]
                    old2 = overlap_mat[path[i]][path[i+1]]
                    new1 = overlap_mat[path[i]][path[N-1]]
                    new2 = overlap_mat[path[N-1]][path[i+1]]
                    delta = new1 + new2 - old1 - old2
                else:
                    # Remove internal element k
                    if i == k - 1 or i == k:
                        continue  # No-op or invalid
                    
                    # Edges removed: (k-1, k), (k, k+1), (i, i+1)
                    # Edges added: (k-1, k+1), (i, k), (k, i+1)
                    # But careful: if i > k, then after removing k, the indices shift
                    # Let's use the values directly
                    
                    A = path[k-1]
                    K = path[k]
                    B = path[k+1]
                    
                    if i > k:
                        # i is after k, so after removal, i shifts to i-1
                        # The insertion point is between original path[i-1] and path[i]
                        # Wait, let me reconsider.
                        # Original: ... A, K, B, ..., path[i-1], path[i], path[i+1], ...
                        # Remove K: ... A, B, ..., path[i-1], path[i], path[i+1], ...
                        # Insert K between path[i-1] and path[i]: ... A, B, ..., path[i-1], K, path[i], path[i+1], ...
                        # So the edge (path[i-1], path[i]) is split.
                        # Edges removed: (A,K), (K,B), (path[i-1], path[i])
                        # Edges added: (A,B), (path[i-1], K), (K, path[i])
                        C = path[i-1]
                        D = path[i]
                        E = path[i+1] if i+1 < N else None
                        # Actually, we only care about the edges that change
                        old_edges = overlap_mat[A][K] + overlap_mat[K][B] + overlap_mat[C][D]
                        new_edges = overlap_mat[A][B] + overlap_mat[C][K] + overlap_mat[K][D]
                        delta = new_edges - old_edges
                    else:
                        # i < k-1 or i is before k
                        # Original: ... path[i], path[i+1], ..., A, K, B, ...
                        # Remove K: ... path[i], path[i+1], ..., A, B, ...
                        # Insert K between path[i] and path[i+1]: ... path[i], K, path[i+1], ..., A, B, ...
                        # Edges removed: (A,K), (K,B), (path[i], path[i+1])
                        # Edges added: (A,B), (path[i], K), (K, path[i+1])
                        C = path[i]
                        D = path[i+1]
                        old_edges = overlap_mat[A][K] + overlap_mat[K][B] + overlap_mat[C][D]
                        new_edges = overlap_mat[A][B] + overlap_mat[C][K] + overlap_mat[K][D]
                        delta = new_edges - old_edges
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    # Perform the move
                    K = path[k]
                    del path[k]
                    if i > k:
                        i -= 1
                    path.insert(i+1, K)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_path = path[:]
                        
            elif move_type < 0.9:
                # Or-Opt 2: Move two consecutive elements
                k = random.randint(0, N - 2)
                i = random.randint(0, N - 3)
                
                if i == k or i == k - 1 or i == k + 1:
                    continue
                
                K1 = path[k]
                K2 = path[k+1]
                
                if k == 0:
                    if i == 0:
                        continue
                    # Remove first two
                    A = path[2]
                    B = path[i]
                    C = path[i+1]
                    old_edges = overlap_mat[K1][K2] + overlap_mat[K2][A] + overlap_mat[B][C]
                    new_edges = overlap_mat[K1][K2] + overlap_mat[B][K1] + overlap_mat[K2][C]
                    # Wait, this is getting complex. Let's just do adjacent swap for simplicity in SA
                    # and use a separate local search for Or-Opt.
                    continue
                elif k == N - 2:
                    if i == N - 3:
                        continue
                    A = path[k-1]
                    B = path[i]
                    C = path[i+1]
                    old_edges = overlap_mat[A][K1] + overlap_mat[K1][K2] + overlap_mat[B][C]
                    new_edges = overlap_mat[A][K2] + overlap_mat[K1][K2] + overlap_mat[B][K1]
                    # Hmm, this is still complex. Let's simplify.
                    continue
                else:
                    A = path[k-1]
                    B = path[k+2]
                    if i > k + 1:
                        C = path[i-1]
                        D = path[i]
                        E = path[i+1]
                        old_edges = overlap_mat[A][K1] + overlap_mat[K1][K2] + overlap_mat[K2][B] + overlap_mat[C][D]
                        new_edges = overlap_mat[A][K2] + overlap_mat[K1][K2] + overlap_mat[K2][B] + overlap_mat[C][K1]
                        # This is getting too complex. Let's just use adjacent swap in SA.
                        continue
                    else:
                        C = path[i]
                        D = path[i+1]
                        E = path[i+2]
                        old_edges = overlap_mat[A][K1] + overlap_mat[K1][K2] + overlap_mat[K2][B] + overlap_mat[C][D]
                        new_edges = overlap_mat[A][K2] + overlap_mat[K1][K2] + overlap_mat[K2][B] + overlap_mat[C][K1]
                        continue
                
            else:
                # Or-Opt 3: Move three consecutive elements
                # For simplicity, just use adjacent swap
                k = random.randint(0, N - 2)
                if k == 0:
                    delta = overlap_mat[path[1]][path[0]] - overlap_mat[path[0]][path[1]]
                elif k == N - 2:
                    delta = (overlap_mat[path[N-3]][path[N-1]] + overlap_mat[path[N-1]][path[N-2]]) - \
                            (overlap_mat[path[N-3]][path[N-2]] + overlap_mat[path[N-2]][path[N-1]])
                else:
                    delta = (overlap_mat[path[k-1]][path[k+1]] + overlap_mat[path[k+1]][path[k]] + overlap_mat[path[k]][path[k+2]]) - \
                            (overlap_mat[path[k-1]][path[k]] + overlap_mat[path[k]][path[k+1]] + overlap_mat[path[k+1]][path[k+2]])
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    path[k], path[k+1] = path[k+1], path[k]
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_path = path[:]
            
            T *= cooling

        # Final local search polish: Or-Opt 1 with multiple passes
        path = best_path[:]
        current_score = best_score
        
        # Multiple passes of Or-Opt 1
        for pass_num in range(5):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for k in range(N):
                if time.time() > deadline - 10:
                    break
                for i in range(N - 1):
                    if i == k or i == k - 1:
                        continue
                    
                    if k == 0:
                        if i == 0:
                            continue
                        old1 = overlap_mat[path[0]][path[1]]
                        old2 = overlap_mat[path[i]][path[i+1]]
                        new1 = overlap_mat[path[i]][path[0]]
                        new2 = overlap_mat[path[0]][path[i+1]]
                        delta = new1 + new2 - old1 - old2
                    elif k == N - 1:
                        if i == N - 2:
                            continue
                        old1 = overlap_mat[path[N-2]][path[N-1]]
                        old2 = overlap_mat[path[i]][path[i+1]]
                        new1 = overlap_mat[path[i]][path[N-1]]
                        new2 = overlap_mat[path[N-1]][path[i+1]]
                        delta = new1 + new2 - old1 - old2
                    else:
                        A = path[k-1]
                        K = path[k]
                        B = path[k+1]
                        
                        if i > k:
                            C = path[i-1]
                            D = path[i]
                            old_edges = overlap_mat[A][K] + overlap_mat[K][B] + overlap_mat[C][D]
                            new_edges = overlap_mat[A][B] + overlap_mat[C][K] + overlap_mat[K][D]
                            delta = new_edges - old_edges
                        else:
                            C = path[i]
                            D = path[i+1]
                            old_edges = overlap_mat[A][K] + overlap_mat[K][B] + overlap_mat[C][D]
                            new_edges = overlap_mat[A][B] + overlap_mat[C][K] + overlap_mat[K][D]
                            delta = new_edges - old_edges
                    
                    if delta > 0:
                        # Perform the move
                        K = path[k]
                        del path[k]
                        if i > k:
                            i -= 1
                        path.insert(i+1, K)
                        current_score += delta
                        improved = True
                        break  # Restart scan from beginning
                if improved:
                    break
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_path = path[:]

        # Use best path found
        path = best_path

        # Construct S_7 using the optimized path
        # The word is built by concatenating blocks with maximum overlap
        # Each block is: perm + [7] + perm
        # We need to find the maximum overlap between the end of new_word and start of block
        
        new_word = []
        for idx in path:
            p = perms[idx]
            block = list(p) + [7] + list(p)
            # Find maximum overlap: longest suffix of new_word that is prefix of block
            # Since block has length 13, we only need to check up to 13 characters
            max_ov = min(len(new_word), 12)
            overlap = 0
            for o in range(max_ov, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]

        return "".join(map(str, new_word))

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
