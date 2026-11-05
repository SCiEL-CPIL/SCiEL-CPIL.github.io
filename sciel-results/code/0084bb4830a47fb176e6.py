import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0
    
    total_points = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute neighbors
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        neighbors[idx, 0] = idx
        nb = 1
        for pos in range(n):
            orig = word[pos]
            pw = 1
            for j in range(pos):
                pw *= q
            # +1
            nd = (orig + 1) % q
            delta = -2 * pw if nd == 0 else pw
            neighbors[idx, nb] = idx + delta
            nb += 1
            # -1
            nd = (orig - 1) % q
            delta = 2 * pw if nd == 2 else -pw
            neighbors[idx, nb] = idx + delta
            nb += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def get_counts(code):
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        return counts

    def reduce_code(code, counts):
        current_code = list(code)
        current_counts = counts.copy()
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                if np.all(current_counts[neighbors[c]] > 1):
                    current_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    def fast_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        potential = np.full(total_points, ball_size, dtype=np.int32)
        selected = []
        while True:
            if time.time() > deadline - 40.0:
                break
            maxp = potential.max()
            if maxp <= 0:
                break
            cands = np.where(potential == maxp)[0]
            c = int(cands[rng.integers(len(cands))])
            selected.append(c)
            potential[c] = 0
            for x in neighbors[c]:
                if not is_covered[x]:
                    is_covered[x] = True
                    potential[neighbors[x]] -= 1
                    np.maximum(potential, 0, out=potential)
            if is_covered.all():
                break
        return selected

    def local_search(code, counts, deadline):
        code = list(code)
        counts = counts.copy()
        code_set = set(code)
        best_len = len(code)
        best_code = code[:]
        rng = np.random.default_rng(42)
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            
            # Pass 1: Removals
            for _ in range(20):
                if not code: break
                if time.time() > deadline: break
                c = code[rng.integers(len(code))]
                if np.all(counts[neighbors[c]] > 1):
                    counts[neighbors[c]] -= 1
                    code.remove(c)
                    code_set.discard(c)
                    improved = True
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = code[:]
                    break

            # Pass 2: Moves
            if not improved and code:
                # Pick a codeword with risky points
                # Sample to avoid scanning all
                sample_size = min(50, len(code))
                sample = rng.choice(len(code), size=sample_size, replace=False)
                
                best_move_c = None
                best_move_to = None
                best_score = -1
                
                for s in sample:
                    c = code[s]
                    risky = neighbors[c][counts[neighbors[c]] == 1]
                    if len(risky) == 0:
                        # Should have been removed in Pass 1, but maybe race condition?
                        # Or Pass 1 didn't pick it.
                        continue
                    
                    # Find candidates covering all risky
                    if len(risky) == 1:
                        cand_set = set(neighbors[risky[0]].tolist())
                    else:
                        cand_set = set(neighbors[risky[0]].tolist())
                        for r in risky[1:]:
                            cand_set &= set(neighbors[r].tolist())
                            if not cand_set: break
                    
                    # Filter out existing codewords
                    cand_list = [x for x in cand_set if x not in code_set]
                    if not cand_list:
                        continue
                    
                    # Pick a random candidate or best
                    # Score: number of points in new ball with count == 1 (robustness)
                    # Or just random
                    c_new = cand_list[rng.integers(len(cand_list))]
                    
                    # Validate move
                    # Risky points of c are covered by c_new.
                    # Are there points in Ball(c) not in Ball(c_new) with count 1?
                    # No, because risky = points in Ball(c) with count 1.
                    # So all count-1 points in Ball(c) are in Ball(c_new).
                    # Points in Ball(c) with count > 1 will have count >= 1 after removal of c.
                    # So coverage is maintained.
                    
                    # Update
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_new]] += 1
                    code.remove(c)
                    code.append(c_new)
                    code_set.discard(c)
                    code_set.add(c_new)
                    
                    improved = True
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = code[:]
                    break # Restart loop after one move

            if improved:
                no_improve = 0
            else:
                no_improve += 1
                # Periodic reduce
                if no_improve % 10 == 0:
                    new_code, new_counts = reduce_code(code, counts)
                    if len(new_code) < len(code):
                        code = new_code
                        counts = new_counts
                        code_set = set(code)
                        no_improve = 0
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]

        return best_code, get_counts(best_code)

    best_code = None
    best_len = float('inf')
    
    # Multiple Greedy + Local Search
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        sel = fast_greedy(seed)
        counts = get_counts(sel)
        sel, counts = reduce_code(sel, counts)
        
        # Local Search
        sel, counts = local_search(sel, counts, deadline)
        sel, counts = reduce_code(sel, counts)
        
        if len(sel) < best_len:
            best_len = len(sel)
            best_code = sel

    if best_code is None:
        sel = fast_greedy(1)
        best_code = sel

    result = [idx_to_word(int(idx)) for idx in best_code]
    return result
