import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays
    neighbors_list = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))

    def run_search(seed):
        random.seed(seed)
        
        # Phase 1: Greedy Set-Cover
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            gains = np.zeros(N, dtype=np.int32)
            uncovered_indices = np.where(uncovered)[0]
            for idx in uncovered_indices:
                gains[neighbors_list[idx]] += 1
            
            for c in code_set:
                gains[c] = -1
            
            best_idx = int(np.argmax(gains))
            max_gain = gains[best_idx]
            
            if max_gain == 0:
                break

            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors_list[best_idx]] = False

        # Phase 2: Local Search
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

        best_code_in_run = list(code)
        best_size_in_run = len(code)

        def greedy_repair(uncovered_mask):
            if not np.any(uncovered_mask):
                return []
            
            added = []
            temp_uncovered = uncovered_mask.copy()
            
            while np.any(temp_uncovered):
                if time.time() > deadline - 2:
                    break
                
                gains = np.zeros(N, dtype=np.int32)
                uncovered_indices = np.where(temp_uncovered)[0]
                for idx in uncovered_indices:
                    gains[neighbors_list[idx]] += 1
                
                for c in code_set:
                    gains[c] = -1
                for c in added:
                    gains[c] = -1
                
                best_idx = int(np.argmax(gains))
                max_gain = gains[best_idx]
                
                if max_gain == 0:
                    break

                added.append(best_idx)
                temp_uncovered[neighbors_list[best_idx]] = False

            return added

        def try_remove():
            removed = False
            for c in list(code):
                if np.min(counts[neighbors_list[c]]) > 1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                    removed = True
            return removed

        def try_swap(c):
            if np.min(counts[neighbors_list[c]]) > 1:
                return False

            crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
            if not crit_points:
                return False

            first_cp = crit_points[0]
            cands = [x for x in neighbors_list[first_cp] if x not in code_set]
            
            best_w = -1
            best_score = -1
            
            for w in cands:
                valid = True
                for k in crit_points:
                    if k not in neighbors_list[w]:
                        valid = False
                        break
                if not valid:
                    continue
                
                score = int(np.sum(counts[neighbors_list[w]] == 1))
                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1
                return True

            return False

        def kick(k):
            if len(code) < k:
                return False

            old_size = len(code)
            indices = random.sample(range(len(code)), k)
            to_remove = [code[i] for i in indices]

            temp_counts = counts.copy()
            for c in to_remove:
                temp_counts[neighbors_list[c]] -= 1

            uncovered_mask = (temp_counts == 0)
            
            if not np.any(uncovered_mask):
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                return True

            added = greedy_repair(uncovered_mask)

            new_size = old_size - k + len(added)

            if new_size < old_size:
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                for c in added:
                    code.append(c)
                    code_set.add(c)
                    counts[neighbors_list[c]] += 1
                return True
            else:
                return False

        # Local Search Loop
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 1. Removal
            while try_remove():
                improved = True
            
            if len(code) < best_size_in_run:
                best_size_in_run = len(code)
                best_code_in_run = list(code)
            
            if time.time() > deadline - 10:
                break

            # 2. Swap
            shuffled_code = list(code)
            random.shuffle(shuffled_code)
            for c in shuffled_code:
                if time.time() > deadline - 10:
                    break
                if c not in code_set:
                    continue
                if try_swap(c):
                    improved = True
            
            if len(code) < best_size_in_run:
                best_size_in_run = len(code)
                best_code_in_run = list(code)
            
            if time.time() > deadline - 10:
                break

            # 3. Kick (1 removal)
            kicks_tried = 0
            max_kicks = 50
            while kicks_tried < max_kicks and time.time() < deadline - 10:
                if kick(1):
                    improved = True
                    kicks_tried = 0
                else:
                    kicks_tried += 1
            
            if len(code) < best_size_in_run:
                best_size_in_run = len(code)
                best_code_in_run = list(code)
            
            if time.time() > deadline - 10:
                break

            # 4. Kick (2 removals)
            if len(code) > 2:
                kicks2_tried = 0
                max_kicks2 = 30
                while kicks2_tried < max_kicks2 and time.time() < deadline - 10:
                    if kick(2):
                        improved = True
                        kicks2_tried = 0
                    else:
                        kicks2_tried += 1
                
                if len(code) < best_size_in_run:
                    best_size_in_run = len(code)
                    best_code_in_run = list(code)

        return best_code_in_run

    global_best = []
    global_best_size = float('inf')
    
    seed = 42
    while time.time() < deadline - 20:
        code_found = run_search(seed)
        if len(code_found) < global_best_size:
            global_best_size = len(code_found)
            global_best = code_found
        seed += 1

    # Final polish on the best code found
    if global_best and time.time() < deadline - 5:
        code = list(global_best)
        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1
        
        # Quick removal pass
        for c in list(code):
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
        global_best = code

    result = [idx_to_tuple(cw) for cw in global_best]
    return result
