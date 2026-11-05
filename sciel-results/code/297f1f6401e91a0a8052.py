"""Enhanced ternary covering code K_3(7,1): heap greedy + multi-restart local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import heapq
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    # Precompute neighbor lists and reverse neighbors
    neighbors_list = []
    neighbor_sets = []
    rev_neighbors = [[] for _ in range(N)]  # rev_neighbors[p] = list of codewords whose neighborhood includes p
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
        neighbor_sets.append(set(nb))
        for p in nb:
            rev_neighbors[p].append(i)

    def greedy_heap(uncovered_mask):
        """Heap-based greedy: maintain a max-heap of (gain, codeword) and update lazily."""
        code = []
        code_set = set()
        counts = np.zeros(N, dtype=np.int32)
        
        # Initial gains: for each codeword, how many uncovered points does it cover?
        # Since all are uncovered initially, gain = len(neighbors) = 16 for all
        # But we need to track which are still uncovered
        
        # Build heap with initial gains
        heap = []
        for c in range(N):
            g = int(np.sum(uncovered_mask[neighbors_list[c]]))
            heapq.heappush(heap, (-g, c))
        
        while heap and np.any(uncovered_mask):
            if time.time() > deadline - 15:
                break
            
            # Pop until we find a valid entry
            while heap:
                neg_gain, c = heapq.heappop(heap)
                gain = -neg_gain
                actual_gain = int(np.sum(uncovered_mask[neighbors_list[c]]))
                if actual_gain == gain and actual_gain > 0:
                    break
            else:
                break
            
            if gain <= 0:
                break
            
            code.append(c)
            code_set.add(c)
            counts[neighbors_list[c]] += 1
            uncovered_mask[neighbors_list[c]] = False
            
            # Update heap: only push codewords whose gain might have changed
            # A codeword's gain changes only if it covers newly covered points
            # That happens for codewords in rev_neighbors of the newly covered points
            newly_covered = neighbors_list[c]
            updated = set()
            for p in newly_covered:
                for cw in rev_neighbors[p]:
                    if cw not in code_set and cw not in updated:
                        updated.add(cw)
                        new_g = int(np.sum(uncovered_mask[neighbors_list[cw]]))
                        heapq.heappush(heap, (-new_g, cw))
        
        return code, code_set, counts

    def local_search(code, code_set, counts, time_limit):
        """Run local search with removal, swap, and kick operations."""
        best_size = len(code)
        best_code = list(code)
        
        def try_remove():
            removed = False
            for c in list(code):
                if c not in code_set:
                    continue
                if np.min(counts[neighbors_list[c]]) > 1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                    removed = True
            return removed

        def try_swap(c):
            """Try to swap codeword c with a better-positioned codeword."""
            if c not in code_set:
                return False
            
            if np.min(counts[neighbors_list[c]]) > 1:
                return False

            crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
            if not crit_points:
                return False

            shuffled_crit = list(crit_points)
            random.shuffle(shuffled_crit)

            for p in shuffled_crit:
                cands = [x for x in neighbor_sets[p] if x not in code_set]
                
                best_w = -1
                best_score = -1

                for w in cands:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Score: maximize redundancy creation
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

            # Relocation strategy
            temp_counts = counts.copy()
            temp_counts[neighbors_list[c]] -= 1
            would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]
            
            if not would_be_uncovered:
                return False

            all_needed = set(crit_points) | set(would_be_uncovered)
            first_point = list(all_needed)[0]
            cands = [x for x in neighbor_sets[first_point] if x not in code_set]

            best_w = -1
            best_score = -1
            for w in cands:
                valid = True
                for k in all_needed:
                    if k not in neighbor_sets[w]:
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

        def greedy_repair(uncovered_set):
            if not uncovered_set:
                return []
            added = []
            temp_uncovered = set(uncovered_set)
            
            while temp_uncovered:
                if time.time() > deadline - 3:
                    break
                gains = np.zeros(N, dtype=np.int32)
                for idx in temp_uncovered:
                    gains[neighbors_list[idx]] += 1
                
                max_gain = int(np.max(gains))
                if max_gain == 0:
                    break
                
                best_indices = np.where(gains == max_gain)[0]
                if len(best_indices) > 1:
                    best_idx = int(random.choice(best_indices))
                else:
                    best_idx = int(best_indices[0])
                
                added.append(best_idx)
                for w in neighbors_list[best_idx]:
                    temp_uncovered.discard(w)
            return added

        def kick(kick_size):
            if len(code) < kick_size:
                return False
            
            old_size = len(code)
            indices = random.sample(range(len(code)), kick_size)
            to_remove = [code[i] for i in indices]
            
            temp_coverage = counts.copy()
            for c in to_remove:
                temp_coverage[neighbors_list[c]] -= 1
            
            uncovered = np.where(temp_coverage == 0)[0]
            if len(uncovered) == 0:
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                return True

            uncovered_set = set(uncovered.tolist())
            added = greedy_repair(uncovered_set)
            new_size = old_size - kick_size + len(added)

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
            return False

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # 1. Exhaustive removal
            while try_remove():
                improved = True
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

            if time.time() > time_limit:
                break

            # 2. Swap pass
            shuffled_code = list(code)
            random.shuffle(shuffled_code)
            for c in shuffled_code:
                if time.time() > time_limit:
                    break
                if try_swap(c):
                    improved = True

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

            if time.time() > time_limit:
                break

            # 3. Kick passes with varied sizes
            for ks in [1, 2, 3]:
                kicks_tried = 0
                max_kicks = 20
                while kicks_tried < max_kicks and time.time() < time_limit:
                    if kick(ks):
                        improved = True
                        kicks_tried = 0
                    else:
                        kicks_tried += 1

                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)

        # Restore best
        if len(best_code) < len(code):
            code.clear()
            code_set.clear()
            counts[:] = 0
            for c in best_code:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1

        return code, code_set, counts

    # ---------- Main: Multiple restarts ----------
    global_best_code = None
    global_best_size = float('inf')

    # Determine number of restarts based on time budget
    # First greedy takes ~2-5s, each local search iteration takes ~1-3s
    # We'll do as many restarts as fit in the time budget

    while time.time() < deadline - 20:
        # Phase 1: Greedy with heap
        uncovered = np.ones(N, dtype=bool)
        code, code_set, counts = greedy_heap(uncovered)
        
        if not code:
            break
        
        # Phase 2: Local search
        # Allocate remaining time for this restart
        remaining = deadline - time.time() - 5
        ls_time = min(remaining * 0.9, 40.0)  # Cap per-restart local search
        code, code_set, counts = local_search(code, code_set, counts, time.time() + ls_time)

        if len(code) < global_best_size:
            global_best_size = len(code)
            global_best_code = list(code)
        
        # If we've reached a very good solution, do a final polish
        if global_best_size <= 188 and time.time() < deadline - 15:
            # Final polish: aggressive removal and swap
            polish_time = deadline - 10
            code, code_set, counts = local_search(code, code_set, counts, polish_time)
            if len(code) < global_best_size:
                global_best_size = len(code)
                global_best_code = list(code)
            break

    if global_best_code is None:
        # Fallback: full space
        return [list(w) for w in itertools.product(range(3), repeat=n)]

    result = [idx_to_tuple(cw) for cw in global_best_code]
    return result
# EVOLVE-BLOCK-END
