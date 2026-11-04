import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (balls of radius 1)
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= radius
    neighbors = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val # Restore
        neighbors.append(neigh)
        
    # Precompute neighbor sets for faster intersection
    neighbor_sets = [set(neigh) for neigh in neighbors]
    
    # 3. Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    
    while np.any(uncovered):
        if time.time() > deadline - 30:
            break
            
        best_idx = -1
        max_gain = 0
        
        for i in range(N):
            if i in code_set:
                continue
            gain = np.sum(uncovered[neighbors[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if max_gain == 0:
            break
            
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors[best_idx]] = False
        
    # 4. Local Search
    # counts[i] is the number of codewords in code that cover point i
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors[idx]] += 1
        
    def get_critical_points(c):
        """Return list of points covered ONLY by codeword c"""
        crit = []
        for p in neighbors[c]:
            if counts[p] == 1:
                crit.append(p)
        return crit

    def is_removable(c):
        """Check if codeword c can be removed"""
        # If any point in neighbors[c] has count 1, it is NOT removable
        for p in neighbors[c]:
            if counts[p] == 1:
                return False
        return True

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # --- Deletion Pass ---
        to_remove = []
        for c in code:
            if is_removable(c):
                to_remove.append(c)
                
        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
            
        # --- Swap Pass ---
        # Iterate through a copy of the code to allow modifications
        # We try to swap codewords that have critical points with other codewords
        # that cover those critical points, hoping to increase redundancy elsewhere.
        
        # Shuffle to avoid bias
        current_code_copy = list(code)
        random.shuffle(current_code_copy)
        
        for c in current_code_copy:
            if time.time() > deadline - 5:
                    break
            
            if c not in code_set:
                continue

            crit_points = get_critical_points(c)
            
            # If no critical points, it was handled in deletion
            if not crit_points:
                continue
                
            # If there are more than 1 critical points, it's harder to replace
            # We can still try, but candidates must cover ALL of them.
            # For simplicity and speed, we prioritize cases with 1 critical point,
            # but we will attempt all.
            
            # Find candidates
            # Candidates must cover all points in crit_points
            # Let's intersect the neighbor sets of all crit_points
            # Since |crit_points| is small, we can iterate neighbors of the first crit point
            if len(crit_points) == 0:
                continue
                
            candidate_set = set(neighbors[crit_points[0]])
            for p in crit_points[1:]:
                candidate_set &= neighbor_sets[p]
            
            # Remove current code from candidates (we can't swap to itself, 
            # and swapping to another existing code word is effectively a move/delete, 
            # but usually we look for NEW positions or different existing ones.
            # Actually, if we swap c with an existing c', we are just moving c.
            # This is valid.
            # However, if c' is in code, removing c and adding c' (which is already there)
            # means we just remove c. This is a deletion.
            # So we should only consider c' NOT in code_set OR c' == c (skip).
            # If c' is in code_set, it's a deletion. Deletions are already handled.
            # So we look for c' NOT in code_set.
            
            candidates = [x for x in candidate_set if x not in code_set]
            
            if not candidates:
                continue
            
            # Evaluate candidates
            best_candidate = -1
            best_score = -1
            
            # Precompute sets for faster membership checks
            neigh_c_set = neighbor_sets[c]
            
            for w in candidates:
                # Simulate swap: remove c, add w
                # We need to check if this swap allows any deletions
                # Or simply if it reduces the number of critical points
                
                # Temporary update counts
                # We don't want to modify global counts unless we commit
                # Let's compute the "gain"
                
                # Points in neighbors[w] that are currently covered by 0 or 1?
                # If we add w, points in neighbors[w] get +1 count.
                # Points in neighbors[c] get -1 count.
                
                # A swap is good if:
                # 1. It covers the critical points of c (guaranteed by candidate selection).
                # 2. It allows other codewords to become removable.
                
                # Let's count how many codewords become removable if we do this swap.
                # This is expensive if done naively.
                
                # Heuristic Score:
                # Score = Number of points in neighbors[w] that have count 1 (before swap) 
                #         - Number of points in neighbors[c] that have count > 1 (before swap, 
                #           because they will lose coverage)?
                
                # Actually, a simpler and effective metric:
                # After swapping, how many points have count 1?
                # We want to MINIMIZE the number of critical points.
                
                # Let's estimate the change in critical points.
                # Old critical points of c are covered by w, so they become count 2 (not critical).
                # Points in neighbors[w] that were count 1 (and not in neighbors[c]) become count 2 (not critical).
                # Points in neighbors[c] that were count 2 (and not in neighbors[w]) become count 1 (become critical).
                
                # Let's just do a cheap check:
                # Does w cover points that are currently count 1?
                # If yes, it might reduce critical points.
                
                score = 0
                for p in neighbors[w]:
                    if counts[p] == 1 and p not in neigh_c_set:
                        # This point becomes count 2, so it stops being critical
                        # But wait, if p was critical, it was covered by some c_i.
                        # If c_i != c, then p is still covered by c_i.
                        # So p goes from 1 to 2. It stops being critical. Good.
                        score += 1
                        
                # Penalty: Points in neighbors[c] that become critical?
                # If p in neighbors[c] has count 2, and is NOT in neighbors[w], it becomes count 1.
                # This is BAD (new critical point).
                for p in neighbors[c]:
                    if counts[p] == 2 and p not in neighbor_sets[w]:
                        score -= 1
                        
                if score > best_score:
                    best_score = score
                    best_candidate = w
                    
            if best_candidate != -1 and best_score > 0:
                # Perform swap
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
                
                code.append(best_candidate)
                code_set.add(best_candidate)
                counts[neighbors[best_candidate]] += 1
                
                improved = True
                # After a swap, run deletion pass immediately to take advantage
                to_remove = []
                for c2 in code:
                    if is_removable(c2):
                        to_remove.append(c2)
                for c2 in to_remove:
                    code.remove(c2)
                    code_set.remove(c2)
                    counts[neighbors[c2]] -= 1

    # Final cleanup
    changed = True
    while changed:
        changed = False
        for c in list(code):
            if is_removable(c):
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
                changed = True

    return [list(words[i]) for i in code]
