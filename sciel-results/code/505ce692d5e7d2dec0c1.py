import math
import time
import random
import numpy as np

def compute_score(u):
    """Compute the score for a given set U using numpy for speed."""
    if not u:
        return 0.0
    u_arr = np.array(u, dtype=np.int64)
    max_u = int(u_arr.max())
    m = 2 * max_u + 1
    if m <= 1:
        return 1.0
    
    # Sumset: u_arr[i] + u_arr[j] for all i, j
    sumset = np.unique(u_arr[:, None] + u_arr[None, :])
    
    # Diffset: u_arr[i] - u_arr[j] for all i, j
    diffset = np.unique(u_arr[:, None] - u_arr[None, :])
    
    r = len(diffset) / len(sumset)
    return 1 + math.log(r) / math.log(m)

def is_golomb_ruler(u):
    """Check if all pairwise differences are distinct."""
    diffs = set()
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            d = u[j] - u[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def greedy_golomb_ruler(n, max_val=200):
    """Generate a Golomb ruler with n marks using greedy approach."""
    marks = [0]
    for _ in range(n - 1):
        best_pos = None
        best_diff_count = 0
        for pos in range(marks[-1] + 1, max_val + 1):
            # Check if adding pos creates duplicate differences
            valid = True
            diffs = set()
            for m in marks:
                d = pos - m
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if valid:
                # Count how many new differences this creates
                if len(diffs) > best_diff_count or best_pos is None:
                    best_pos = pos
                    best_diff_count = len(diffs)
        if best_pos is None:
            break
        marks.append(best_pos)
    return sorted(marks)

def backtracking_golomb_ruler(n, max_val=50):
    """Try to find a Golomb ruler with n marks using backtracking."""
    def backtrack(marks, start):
        if len(marks) == n:
            return list(marks)
        for pos in range(start, max_val + 1):
            # Check validity
            valid = True
            diffs = set()
            for m in marks:
                d = pos - m
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if valid:
                marks.append(pos)
                result = backtrack(marks, pos + 1)
                if result:
                    return result
                marks.pop()
        return None
    
    return backtrack([0], 1)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    best_set = None
    best_score = 0.0
    
    # Phase 1: Try known good Golomb rulers and structured candidates
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 68],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 68, 82],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 68, 82, 97],
    ]
    
    # Add greedy-generated rulers
    for n in range(3, 12):
        if time.time() > deadline:
            break
        ruler = greedy_golomb_ruler(n, max_val=150)
        if len(ruler) == n:
            initial_candidates.append(ruler)
    
    # Try backtracking for small n
    for n in range(3, 8):
        if time.time() > deadline:
            break
        ruler = backtracking_golomb_ruler(n, max_val=40)
        if ruler:
            initial_candidates.append(ruler)
    
    # Evaluate all initial candidates
    for u in initial_candidates:
        if time.time() > deadline:
            break
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        score = compute_score(u)
        if score > best_score:
            best_score = score
            best_set = u
    
    if best_set is None:
        return [0, 1, 3]
    
    # Phase 2: Local search to improve
    current_set = list(best_set)
    current_score = best_score
    
    # Track no-improvement counter for early stopping
    no_improve = 0
    max_no_improve = 300
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        no_improve += 1
        
        # Try adding new elements
        if len(current_set) < 25:
            max_cur = max(current_set)
            # Try positions from 1 to max_cur * 2 + 20
            for new_val in range(1, max_cur * 2 + 21):
                if new_val in current_set:
                    continue
                new_set = sorted(current_set + [new_val])
                score = compute_score(new_set)
                if score > current_score + 1e-10:
                    current_set = new_set
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_set = list(current_set)
                    improved = True
                    no_improve = 0
                    break
        
        # Try removing elements
        if len(current_set) > 3 and not improved:
            for i in range(len(current_set) - 1, 0, -1):  # Don't remove 0
                new_set = current_set[:i] + current_set[i+1:]
                score = compute_score(new_set)
                if score > current_score + 1e-10:
                    current_set = new_set
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_set = list(current_set)
                    improved = True
                    no_improve = 0
                    break
        
        # Try modifying elements (shifting)
        if not improved:
            for i in range(1, len(current_set)):  # Don't modify 0
                orig = current_set[i]
                for delta in [-2, -1, 1, 2, -3, 3, -4, 4, -5, 5]:
                    new_val = orig + delta
                    if new_val < 1 or new_val in current_set:
                        continue
                    new_set = sorted([x if j != i else new_val for j, x in enumerate(current_set)])
                    score = compute_score(new_set)
                    if score > current_score + 1e-10:
                        current_set = new_set
                        current_score = score
                        if score > best_score:
                            best_score = score
                            best_set = list(current_set)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # Random restarts if stuck
        if no_improve >= max_no_improve // 2 and not improved:
            # Try a random perturbation
            if len(current_set) > 3:
                idx = random.randint(1, len(current_set) - 1)
                orig = current_set[idx]
                new_val = orig + random.randint(-10, 10)
                if new_val >= 1 and new_val not in current_set:
                    new_set = sorted([x if j != idx else new_val for j, x in enumerate(current_set)])
                    score = compute_score(new_set)
                    if score > current_score - 0.001:  # Allow slight degradation for exploration
                        current_set = new_set
                        current_score = score
                        no_improve = 0
    
    # Phase 3: Final polish - try all single-element modifications
    if best_set and time.time() < deadline - 5:
        for i in range(1, len(best_set)):
            if time.time() > deadline - 5:
                break
            orig = best_set[i]
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = orig + delta
                if new_val < 1 or new_val in best_set:
                    continue
                new_set = sorted([x if j != i else new_val for j, x in enumerate(best_set)])
                score = compute_score(new_set)
                if score > best_score + 1e-10:
                    best_score = score
                    best_set = new_set
    
    return best_set
