import math
import time
import random

def compute_score(u):
    """Compute the score for a set u of nonnegative integers containing 0."""
    if not u or 0 not in u:
        return 0.0
    max_u = max(u)
    m = 2 * max_u + 1
    if m <= 1:
        return 1.0
    
    n = len(u)
    # Sumset: all pairwise sums
    sumset = set()
    for i in range(n):
        for j in range(i, n):
            sumset.add(u[i] + u[j])
    
    # Diffset: all pairwise differences (absolute value gives symmetric coverage)
    # But we need actual U-U, so we need all a-b for a,b in U
    diffset = set()
    for i in range(n):
        for j in range(n):
            diffset.add(u[i] - u[j])
    
    r = len(diffset) / len(sumset)
    return 1 + math.log(r) / math.log(m)

def is_golomb(u):
    """Check if u is a Golomb ruler (all pairwise differences are distinct)."""
    n = len(u)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = u[j] - u[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def local_search(u, deadline, max_iterations=5000):
    """Local search to improve the score."""
    u = list(u)
    current_score = compute_score(u)
    best_set = list(u)
    best_score = current_score
    
    n = len(u)
    max_val = max(u) if u else 0
    
    # Try adding new elements, removing elements, and replacing elements
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        random.shuffle(u)  # Randomize order to explore different directions
        
        # Try adding a new element
        if n < 30:  # Limit set size for computational feasibility
            candidates_to_add = list(range(max_val + 1, max_val + 50))
            random.shuffle(candidates_to_add)
            for new_val in candidates_to_add[:10]:
                new_u = sorted(u + [new_val])
                if 0 not in new_u:
                    new_u = [0] + new_u
                new_u = sorted(set(new_u))
                new_score = compute_score(new_u)
                if new_score > current_score + 1e-9:
                    u = new_u
                    current_score = new_score
                    improved = True
                    break
        
        if time.time() > deadline:
            break
        
        # Try removing an element
        if n > 2:
            for i in range(n - 1, -1, -1):  # Don't remove 0
                if u[i] == 0:
                    continue
                new_u = u[:i] + u[i+1:]
                new_score = compute_score(new_u)
                if new_score > current_score + 1e-9:
                    u = new_u
                    current_score = new_score
                    improved = True
                    break
        
        if time.time() > deadline:
            break
        
        # Try replacing an element
        for i in range(len(u)):
            if u[i] == 0:
                continue
            old_val = u[i]
            # Try nearby values
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val < 0 or new_val == old_val:
                    continue
                new_u = u[:]
                new_u[i] = new_val
                new_u = sorted(set(new_u))
                if 0 not in new_u:
                    new_u = [0] + new_u
                new_u = sorted(new_u)
                new_score = compute_score(new_u)
                if new_score > current_score + 1e-9:
                    u = new_u
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(u)
        
        if not improved:
            # Try a random perturbation
            if n > 2:
                i = random.randint(1, n - 1)
                delta = random.randint(-10, 10)
                new_val = u[i] + delta
                if new_val >= 0 and new_val != u[i]:
                    new_u = u[:]
                    new_u[i] = new_val
                    new_u = sorted(set(new_u))
                    if 0 not in new_u:
                        new_u = [0] + new_u
                    new_u = sorted(new_u)
                    new_score = compute_score(new_u)
                    if new_score > current_score:
                        u = new_u
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(u)
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Known Golomb rulers (verified optimal or near-optimal)
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109],
    ]
    
    # Also add some non-Golomb sets that might have better ratio
    additional_candidates = [
        [0, 1, 2, 4],
        [0, 1, 2, 4, 7],
        [0, 1, 2, 4, 7, 11],
        [0, 1, 3, 5, 8],
        [0, 1, 3, 5, 8, 12],
        [0, 1, 3, 5, 8, 12, 16],
        [0, 1, 3, 6, 9, 12, 15],
        [0, 1, 3, 6, 10, 14, 18],
        [0, 1, 4, 7, 10, 13, 16, 19],
        [0, 1, 4, 7, 10, 13, 16, 19, 22],
    ]
    
    candidates.extend(additional_candidates)
    
    best_set = [0, 1, 3]
    best_score = 0.0
    
    # Evaluate all candidates
    for u in candidates:
        if time.time() > deadline - 5:
            break
        
        if 0 not in u:
            u = [0] + u
        u = sorted(set(u))
        
        score = compute_score(u)
        
        if score > best_score:
            best_score = score
            best_set = list(u)
    
    # Local search from the best candidate
    if time.time() < deadline - 10:
        improved_set, improved_score = local_search(best_set, deadline)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    return best_set
