import time
import itertools
from math import log

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    return 1.0 + log(ratio) / log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 110  # 110s deadline, 40s safety margin
    
    def greedy_construct(target_size, max_val_limit=10**6):
        """
        Greedily builds a set by adding the element that maximizes the score.
        """
        U = [0, 1]
        if target_size <= 2:
            return U
        
        # Candidate pool: Integers that seem "interesting"
        # We will generate candidates dynamically based on current U
        # Candidates: 
        # 1. Small integers not in U
        # 2. Multiples of existing elements
        # 3. Gaps
        # To keep it fast, we limit the candidate pool size.
        
        for _ in range(target_size - len(U)):
            if time.time() > deadline:
                break
            
            current_score = compute_score(U)
            max_u = max(U)
            
            # Generate candidates
            candidates = set()
            # Small numbers
            for i in range(1, min(50, max_u + 10)):
                if i not in U:
                    candidates.add(i)
            # Multiples
            for u in U:
                if u == 0: continue
                for m in range(2, 5):
                    c = u * m
                    if c > max_u and c < max_val_limit:
                        candidates.add(c)
            # Geometric-like extensions
            if len(U) > 1:
                last = U[-1]
                prev = U[-2]
                ratio = last / prev if prev > 0 else 2
                next_est = int(last * ratio)
                for delta in range(-5, 6):
                    c = next_est + delta
                    if c > max_u and c < max_val_limit:
                        candidates.add(c)
            
            # Limit candidates to top K by some heuristic (e.g., larger is better for max_u term?)
            # Actually, we need to evaluate score.
            # To save time, we can filter candidates that are too close to existing ones?
            # No, just evaluate a subset.
            # Sort candidates by "novelty" or just take a random sample if too many?
            # Let's take the largest 50 candidates and the smallest 20.
            candidates_list = sorted(candidates)
            if len(candidates_list) > 70:
                # Keep small ones and large ones
                candidates_list = candidates_list[:20] + candidates_list[-50:]
            
            best_candidate = None
            best_new_score = current_score
            
            for c in candidates_list:
                if c in U:
                    continue
                new_U = U + [c]
                # Fast score check?
                # compute_score is O(n^2). n is small (<100).
                s = compute_score(new_U)
                if s > best_new_score:
                    best_new_score = s
                    best_candidate = c
            
            if best_candidate is not None:
                U.append(best_candidate)
            else:
                # If no improvement, break or add a random large number?
                # Adding a number that doesn't improve score might hurt if it increases max_u too much.
                # Break if stuck.
                break
                
        return U

    # Strategy:
    # 1. Try greedy construction for various target sizes.
    # 2. Try structured families (powers) as seeds.
    # 3. Polish the best result with local search.

    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try structured families first to get good seeds
    # Family 1: Powers of 2
    for k in range(3, 25):
        if time.time() > deadline:
            break
        U = [2**i for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Family 2: Powers of 3
    for k in range(3, 20):
        if time.time() > deadline:
            break
        U = [3**i for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Run Greedy Construction
    # Start from small sizes and grow
    for target_size in [10, 20, 30, 40, 50, 60]:
        if time.time() > deadline:
            break
        U = greedy_construct(target_size)
        if U:
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    # Local Search Polish on the best_U
    # Try to improve by swapping or adding/removing elements
    U = best_U[:]
    current_score = best_score
    max_u = max(U) if U else 0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Generate candidates for local search
        candidates = set()
        # Neighbors of existing elements
        for u in U:
            for delta in range(-5, 6):
                c = u + delta
                if c >= 0 and c not in U:
                    candidates.add(c)
        # Multiples
        for u in U:
            if u == 0: continue
            for m in range(2, 4):
                c = u * m
                if c < 4 * max_u:
                    candidates.add(c)
        
        # Try adding
        for c in candidates:
            if c in U: continue
            new_U = U + [c]
            s = compute_score(new_U)
            if s > current_score:
                current_score = s
                U = new_U
                improved = True
                break
        
        if improved:
            continue
            
        # Try removing (if size > 2)
        if len(U) > 2:
            for i in range(len(U)):
                # Don't remove 0
                if U[i] == 0: continue
                new_U = U[:i] + U[i+1:]
                s = compute_score(new_U)
                if s > current_score:
                    current_score = s
                    U = new_U
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try swapping
        if len(U) > 2:
            for i in range(len(U)):
                if U[i] == 0: continue
                for c in candidates:
                    if c in U: continue
                    new_U = U[:]
                    new_U[i] = c
                    # Ensure sorted and unique
                    new_U = sorted(set(new_U))
                    if 0 not in new_U: continue
                    s = compute_score(new_U)
                    if s > current_score:
                        current_score = s
                        U = new_U
                        improved = True
                        break
                if improved:
                    break

    if current_score > best_score:
        best_score = current_score
        best_U = U
        
    return best_U
