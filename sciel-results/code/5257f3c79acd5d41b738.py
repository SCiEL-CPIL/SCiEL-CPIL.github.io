"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import itertools

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the score for a finite set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    maxU = max(U)
    if maxU == 0:
        return -1
    # Compute |U+U| and |U-U|
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ss = len(sumset)
    ds = len(diffset)
    if ss == 0:
        return -1
    ratio = ds / ss
    if ratio <= 0:
        return -1
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom

def _local_search(U, max_iter=500, time_limit=10.0):
    """Local search: try adding, removing, or shifting elements."""
    start = time.time()
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = list(U)
    
    max_val = max(U) if U else 10
    # Search space for new elements
    search_range = max(max_val * 2, 50)
    
    for iteration in range(max_iter):
        if time.time() - start > time_limit:
            break
        improved = False
        
        # Try adding a new element
        candidates_add = []
        if len(U) < 30:
            candidates_add = list(range(1, search_range + 1))
        
        # Try removing an element (keep 0)
        candidates_remove = [x for x in U if x != 0]
        
        # Try shifting a non-zero element
        candidates_shift = []
        for x in U:
            if x == 0:
                continue
            for delta in [-3, -2, -1, 1, 2, 3, 5, 10]:
                new_x = x + delta
                if new_x >= 0 and new_x != x:
                    candidates_shift.append((x, new_x))
        
        all_moves = []
        for x in candidates_add:
            if x not in U:
                all_moves.append(('add', x))
        for x in candidates_remove:
            all_moves.append(('remove', x))
        for old, new in candidates_shift:
            if new not in U or new == old:
                all_moves.append(('shift', old, new))
        
        random.shuffle(all_moves)
        
        for move in all_moves[:20]:  # Limit moves per iteration for speed
            if time.time() - start > time_limit:
                break
            if move[0] == 'add':
                new_U = list(U) + [move[1]]
            elif move[0] == 'remove':
                new_U = [x for x in U if x != move[1]]
            else:
                new_U = [move[2] if x == move[1] else x for x in U]
            
            if 0 not in new_U or len(new_U) < 2:
                continue
            
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                improved = True
                break
        
        if not improved:
            # Try larger random perturbations
            for _ in range(5):
                if time.time() - start > time_limit:
                    break
                new_U = list(U)
                # Randomly add 1-2 elements
                for _ in range(random.randint(1, 2)):
                    new_U.append(random.randint(1, search_range))
                # Randomly remove 1 element (not 0)
                if len(new_U) > 3:
                    rem = random.choice([x for x in new_U if x != 0])
                    new_U.remove(rem)
                
                if 0 not in new_U or len(new_U) < 2:
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        if not improved:
            # Accept small improvements or restart
            pass
    
    return best_U, best_score

def _construct_candidate(seed_params):
    """Construct a candidate set from parameters."""
    U = {0}
    for p in seed_params:
        if p >= 0:
            U.add(p)
    return sorted(U)

def construct():
    deadline = time.time() + 145  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try several structured initial candidates
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 6],
        [0, 1, 2, 3, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 2, 4, 5],
        [0, 1, 3, 5, 6],
        [0, 1, 2, 3, 4, 6],
        [0, 1, 3, 4, 5, 7],
        [0, 1, 2, 4, 6, 7],
        # Sparse structures
        [0, 1, 4, 5, 8],
        [0, 1, 3, 7, 8],
        [0, 2, 3, 5, 7],
        [0, 1, 4, 6, 7],
        # Larger
        [0, 1, 2, 4, 5, 7, 8],
        [0, 1, 3, 4, 6, 7, 9],
    ]
    
    for cand in initial_candidates:
        if time.time() > deadline:
            break
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_U = list(cand)
    
    # Local search from best
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iter=300, time_limit=min(15, deadline - time.time() - 5))
    
    # Try more random starts
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5 and attempts < 50:
        attempts += 1
        # Generate random candidate
        size = random.randint(3, 12)
        max_val = random.randint(10, 60)
        U = {0}
        while len(U) < size:
            U.add(random.randint(1, max_val))
        U = sorted(U)
        
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
        
        # Quick local search
        if time.time() < deadline - 5:
            U2, s2 = _local_search(U, max_iter=100, time_limit=min(5, deadline - time.time() - 5))
            if s2 > best_score:
                best_score = s2
                best_U = U2
    
    # Final polish: try small perturbations on best
    if time.time() < deadline - 2:
        best_U, best_score = _local_search(best_U, max_iter=200, time_limit=min(8, deadline - time.time() - 2))
    
    return best_U
# EVOLVE-BLOCK-END
