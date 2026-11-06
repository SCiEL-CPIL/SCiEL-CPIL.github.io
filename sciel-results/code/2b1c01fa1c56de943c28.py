import time
import random
import numpy as np

BASE = 51975

# All divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
_DIVS = []
for a in range(4):          # 3^a
    for b in range(3):      # 5^b
        for c in range(2):  # 7^c
            for d in range(2):  # 11^d
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    _DIVS.append(v)
_DIVS.sort()
N = len(_DIVS)

# Precompute: for each divisor index i, and residue r, the list of positions
_POS = []
for m in _DIVS:
    lst = []
    for r in range(m):
        lst.append(np.arange(r, BASE, m, dtype=np.int32))
    _POS.append(lst)

# Map from modulus value to its index in _DIVS
_DIV_IDX = {m: i for i, m in enumerate(_DIVS)}


def _score_from_counts(counts):
    """Score = number of positions with zero coverage."""
    return int(np.count_nonzero(counts == 0))


def _build_counts(classes):
    """Build coverage count array from a list of classes."""
    counts = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        idx = _DIV_IDX[m]
        counts[_POS[idx][r]] += 1
    return counts


def _full_greedy(order, deadline):
    """Greedy: for each divisor in order, pick residue covering most uncovered."""
    counts = np.zeros(BASE, dtype=np.int16)
    classes = []
    for i in order:
        if time.time() >= deadline:
            break
        m = _DIVS[i]
        best_r = 0
        best_c = -1
        for r in range(m):
            positions = _POS[i][r]
            c = int(np.count_nonzero(counts[positions] == 0))
            if c > best_c:
                best_c = c
                best_r = r
        classes.append([m, best_r])
        counts[_POS[i][best_r]] += 1
    return classes, counts


def _local_search(classes, counts, deadline, rng):
    """Local search using coverage counts for accurate single-move evaluation."""
    best_score = _score_from_counts(counts)
    best_classes = [c[:] for c in classes]
    best_counts = counts.copy()
    n_classes = len(classes)
    
    # Precompute divisor indices for each class
    div_indices = [_DIV_IDX[m] for m, r in classes]
    
    no_improve_rounds = 0
    max_no_improve = 5
    
    while time.time() < deadline and no_improve_rounds < max_no_improve:
        improved_this_round = False
        
        # Shuffle class order for this round
        order = list(range(n_classes))
        rng.shuffle(order)
        
        for ci in order:
            if time.time() >= deadline:
                return best_classes, best_score
            if time.time() > deadline - 2.0:
                break
            
            m, r = classes[ci]
            di = div_indices[ci]
            
            # Try all residues for this class
            old_positions = _POS[di][r]
            best_r = r
            best_score_candidate = best_score
            
            # First, remove old coverage
            counts[old_positions] -= 1
            
            for new_r in range(m):
                if new_r == r:
                    continue
                new_positions = _POS[di][new_r]
                
                # Temporarily add new coverage
                counts[new_positions] += 1
                candidate_score = _score_from_counts(counts)
                counts[new_positions] -= 1
                
                if candidate_score < best_score_candidate:
                    best_score_candidate = candidate_score
                    best_r = new_r
            
            # Restore old coverage
            counts[old_positions] += 1
            
            if best_r != r:
                # Apply the change
                classes[ci] = [m, best_r]
                counts[old_positions] -= 1
                counts[_POS[di][best_r]] += 1
                best_score = best_score_candidate
                best_classes = [c[:] for c in classes]
                best_counts = counts.copy()
                improved_this_round = True
                
                if best_score == 0:
                    return best_classes, best_score
        
        if improved_this_round:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
    
    return best_classes, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    best_classes = None
    best_score = BASE + 1
    rng = random.Random(42)
    
    # Phase 1: Greedy with different orderings
    # Ordering 1: smallest moduli first
    order_asc = list(range(N))
    classes, counts = _full_greedy(order_asc, deadline)
    s = _score_from_counts(counts)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    
    # Ordering 2: largest moduli first
    order_desc = list(range(N - 1, -1, -1))
    classes, counts = _full_greedy(order_desc, deadline)
    s = _score_from_counts(counts)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    
    # Phase 2: Randomized greedy with local search
    local_search_budget = 60.0  # seconds for local search
    
    while time.time() < deadline - local_search_budget:
        # Random permutation
        order = list(range(N))
        rng.shuffle(order)
        classes, counts = _full_greedy(order, deadline)
        s = _score_from_counts(counts)
        
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
            if best_score == 0:
                return best_classes
        
        # Quick local search on this solution if it's promising
        if s <= best_score + 50 and time.time() < deadline - local_search_budget:
            ls_deadline = min(deadline - 5.0, time.time() + 5.0)
            improved_classes, improved_score = _local_search(
                [c[:] for c in classes], counts.copy(), ls_deadline, rng
            )
            if improved_score < best_score:
                best_score = improved_score
                best_classes = improved_classes
                if best_score == 0:
                    return best_classes
    
    # Phase 3: Final local search on the best solution found
    if best_classes is not None and time.time() < deadline - 3.0:
        counts = _build_counts(best_classes)
        ls_deadline = min(deadline - 2.0, time.time() + (deadline - local_search_budget - time.time()))
        if ls_deadline > time.time() + 1.0:
            best_classes, best_score = _local_search(
                best_classes, counts, ls_deadline, rng
            )
    
    # Ensure we return before the hard deadline
    if time.time() > deadline - 1.0 and best_classes is not None:
        pass  # Just return what we have
    
    return best_classes
