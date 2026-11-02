"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations, product

def compute_score(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    maxU = max(U)
    if maxU == 0:
        return 0.0
    
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    sumset_size = len(set(sums.flatten().tolist()))
    
    diffs = arr[:, None] - arr[None, :]
    diffset_size = len(set(diffs.flatten().tolist()))
    
    ratio = diffset_size / sumset_size
    denom = math.log(2 * maxU + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _score_set(U):
    """Alias for compute_score."""
    return compute_score(U)

def _score_fast(U):
    """Faster scoring using numpy, returns (score, diffset_size, sumset_size)."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return 0.0, 0, 0
    maxU = max(U)
    if maxU == 0:
        return 0.0, 0, 0
    
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    sumset_size = len(np.unique(sums))
    
    diffs = arr[:, None] - arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    ratio = diffset_size / sumset_size
    denom = math.log(2 * maxU + 1)
    if denom == 0:
        return 0.0, diffset_size, sumset_size
    return 1.0 + math.log(ratio) / denom, diffset_size, sumset_size

def _golomb_ruler_greedy(n, max_val=200):
    """Generate a Golomb ruler of order n using greedy method."""
    if n <= 1:
        return [0]
    ruler = [0]
    for _ in range(1, n):
        best_pos = None
        best_min_diff = float('inf')
        for pos in range(ruler[-1] + 1, max_val + 1):
            min_diff = float('inf')
            valid = True
            for existing in ruler:
                d = abs(pos - existing)
                if d == 0:
                    valid = False
                    break
                # Check if this difference already exists
                for other in ruler:
                    if other != existing and abs(pos - other) == abs(existing - other) and other != pos:
                        pass
                min_diff = min(min_diff, d)
            # Check all pairwise differences are unique
            diffs = set()
            valid = True
            for i in range(len(ruler) + 1):
                elems = ruler + ([pos] if i == len(ruler) else [])
                # Actually let me just check properly
            # Simpler: check if adding pos creates duplicate differences
            all_diffs = set()
            new_elems = ruler + [pos]
            for i in range(len(new_elems)):
                for j in range(i + 1, len(new_elems)):
                    d = new_elems[j] - new_elems[i]
                    if d in all_diffs:
                        valid = False
                        break
                    all_diffs.add(d)
                if not valid:
                    break
            if valid:
                if min_diff < best_min_diff:
                    best_min_diff = min_diff
                    best_pos = pos
        if best_pos is None:
            break
        ruler.append(best_pos)
    return ruler

def _try_structured_sets():
    """Try various structured set constructions."""
    candidates = []
    
    # Exponential sets {0, 1, b, b^2, ..., b^(n-1)}
    for b in range(2, 10):
        for n in range(2, 10):
            U = [0]
            val = 1
            for i in range(1, n):
                U.append(val)
                val *= b
            candidates.append(sorted(U))
    
    # Golomb rulers
    for n in range(2, 10):
        gr = _golomb_ruler_greedy(n, max_val=300)
        if len(gr) == n:
            candidates.append(gr)
    
    # Sets of form {0, 1, a, a+1, ...}
    for n in range(3, 10):
        for a in range(2, 20):
            U = [0, 1] + list(range(a, a + n - 2))
            if len(set(U)) == n:
                candidates.append(sorted(U))
    
    # Random-ish structured: {0, 1, 2, k, 2k, ...}
    for k in range(3, 15):
        for n in range(3, 8):
            U = [0, 1, 2]
            val = k
            while len(U) < n and val < 500:
                if val not in U:
                    U.append(val)
                val += k
            if len(U) >= 3:
                candidates.append(sorted(U[:n]))
    
    # Geometric with small base
    for base in [2, 3, 4, 5]:
        for n in range(2, 8):
            U = [0]
            val = 1
            for i in range(1, n):
                U.append(val)
                val = val * base + 1  # Slightly different growth
            candidates.append(sorted(U))
    
    return candidates

def _local_search(U, max_iters=5000, deadline=None):
    """Local search to improve the set."""
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    best_score, _, _ = _score_fast(U)
    best_U = list(U)
    n = len(U)
    
    random.seed(42)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        current_score = best_score
        n = len(best_U)
        maxU = max(best_U)
        
        # Generate candidate moves
        moves = []
        
        # 1. Shift individual elements
        for i in range(1, n):
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = best_U[i] + delta
                if new_val > 0 and new_val not in best_U:
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U.sort()
                    moves.append(new_U)
        
        # 2. Replace an element with a random value
        for i in range(1, n):
            for _ in range(8):
                new_val = random.randint(1, maxU + 10)
                if new_val not in best_U:
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U.sort()
                    moves.append(new_U)
        
        # 3. Add a new element
        if n < 15:
            for new_val in [maxU + 1, maxU + 2, maxU + 3, 2 * maxU + 1]:
                if new_val not in best_U:
                    new_U = best_U + [new_val]
                    new_U.sort()
                    moves.append(new_U)
            for _ in range(10):
                new_val = random.randint(1, maxU + 20)
                if new_val not in best_U:
                    new_U = best_U + [new_val]
                    new_U.sort()
                    moves.append(new_U)
        
        # 4. Remove an element (if n > 2)
        if n > 2:
            for i in range(1, n):
                new_U = best_U[:i] + best_U[i+1:]
                moves.append(new_U)
        
        if not moves:
            break
        
        # Evaluate moves and find best
        best_move_score = current_score
        best_move_U = best_U
        
        for new_U in moves:
            if deadline and time.time() > deadline:
                break
            s, _, _ = _score_fast(new_U)
            if s > best_move_score + 1e-10:
                best_move_score = s
                best_move_U = new_U
        
        if best_move_U != best_U:
            best_U = best_move_U
            best_score = best_move_score
    
    return best_U, best_score

def _multi_start_local_search(deadline):
    """Try multiple starting points and local search from each."""
    best_overall_U = [0, 1, 3]
    best_overall_score = compute_score(best_overall_U)
    
    # Get structured candidates
    candidates = _try_structured_sets()
    
    # Also add some random starts
    for _ in range(30):
        n = random.randint(3, 8)
        U = [0]
        for _ in range(n - 1):
            val = random.randint(1, 50)
            if val not in U:
                U.append(val)
        candidates.append(sorted(U))
    
    # Evaluate all candidates first
    scored_candidates = []
    for U in candidates:
        if 0 not in U or len(U) < 2:
            continue
        s, _, _ = _score_fast(U)
        scored_candidates.append((s, U))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local search from top candidates
    for s, U in scored_candidates[:50]:
        if time.time() > deadline:
            break
        if s < best_overall_score - 0.05:
            break  # Skip if too far behind
        
        improved_U, improved_score = _local_search(U, max_iters=2000, deadline=deadline)
        if improved_score > best_overall_score:
            best_overall_score = improved_score
            best_overall_U = improved_U
    
    return best_overall_U, best_overall_score

def _refine_with_additions(U, deadline):
    """Try to improve by carefully adding elements."""
    best_U = list(U)
    best_score, _, _ = _score_fast(best_U)
    maxU = max(best_U)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        best_move = None
        best_move_score = best_score
        
        # Try adding various values
        candidates = set()
        for offset in range(1, 30):
            candidates.add(maxU + offset)
        for _ in range(50):
            candidates.add(random.randint(1, maxU + 30))
        # Also try values between existing elements
        for i in range(len(best_U) - 1):
            mid = (best_U[i] + best_U[i+1]) // 2
            if mid > best_U[i] and mid < best_U[i+1]:
                candidates.add(mid)
        
        for val in candidates:
            if val in best_U or val == 0:
                continue
            new_U = best_U + [val]
            new_U.sort()
            s, _, _ = _score_fast(new_U)
            if s > best_move_score + 1e-10:
                best_move_score = s
                best_move = new_U
        
        if best_move:
            best_U = best_move
            best_score = best_move_score
            maxU = max(best_U)
        else:
            break
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Multi-start local search
    best_U, best_score = _multi_start_local_search(deadline)
    
    # Phase 2: Refine with careful additions
    if time.time() < deadline - 5:
        best_U, best_score = _refine_with_additions(best_U, deadline)
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 2:
        best_U, best_score = _local_search(best_U, max_iters=3000, deadline=deadline)
    
    # Phase 4: Try a few more structured approaches
    if time.time() < deadline - 1:
        # Try sets with specific patterns
        extra_candidates = []
        
        # {0, 1, 2, a, 2a, 3a, ...}
        for a in range(3, 15):
            for n in range(4, 9):
                U = [0, 1, 2]
                for k in range(3, n):
                    U.append(k * a)
                extra_candidates.append(sorted(U))
        
        # {0, 1, b, b+1, 2b, 2b+1, ...}
        for b in range(3, 12):
            for n in range(4, 9):
                U = [0, 1]
                for k in range(b, b + n - 2, 2):
                    if len(U) < n:
                        U.append(k)
                extra_candidates.append(sorted(U))
        
        for U in extra_candidates:
            if time.time() > deadline:
                break
            s, _, _ = _score_fast(U)
            if s > best_score:
                best_score = s
                best_U = U
                # Quick local search
                if time.time() < deadline - 5:
                    improved_U, improved_score = _local_search(U, max_iters=1000, deadline=deadline)
                    if improved_score > best_score:
                        best_score = improved_score
                        best_U = improved_U
    
    return best_U

# EVOLVE-BLOCK-END
