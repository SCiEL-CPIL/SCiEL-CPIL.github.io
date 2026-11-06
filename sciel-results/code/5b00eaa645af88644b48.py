"""Search for ambient 12-set minimizing largest Sidon subset size."""
import random
import time

def _is_sidon(S):
    """Check if all unordered pair sums (including diagonals) are unique."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(A):
    """Exhaustively find the largest Sidon subset of A. Returns size."""
    n = len(A)
    # Check from largest to smallest
    from itertools import combinations
    for k in range(n, 0, -1):
        for combo in combinations(range(n), k):
            cand = [A[i] for i in combo]
            if _is_sidon(cand):
                return k
    return 0

def _max_sidon_size_fast(A, max_check=8):
    """Faster: check if full set is Sidon, then check smaller subsets."""
    n = len(A)
    if _is_sidon(A):
        return n
    # Check size n-1
    for i in range(n):
        cand = A[:i] + A[i+1:]
        if _is_sidon(cand):
            return n - 1
    # Check size n-2
    for i in range(n):
        for j in range(i+1, n):
            cand = A[:i] + A[i+1:j] + A[j+1:]
            if _is_sidon(cand):
                return n - 2
    # Check size 7 down to 1
    from itertools import combinations
    for k in range(7, 0, -1):
        for combo in combinations(range(n), k):
            cand = [A[i] for i in combo]
            if _is_sidon(cand):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def _validate(A):
        if len(A) != 12:
            return False
        if len(set(A)) != 12:
            return False
        if any(x < 0 or x > 128 for x in A):
            return False
        return True

    best_set = None
    best_size = 12

    # Strategy 1: Dense clusters - 3 groups of 4 consecutive
    # Within 4 consecutive {a,a+1,a+2,a+3}: max Sidon is 3
    # With 3 such groups spaced apart, cross-collisions help reduce further
    for gap in range(14, 40, 2):
        for offset in range(0, 129 - 3*gap - 3, 1):
            A = []
            for k in range(3):
                base = offset + k * gap
                if base + 3 > 128:
                    break
                A.extend([base, base+1, base+2, base+3])
            if len(A) == 12 and len(set(A)) == 12:
                ms = _max_sidon_size_fast(A)
                if ms < best_size:
                    best_size = ms
                    best_set = A[:]
                    if best_size <= 4:
                        return best_set

    # Strategy 2: 4 groups of 3 consecutive
    for gap in range(10, 35, 2):
        for offset in range(0, 129 - 4*gap - 2, 1):
            A = []
            for k in range(4):
                base = offset + k * gap
                if base + 2 > 128:
                    break
                A.extend([base, base+1, base+2])
            if len(A) == 12 and len(set(A)) == 12:
                ms = _max_sidon_size_fast(A)
                if ms < best_size:
                    best_size = ms
                    best_set = A[:]
                    if best_size <= 4:
                        return best_set

    # Strategy 3: 6 groups of 2 consecutive
    for gap in range(6, 25, 2):
        for offset in range(0, 129 - 6*gap - 1, 1):
            A = []
            for k in range(6):
                base = offset + k * gap
                if base + 1 > 128:
                    break
                A.extend([base, base+1])
            if len(A) == 12 and len(set(A)) == 12:
                ms = _max_sidon_size_fast(A)
                if ms < best_size:
                    best_size = ms
                    best_set = A[:]
                    if best_size <= 4:
                        return best_set

    # Strategy 4: Random initialization + local search
    rng = random.Random(12345)
    
    def _local_search(A, max_iter=200):
        """Local search: swap elements to reduce max Sidon size."""
        A = sorted(A[:])
        current_size = _max_sidon_size_fast(A)
        for _ in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for i in range(12):
                for j in range(12):
                    if i == j:
                        continue
                    new_A = A[:]
                    new_A[i], new_A[j] = new_A[j], new_A[i]
                    if len(set(new_A)) != 12:
                        continue
                    if any(x < 0 or x > 128 for x in new_A):
                        continue
                    new_size = _max_sidon_size_fast(new_A)
                    if new_size < current_size:
                        A = new_A
                        current_size = new_size
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try random perturbation
                for _ in range(5):
                    if time.time() > deadline - 3.0:
                        break
                    i = rng.randint(0, 11)
                    new_val = rng.randint(0, 128)
                    if new_val in A:
                        continue
                    new_A = A[:]
                    new_A[i] = new_val
                    new_A = sorted(new_A)
                    new_size = _max_sidon_size_fast(new_A)
                    if new_size < current_size:
                        A = new_A
                        current_size = new_size
                        improved = True
                        break
                    elif new_size == current_size and rng.random() < 0.1:
                        A = new_A
                        improved = True
                        break
                if not improved:
                    break
        return A, current_size

    # Multiple random restarts
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        # Initialize with dense structure
        if restart % 3 == 0:
            # 3 groups of 4
            gap = rng.randint(14, 35)
            offset = rng.randint(0, max(0, 128 - 3*gap - 3))
            A = []
            for k in range(3):
                base = offset + k * gap
                if base + 3 <= 128:
                    A.extend([base, base+1, base+2, base+3])
            if len(A) == 12 and len(set(A)) == 12:
                A, sz = _local_search(A, max_iter=100)
                if sz < best_size:
                    best_size = sz
                    best_set = A[:]
                    if best_size <= 4:
                        return best_set
        elif restart % 3 == 1:
            # 4 groups of 3
            gap = rng.randint(10, 30)
            offset = rng.randint(0, max(0, 128 - 4*gap - 2))
            A = []
            for k in range(4):
                base = offset + k * gap
                if base + 2 <= 128:
                    A.extend([base, base+1, base+2])
            if len(A) == 12 and len(set(A)) == 12:
                A, sz = _local_search(A, max_iter=100)
                if sz < best_size:
                    best_size = sz
                    best_set = A[:]
                    if best_size <= 4:
                        return best_set
        else:
            # Pure random
            A = sorted(rng.sample(range(0, 129), 12))
            A, sz = _local_search(A, max_iter=100)
            if sz < best_size:
                best_size = sz
                best_set = A[:]
                if best_size <= 4:
                    return best_set

    if best_set is not None and _validate(best_set):
        return best_set
    
    # Fallback
    return [0, 1, 2, 3, 14, 15, 16, 17, 28, 29, 30, 31]
