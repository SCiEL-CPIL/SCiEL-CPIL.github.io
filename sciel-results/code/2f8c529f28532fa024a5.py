import itertools
import time

def _is_sidon(subset):
    """Check if subset has all pair sums (i<=j) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(S):
    """Compute the largest Sidon subset size of S."""
    n = len(S)
    # Check all subsets in decreasing size order
    for k in range(n, 2, -1):
        for subset in itertools.combinations(S, k):
            if _is_sidon(subset):
                return k
    # Check size 3
    for subset in itertools.combinations(S, 3):
        if _is_sidon(subset):
            return 3
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)  # Should be 4
    
    # Try to find sets with max Sidon size 3 (4-Sidon-free)
    # Strategy: use structured sets with many additive collisions
    
    # Approach: search over sets constructed from arithmetic progressions
    # and other structured patterns
    
    # First, try sets that are unions of small APs
    # A set where every 4-subset has a collision
    
    # Key insight: if S contains 4 elements a,b,c,d with a+d=b+c, 
    # then no 4-Sidon can contain all four.
    # If S is "4-Sidon-free", max Sidon size <= 3.
    
    # Try: sets with many 4-term relations
    # Pattern: {0,1,2,3} has 0+3=1+2
    # Tile with large gaps: {0,1,2,3, 16,17,18,19, 32,33,34,35}
    
    candidates = []
    
    # Candidate 1: Tiled {0,1,2,3} pattern
    for offset in range(0, 100):
        base = [offset + 0, offset + 1, offset + 2, offset + 3]
        # Add more elements with collision structure
        pass
    
    # Let's try random search with structured initialization
    import random
    random.seed(42)
    
    # Generate many candidate sets and evaluate
    # Focus on dense-ish sets with structure
    
    tried = set()
    best_set = list(range(12))
    best_size = 4  # _max_sidon_size([0..11]) = 4
    
    # First verify
    test_size = _max_sidon_size(list(range(12)))
    best_size = test_size
    
    # Try to find 4-Sidon-free sets
    # A set is 4-Sidon-free if every 4-element subset has a sum collision
    
    # Strategy: use sets with many elements in APs
    # {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22} - all even
    # This is isomorphic to {0..11} * 2, so same Sidon structure
    
    # Better: mix APs
    # Try {0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16}
    # This has many collisions
    
    structured_candidates = [
        list(range(12)),
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 4, 8, 9, 11, 15, 16, 18, 23, 24],
        [0, 1, 3, 7, 9, 11, 13, 15, 19, 21, 25, 27],
    ]
    
    for cand in structured_candidates:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        key = tuple(sorted(cand))
        if key in tried:
            continue
        tried.add(key)
        sz = _max_sidon_size(sorted(cand))
        if sz < best_size:
            best_size = sz
            best_set = sorted(cand)
            if best_size <= 3:
                break
    
    if best_size <= 3:
        return best_set
    
    # Random search
    attempts = 0
    while time.time() < deadline and attempts < 50000:
        # Generate random sets with structure bias
        if attempts % 4 == 0:
            # Dense random
            S = sorted(random.sample(range(0, 30), 12))
        elif attempts % 4 == 1:
            # Structured: two APs
            a = random.randint(0, 5)
            d1 = random.randint(1, 3)
            d2 = random.randint(1, 3)
            base = random.randint(0, 20)
            S = sorted(set([base + i*d1 for i in range(6)] + 
                          [base + random.randint(10, 30) + i*d2 for i in range(6)]))
            if len(S) < 12:
                while len(S) < 12:
                    S.append(random.randint(0, 50))
                    S = sorted(set(S))
        elif attempts % 4 == 2:
            # Medium density
            S = sorted(random.sample(range(0, 60), 12))
        else:
            # Very dense
            S = sorted(random.sample(range(0, 20), 12))
        
        key = tuple(S)
        if key in tried:
            attempts += 1
            continue
        tried.add(key)
        
        sz = _max_sidon_size(S)
        if sz < best_size:
            best_size = sz
            best_set = S
            if best_size <= 3:
                break
        attempts += 1
    
    return best_set
