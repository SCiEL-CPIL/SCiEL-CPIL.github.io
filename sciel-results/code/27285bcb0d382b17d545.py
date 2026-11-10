"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START

def _is_sidon_with_diagonals(subset):
    """Check if subset has all unordered pair sums unique including diagonals.
    For a set S, the pair sums are {a+b : a,b in S, a<=b}.
    All these sums must be distinct.
    """
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(candidate_set):
    """Find the maximum size of a Sidon subset (with diagonals) within candidate_set.
    Returns the maximum k such that there exists a k-element subset with all pair sums distinct.
    """
    arr = sorted(candidate_set)
    n = len(arr)
    
    # Try from largest possible size down
    for k in range(n, 0, -1):
        # Check if any k-subset is Sidon
        for combo in itertools.combinations(arr, k):
            if _is_sidon_with_diagonals(list(combo)):
                return k
    return 0

def construct():
    """Return exactly 12 distinct integers in [0,128] that minimize the largest Sidon subset size.
    
    Strategy: We want to find a 12-element set where the largest subset with all pair sums
    unique (including diagonals) is as small as possible.
    
    Key insight: If we can make the set such that no 4-element subset is Sidon, the answer is 3.
    If no 3-element subset is Sidon... but any 3-element set {a,b,c} with a<b<c has sums
    a+a, a+b, a+c, b+b, b+c, c+c. For these to be all distinct we need:
    2a < a+b < a+c < 2b < b+c < 2c (not necessarily in that order)
    Actually the sums are: 2a, a+b, a+c, 2b, b+c, 2c
    For all distinct: 2a != a+b (a!=b, true), 2a != a+c (a!=c, true), 2a != 2b (a!=b), 
    2a != b+c, 2a != 2c, a+b != a+c (b!=c), a+b != 2b (a!=b), a+b != b+c (a!=c),
    a+b != 2c, a+c != 2b, a+c != b+c (a!=b), a+c != 2c (a!=c), 2b != b+c (b!=c), 2b != 2c (b!=c)
    
    So the only constraints for a 3-element set to NOT be Sidon are:
    2a == b+c or a+b == 2c or a+c == 2b
    
    These are arithmetic progression conditions! {a,b,c} is not Sidon iff it's an arithmetic progression
    (a+c=2b) or {2a=b+c} or {a+b=2c}.
    
    Actually let me reconsider. For 3 elements a<b<c:
    Sums: 2a, a+b, a+c, 2b, b+c, 2c
    - 2a = b+c means b+c=2a, but b>a and c>a so b+c>2a, impossible
    - a+b = 2c means a+b=2c, but a<b<c so a+b<2c, impossible  
    - a+c = 2b means arithmetic progression
    - 2a = a+b means a=b, impossible
    - 2a = 2b means a=b, impossible
    - a+b = 2b means a=b, impossible
    - a+c = 2c means a=c, impossible
    - b+c = 2c means b=c, impossible
    - 2b = 2c means b=c, impossible
    - a+b = b+c means a=c, impossible
    - a+c = b+c means a=b, impossible
    - 2b = a+c means a+c=2b, arithmetic progression
    - 2b = b+c means b=c, impossible
    - b+c = 2c means b=c, impossible
    - 2a = a+c means a=c, impossible
    - a+b = 2c: a+b < 2c since b<c, so a+b<2c always
    - 2b = 2c: b=c impossible
    
    Wait, I need to be more careful. Let me list all 6 sums for a<b<c:
    2a, a+b, a+c, 2b, b+c, 2c
    
    Check all pairs for equality:
    2a vs a+b: 2a=a+b iff a=b (no)
    2a vs a+c: 2a=a+c iff a=c (no)
    2a vs 2b: 2a=2b iff a=b (no)
    2a vs b+c: 2a=b+c. Since a<b<c, 2a < a+b < a+c < b+c. So 2a < b+c always. No collision.
    2a vs 2c: 2a=2c iff a=c (no)
    a+b vs a+c: a+b=a+c iff b=c (no)
    a+b vs 2b: a+b=2b iff a=b (no)
    a+b vs b+c: a+b=b+c iff a=c (no)
    a+b vs 2c: a+b=2c. Possible if a+b=2c. Since a<b<c, a+b can equal 2c? No, a+b < b+c < 2c (since b<c). So a+b < 2c always.
    a+c vs 2b: a+c=2b iff a+c=2b (arithmetic progression)
    a+c vs b+c: a+c=b+c iff a=b (no)
    a+c vs 2c: a+c=2c iff a=c (no)
    2b vs b+c: 2b=b+c iff b=c (no)
    2b vs 2c: 2b=2c iff b=c (no)
    b+c vs 2c: b+c=2c iff b=c (no)
    
    So for a 3-element set {a,b,c} with a<b<c, it is NOT Sidon iff a+c=2b (arithmetic progression).
    
    Therefore, to prevent any 3-element subset from being Sidon, every triple in our set must form an arithmetic progression. But that's impossible for a set of size > 3 unless the whole set is an arithmetic progression.
    
    Wait, that's not right either. We need EVERY 3-element subset to NOT be Sidon, which means every 3-element subset must be an arithmetic progression. That's only possible if the entire set is an arithmetic progression (or has size <= 2).
    
    So the minimum possible maximum Sidon subset size is at least 3 for any set of size >= 3.
    
    Can we achieve exactly 3? We need a 12-element set where no 4-element subset is Sidon.
    
    For a 4-element set {a<b<c<d}, the pair sums (including diagonals) are:
    2a, a+b, a+c, a+d, 2b, b+c, b+d, 2c, c+d, 2d
    
    There are C(4+1,2) = 10 sums. For it to be Sidon, all 10 must be distinct.
    
    To make a 12-element set where no 4-subset is Sidon, we could try an arithmetic progression. Let's check: if our set is {0, 1, 2, ..., 11} (or any AP), what's the largest Sidon subset?
    
    For an AP {0, 1, 2, ..., n-1}, a subset is Sidon iff... Let me think about small cases.
    
    Actually, let me just try a computational approach. Let me search for a 12-element set in [0,128] that minimizes the max Sidon subset size.
    
    Given the 90s time limit, I'll do a heuristic search.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s safety margin
    
    # First, let's try some structured sets
    # An arithmetic progression
    candidates = []
    
    # Try various APs
    for start in range(0, 120):
        for step in range(1, 12):
            s = set()
            for i in range(12):
                val = start + i * step
                if 0 <= val <= 128:
                    s.add(val)
                else:
                    break
            if len(s) == 12:
                candidates.append(sorted(s))
    
    # Try to find a set with small max Sidon size
    # We'll use a greedy/local search approach
    
    def evaluate_set(s):
        """Return the max Sidon subset size for set s."""
        arr = sorted(s)
        n = len(arr)
        for k in range(n, 0, -1):
            for combo in itertools.combinations(arr, k):
                if _is_sidon_with_diagonals(list(combo)):
                    return k
        return 0
    
    best_set = None
    best_score = 13  # worst case is 12
    
    # Evaluate some initial candidates
    for cand in candidates[:50]:
        if time.time() > deadline:
            break
        score = evaluate_set(cand)
        if score < best_score:
            best_score = score
            best_set = cand
            if best_score <= 3:
                break
    
    # Local search: try to improve by swapping elements
    if best_set is None:
        best_set = list(range(12))
        best_score = evaluate_set(best_set)
    
    # Local search
    current = sorted(best_set)
    current_score = best_score
    
    random.seed(42)
    
    while time.time() < deadline:
        # Try random swaps
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            # Pick a new value not in current set
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            
            candidate = current[:]
            candidate[i] = new_val
            candidate.sort()
            
            score = evaluate_set(candidate)
            if score < current_score:
                current = candidate
                current_score = score
                best_set = candidate
                best_score = score
                improved = True
                if best_score <= 3:
                    break
            elif score == current_score:
                # Accept with some probability (simulated annealing-like)
                if random.random() < 0.3:
                    current = candidate
        
        if best_score <= 3:
            break
    
    return best_set

# EVOLVE-BLOCK-END
