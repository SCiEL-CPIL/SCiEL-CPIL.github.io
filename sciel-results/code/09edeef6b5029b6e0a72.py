import random
from itertools import combinations_with_replacement, combinations

def _get_sums(set_a):
    return [a + b for a, b in combinations_with_replacement(set_a, 2)]

def _get_diffs(set_a):
    return {b - a for a, b in combinations(set_a, 2) if b != a}

def _is_valid_sidon(set_a):
    sums = _get_sums(set_a)
    return len(sums) == len(set(sums))

def _try_build_pair(n, k_a, k_b, random_seed):
    random.seed(random_seed)
    numbers = list(range(1, n + 1))
    
    # Strategy: Build A first, then B
    # Shuffle to randomize the greedy choice
    shuffled_nums = numbers[:]
    random.shuffle(shuffled_nums)
    
    A = []
    for x in shuffled_nums:
        if len(A) == k_a:
            break
        candidate_A = A + [x]
        if _is_valid_sidon(candidate_A):
            A = candidate_A
            
    if len(A) < k_a:
        return None

    D_A = _get_diffs(A)
    
    B = []
    # Try to build B avoiding D_A
    # Use a different shuffle or continue from where we left off?
    # Better to re-shuffle or use the remaining numbers?
    # Using remaining numbers might be good, but B can overlap with A.
    # So we should consider all numbers again, or just the remaining ones?
    # The problem says A and B need not be disjoint.
    # So we can pick from all numbers.
    
    shuffled_nums_b = numbers[:]
    random.shuffle(shuffled_nums_b)
    
    for x in shuffled_nums_b:
        if len(B) == k_b:
            break
        candidate_B = B + [x]
        if not _is_valid_sidon(candidate_B):
            continue
        
        D_B = _get_diffs(candidate_B)
        if D_B & D_A:
            continue
            
        B = candidate_B

    if len(B) < k_b:
        return None
        
    return [A, B]

def construct():
    n = 256
    # Target sizes based on difference set bound: |A|(|A|-1) + |B|(|B|-1) <= 255
    # Max sum is 22.
    targets = [
        (15, 7), (14, 8), (13, 9), (12, 10), (11, 11),
        (15, 6), (14, 7), (13, 8), (12, 9), (11, 10),
        (14, 6), (13, 7), (12, 8), (11, 9), (10, 10),
        (12, 7), (11, 8), (10, 9),
        (11, 7), (10, 8), (9, 9),
        (10, 7), (9, 8),
        (9, 7), (8, 8),
        (8, 7)
    ]
    
    best_pair = None
    best_size = 0
    
    # We will try each target size with multiple random seeds
    # To stay within time limits, we limit the number of attempts per target
    attempts_per_target = 50
    
    for k_a, k_b in targets:
        if k_a + k_b <= best_size:
            continue
            
        found = False
        for seed in range(attempts_per_target):
            pair = _try_build_pair(n, k_a, k_b, seed)
            if pair is not None:
                # Verify validity just in case
                A, B = pair
                if len(A) == k_a and len(B) == k_b:
                    if _is_valid_sidon(A) and _is_valid_sidon(B):
                        D_A = _get_diffs(A)
                        D_B = _get_diffs(B)
                        if not (D_A & D_B):
                            best_pair = pair
                            best_size = k_a + k_b
                            found = True
                            break
        
        # If we found a pair for this target, we might still want to check smaller targets?
        # No, we iterate in descending order of sum (mostly).
        # But the list isn't strictly sorted by sum. Let's sort it.
        
    if best_pair is None:
        # Fallback: simple greedy to ensure we return something
        A, B = [], []
        for x in range(1, n + 1):
            # Try adding to A
            cand_A = A + [x]
            if _is_valid_sidon(cand_A):
                D_A = _get_diffs(cand_A)
                D_B = _get_diffs(B)
                if not (D_A & D_B):
                    A = cand_A
            # Try adding to B
            cand_B = B + [x]
            if _is_valid_sidon(cand_B):
                D_A = _get_diffs(A)
                D_B = _get_diffs(cand_B)
                if not (D_A & D_B):
                    B = cand_B
        best_pair = [A, B]
        if not A: A = [1]
        if not B: B = [2]
        
    return best_pair
