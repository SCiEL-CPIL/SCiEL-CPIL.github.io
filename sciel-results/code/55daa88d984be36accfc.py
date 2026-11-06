import random
import time
from itertools import combinations

# Precompute all 4-subsets of 16
N = 16
K = 4
ALL_COMBOS = list(combinations(range(1, N + 1), K))
# Convert to bitmasks for faster intersection
ALL_SETS = [0] * len(ALL_COMBOS)
for i, c in enumerate(ALL_COMBOS):
    mask = 0
    for x in c:
        mask |= (1 << (x - 1))
    ALL_SETS[i] = mask

# Precompute conflict counts for each set
# conflict_count[i] = number of pairs (A, B) such that (A, B, ALL_SETS[i]) is a sunflower
# Formula: Sum over S subset of C of N_pairs(4 - |S|)
# where N_pairs(k) = comb(12, 2k) * factorial(2k) / (factorial(k)^2)
# We only need k=1,2,3,4 (since |S| can be 3,2,1,0)
# k=1 (|S|=3): comb(12, 2) * 2! / (1!^2) = 66 * 2 = 132
# k=2 (|S|=2): comb(12, 4) * 4! / (2!^2) = 495 * 24 / 4 = 2970
# k=3 (|S|=1): comb(12, 6) * 6! / (3!^2) = 924 * 720 / 36 = 18480
# k=4 (|S|=0): comb(12, 8) * 8! / (4!^2) = 495 * 40320 / 576 = 34650

N_PAIRS = {
    1: 132,
    2: 2970,
    3: 18480,
    4: 34650
}

def compute_conflict_counts():
    counts = [0] * len(ALL_SETS)
    for i, mask in enumerate(ALL_SETS):
        total = 0
        # Iterate over all subsets S of mask
        sub = mask
        while True:
            pc = bin(sub).count('1')
            k = 4 - pc
            if k in N_PAIRS:
                total += N_PAIRS[k]
            if sub == 0:
                break
            sub = (sub - 1) & mask
        counts[i] = total
    return counts

CONFLICT_COUNTS = compute_conflict_counts()

def can_add_mask(new_mask, family_masks):
    """
    Check if new_mask can be added to family_masks without forming a sunflower.
    Sunflower: A, B, C such that A&B == A&C == B&C.
    """
    m = len(family_masks)
    for i in range(m):
        a = family_masks[i]
        inter_as = a & new_mask
        for j in range(i + 1, m):
            b = family_masks[j]
            inter_bs = b & new_mask
            if inter_as != inter_bs:
                continue
            # If intersections with new_mask are equal, check if A&B is also equal to that
            if (a & b) == inter_as:
                return False
    return True

def greedy_heuristic(seed=0):
    random.seed(seed)
    indices = list(range(len(ALL_SETS)))
    
    # Sort by conflict count ascending, then randomize ties?
    # To ensure diversity, we can shuffle groups of similar conflict counts?
    # Or just sort by (conflict_count, random_value)
    # This ensures that lower conflict sets are tried first, but randomly among themselves.
    random_vals = [random.random() for _ in range(len(indices))]
    # Sort indices based on (conflict_count, random_val)
    # This ensures that lower conflict sets are tried first, but randomly among themselves.
    sorted_indices = sorted(indices, key=lambda i: (CONFLICT_COUNTS[i], random_vals[i]))
    
    family_masks = []
    for idx in sorted_indices:
        s = ALL_SETS[idx]
        if can_add_mask(s, family_masks):
            family_masks.append(s)
    return family_masks

def construct():
    start_time = time.time()
    timeout = 50.0 # Safety margin for 60s limit
    
    best_family = []
    best_size = 0
    
    seed_counter = 0
    while time.time() - start_time < timeout:
        fam = greedy_heuristic(seed=seed_counter)
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam
            
        seed_counter += 1
        
    # Convert best_family (list of masks) to list of sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(N):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result

if __name__ == "__main__":
    # Quick self-test
    # res = construct()
    # print(f"Size: {len(res)}")
    pass
