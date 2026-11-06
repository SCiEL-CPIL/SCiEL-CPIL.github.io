import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin for 120s limit

    N = 16
    ALL_TRIPLES = []
    ALL_TRIPLE_MASKS = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                mask = (1 << a) | (1 << b) | (1 << c)
                ALL_TRIPLES.append((a, b, c))
                ALL_TRIPLE_MASKS.append(mask)
    
    NUM_TRIPLES = len(ALL_TRIPLES)
    
    # Precompute supersets for each triple mask
    # supersets[i] is a list of masks S such that ALL_TRIPLE_MASKS[i] is subset of S
    supersets = [[] for _ in range(NUM_TRIPLES)]
    for i, mask in enumerate(ALL_TRIPLE_MASKS):
        # Iterate over all supersets of mask
        # There are 2^(16-3) = 8192 supersets
        # We can generate them by taking the complement bits
        complement = ((1 << N) - 1) ^ mask
        # Iterate over all subsets of complement
        sub = complement
        while True:
            supersets[i].append(mask | sub)
            if sub == 0:
                break
            sub = (sub - 1) & complement
            
    # Precompute conflicts
    # conflicts[i] is a list of indices j such that triple i and j conflict
    conflicts = [[] for _ in range(NUM_TRIPLES)]
    for i in range(NUM_TRIPLES):
        mi = ALL_TRIPLE_MASKS[i]
        for j in range(i + 1, NUM_TRIPLES):
            mj = ALL_TRIPLE_MASKS[j]
            if (mi & mj).bit_count() >= 2:
                conflicts[i].append(j)
                conflicts[j].append(i)

    # Precompute masks by size for alpha calculation
    masks_by_size = [[] for _ in range(N + 1)]
    for mask in range(1 << N):
        k = mask.bit_count()
        masks_by_size[k].append(mask)

    def compute_alpha(count):
        # count is an array of size 2^N
        # We want the largest k such that there exists a mask S of size k with count[S] == 0
        # We assume current alpha is known, we can search downwards
        # But for simplicity, we search from N down to 1
        # Optimization: We know alpha is small. Let's start from 8 down.
        # Actually, just search from N down. C(16, k) is small for small k.
        # But for k=16, 1 mask. k=15, 16 masks. ... k=8, 12870 masks.
        # Total masks 65536.
        # Checking 65536 masks is fast.
        for k in range(N, 0, -1):
            for mask in masks_by_size[k]:
                if count[mask] == 0:
                    return k
        return 0

    # Initialize system
    # Greedy maximal
    in_system = [False] * NUM_TRIPLES
    system_set = set()
    conflict_count = [0] * NUM_TRIPLES
    count = [0] * (1 << N)
    
    # We need to update count and conflict_count when adding/removing
    def add_triple(idx):
        in_system[idx] = True
        system_set.add(idx)
        mask = ALL_TRIPLE_MASKS[idx]
        for s in supersets[idx]:
            count[s] += 1
        for j in conflicts[idx]:
            conflict_count[j] += 1

    def remove_triple(idx):
        in_system[idx] = False
        system_set.remove(idx)
        mask = ALL_TRIPLE_MASKS[idx]
        for s in supersets[idx]:
            count[s] -= 1
        for j in conflicts[idx]:
            conflict_count[j] -= 1

    # Greedy initialization
    indices = list(range(NUM_TRIPLES))
    random.shuffle(indices)
    for idx in indices:
        if conflict_count[idx] == 0:
            add_triple(idx)
            
    current_alpha = compute_alpha(count)
    best_alpha = current_alpha
    best_system = list(system_set)
    
    # Local Search
    while time.time() - start_time < DEADLINE:
        if not system_set:
            break
            
        # Pick random triple to remove
        t_old = random.choice(list(system_set))
        
        # Remove t_old
        remove_triple(t_old)
        
        # Find candidate t_new
        t_new = -1
        for idx in range(NUM_TRIPLES):
            if not in_system[idx] and conflict_count[idx] == 0:
                t_new = idx
                break
                
        if t_new != -1:
            # Add t_new
            add_triple(t_new)
            
            # Compute new alpha
            new_alpha = compute_alpha(count)
            
            # Acceptance criteria
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_system = list(system_set)
            elif new_alpha == best_alpha:
                # Accept with some probability to explore
                if random.random() < 0.1:
                    best_system = list(system_set)
            else:
                # Revert
                remove_triple(t_new)
                add_triple(t_old)
        else:
            # No candidate, revert
            add_triple(t_old)

    # Convert best_system indices to triples
    result = []
    for idx in best_system:
        result.append(list(ALL_TRIPLES[idx]))
        
    return result
