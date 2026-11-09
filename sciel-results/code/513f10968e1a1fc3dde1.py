import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Covering design C(v,k,t): find few k-subsets of {0..v-1} covering all t-subsets.
    Strategy:
      1) Random construction: sample random k-subsets until expected uncovered < 1.
      2) Greedy repair: for each uncovered t-subset, pick the best k-subset (containing it)
         that maximizes new coverage, add it.
      3) Local search: try to remove redundant blocks and swap in better ones.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    
    # Precompute for each t-subset, the number of k-subsets containing it
    # A k-subset contains a t-subset if it includes all t elements.
    # Number of such k-subsets = C(v-t, k-t)
    # Expected uncovered after m random blocks = n_t * (1 - 1/C(v-t,k-t))^m
    # We want this < 1, so m > ln(n_t) * C(v-t, k-t) / 1 roughly
    # But we can just iterate.
    
    # For speed, we'll track coverage with a count array
    # Index: we need to map t-subsets to indices.
    # With v=19, t=5, n_t = C(19,5) = 11628. That's manageable.
    
    # Map t-subset -> index
    t_sub_index = {}
    for i, sub in enumerate(all_t_subs):
        t_sub_index[sub] = i
    
    # For each k-subset, precompute the list of t-subset indices it covers
    # This is expensive to precompute for all k-subsets, so we'll compute on the fly.
    
    # Actually, let's precompute for each t-subset, which k-subsets contain it?
    # No, that's C(v-t, k-t) per t-subset, which is C(14,3)=364. Total 11628*364 ~ 4.2M.
    # That's a lot but might be okay. Let's not precompute all k-subsets.
    
    # Better approach: for each t-subset, we know the k-subsets containing it are 
    # {t-subset} union S where S is a (k-t)-subset of the remaining v-t points.
    
    # Let's use a different approach for the greedy repair:
    # For a given uncovered t-subset T, we want to find a k-subset B containing T
    # that maximizes the number of newly covered t-subsets.
    # B = T ∪ S where S is a (k-t)-subset of [v] \ T.
    # The t-subsets covered by B are all t-subsets of B.
    
    # For the random phase, we just pick random k-subsets.
    
    # Phase 1: Random construction
    # Estimate m needed: n_t * (1 - 1/C(v-t,k-t))^m < 1
    # C(v-t, k-t) = C(14, 3) = 364
    # m > ln(11628) / (1/364) ≈ 9.36 * 364 ≈ 3407... that's way too many.
    # Wait, that's the expected number if we pick uniformly from all k-subsets.
    # But we only need to cover, and random might not be efficient enough.
    # Let's just do a smaller random phase and then greedy.
    
    # Actually, for v=19,k=8,t=5, the known best is 335.
    # Let's try: random phase with ~300 blocks, then greedy repair.
    
    random.seed(42)
    covered = [0] * n_t  # count of how many blocks cover each t-subset
    blocks = []
    block_set = set()
    
    # Phase 1: Random blocks
    # Try to get close to the target with random
    target_random = 250
    for _ in range(target_random):
        if time.time() > deadline - 30:
            break
        b = tuple(sorted(random.sample(range(v), k)))
        if b in block_set:
            continue
        block_set.add(b)
        blocks.append(list(b))
        # Update coverage
        for sub in itertools.combinations(b, t):
            covered[t_sub_index[sub]] += 1
    
    # Phase 2: Greedy repair
    while True:
        if time.time() > deadline - 10:
            break
        # Find uncovered t-subsets
        uncovered = [i for i in range(n_t) if covered[i] == 0]
        if not uncovered:
            break
        
        # Pick the most "constrained" uncovered subset? Or just the first?
        # Let's pick one at random from uncovered for diversity, or the first.
        # Actually, for greedy, we should pick the one that is hardest to cover.
        # But for simplicity, let's just iterate.
        
        # For each uncovered t-subset, find the best k-subset to add.
        # To save time, we'll pick a random uncovered subset and find the best block for it.
        idx = random.choice(uncovered)
        T = all_t_subs[idx]
        T_set = set(T)
        
        # Find best k-subset containing T
        remaining = [x for x in range(v) if x not in T_set]
        best_gain = -1
        best_S = None
        # We need to choose (k-t) elements from remaining (v-t elements)
        # C(14, 3) = 364 possibilities. For each, compute gain.
        for S in itertools.combinations(remaining, k - t):
            B = tuple(sorted(T + S))
            # Compute gain: how many currently uncovered t-subsets does B cover?
            gain = 0
            for sub in itertools.combinations(B, t):
                i = t_sub_index[sub]
                if covered[i] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_S = S
        
        if best_S is not None and best_gain > 0:
            B = tuple(sorted(T + best_S))
            if B not in block_set:
                block_set.add(B)
                blocks.append(list(B))
                for sub in itertools.combinations(B, t):
                    covered[t_sub_index[sub]] += 1
            else:
                # Already have this block, find another
                # Just add a random block containing T
                S = tuple(sorted(random.sample(remaining, k - t)))
                B = tuple(sorted(T + S))
                if B not in block_set:
                    block_set.add(B)
                    blocks.append(list(B))
                    for sub in itertools.combinations(B, t):
                        covered[t_sub_index[sub]] += 1
        else:
            # No improvement possible? Shouldn't happen since we can always add a block.
            # Add a random block containing T
            S = tuple(sorted(random.sample(remaining, k - t)))
            B = tuple(sorted(T + S))
            if B not in block_set:
                block_set.add(B)
                blocks.append(list(B))
                for sub in itertools.combinations(B, t):
                    covered[t_sub_index[sub]] += 1
    
    # Phase 3: Local search - remove redundant blocks
    # Try to remove each block if all its t-subsets are still covered.
    removed_any = True
    while removed_any and time.time() < deadline - 5:
        removed_any = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            b = blocks[i]
            # Check if removing this block leaves all its t-subsets covered
            can_remove = True
            for sub in itertools.combinations(b, t):
                if covered[t_sub_index[sub]] == 1:
                    can_remove = False
                    break
            if can_remove:
                for sub in itertools.combinations(b, t):
                    covered[t_sub_index[sub]] -= 1
                blocks.pop(i)
                removed_any = True
    
    # Verify
    for i in range(n_t):
        if covered[i] == 0:
            # Shouldn't happen, but if it does, add a block
            T = all_t_subs[i]
            T_set = set(T)
            remaining = [x for x in range(v) if x not in T_set]
            S = tuple(sorted(random.sample(remaining, k - t)))
            B = tuple(sorted(T + S))
            blocks.append(list(B))
            for sub in itertools.combinations(B, t):
                covered[t_sub_index[sub]] += 1
    
    return blocks
