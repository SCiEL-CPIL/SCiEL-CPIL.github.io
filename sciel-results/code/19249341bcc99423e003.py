import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Precompute masks for all triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        
    # Precompute pair masks for linearity check
    # Pair (i, j) with i < j maps to bit index i*n + j
    # Total pairs: 16*15/2 = 120. This fits in a Python int.
    tpair = [0] * num_triples
    for i, (a, b, c) in enumerate(all_triples):
        pm = 0
        for x, y in ((a, b), (a, c), (b, c)):
            if x > y:
                x, y = y, x
            pm |= 1 << (x * n + y)
        tpair[i] = pm
        
    # Precompute popcount for independence check optimization?
    # Actually, for independence, we check if a set S contains any triple.
    # A set S is independent if no triple is a subset of S.
    # We can precompute for each subset mask, the number of triples it contains?
    # Or just iterate.
    
    def compute_alpha(triple_indices):
        """
        Compute the independence number of the hypergraph defined by triple_indices.
        Returns the size of the largest subset of points containing no triple.
        """
        if not triple_indices:
            return n
            
        # Build a boolean array of size 2^n indicating which subsets are "covered" (contain a triple)
        # Actually, we want to find the largest subset S such that no triple is a subset of S.
        # Let's use the method from inspiration 1:
        # covered[mask] = True if mask contains at least one triple.
        # Then we look for the largest mask that is NOT covered.
        
        covered = np.zeros(1 << n, dtype=bool)
        
        # For each triple, mark all supersets as covered?
        # That is expensive: 560 triples * 2^(16-3) supersets = 560 * 8192 ~ 4.5M operations.
        # This is fast enough in NumPy if vectorized, but let's see.
        
        # Alternative:
        # For each subset S (from 2^n - 1 down to 0), check if it's independent.
        # But 2^16 = 65536.
        # For each S, check if any triple is a subset of S.
        # If we precompute for each triple mask T, the list of supersets? No.
        
        # Let's stick to the inspiration 1 approach which seemed efficient.
        # "covered |= (subsets & t) == t"
        # subsets is an array [0, 1, ..., 2^n - 1]
        # This creates a boolean array for each triple.
        # 560 iterations of vectorized operations on 65536 elements.
        # This is very fast in NumPy.
        
        subsets = np.arange(1 << n, dtype=np.int32)
        
        for idx in triple_indices:
            t_mask = triple_masks[idx]
            # A subset S contains triple T if (S & T) == T
            covered |= (subsets & t_mask) == t_mask
            
        # Find the largest S (by popcount) such that not covered[S]
        # We need popcount for all masks.
        # Precompute popcount once outside?
        # Since this function is called frequently, let's precompute popcount globally.
        
        # Wait, popcount depends on n. n is fixed at 16.
        # Let's precompute popcount array globally to avoid doing it every time.
        
        # For now, let's compute it inside or pass it.
        # Actually, I'll define a global popcount array in the outer scope.
        
        # Let's use the global popcount array 'pop_arr'
        
        # We want max(pop_arr[~covered])
        # ~covered is a boolean mask.
        # pop_arr[~covered] gives popcounts of independent sets.
        # We want the max.
        
        # If all are covered, max is -1? No, empty set is always independent.
        # covered[0] is True if any triple is empty (impossible) or if 0 contains a triple.
        # (0 & T) == T is False for T != 0. So 0 is never covered by a non-empty triple.
        # So ~covered will always have at least one True (mask 0).
        
        max_alpha = np.max(pop_arr[~covered])
        return int(max_alpha)

    # Precompute popcount array for all 2^16 masks
    pop_arr = np.zeros(1 << n, dtype=np.int8)
    for i in range(1, 1 << n):
        pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        
    # Re-define compute_alpha to use global pop_arr
    def compute_alpha_fast(triple_indices):
        if not triple_indices:
            return n
        covered = np.zeros(1 << n, dtype=bool)
        subsets = np.arange(1 << n, dtype=np.int32)
        for idx in triple_indices:
            t_mask = triple_masks[idx]
            covered |= (subsets & t_mask) == t_mask
        return int(np.max(pop_arr[~covered]))

    # Greedy packing
    def greedy_packing(order):
        sys_set = set()
        pair_union = 0
        for idx in order:
            if (tpair[idx] & pair_union) == 0:
                sys_set.add(idx)
                pair_union |= tpair[idx]
        return list(sys_set), pair_union

    # Initial solution
    order = list(range(num_triples))
    random.shuffle(order)
    best_packing, _ = greedy_packing(order)
    best_alpha = compute_alpha_fast(best_packing)
    
    # Local Search
    # State: current set of triple indices, pair_union mask
    # We maintain a set for O(1) lookup and a list for iteration.
    
    current_packing = list(best_packing)
    current_pair_union = 0
    for idx in current_packing:
        current_pair_union |= tpair[idx]
    current_alpha = best_alpha
    
    current_set = set(current_packing)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        r = random.random()
        
        if r < 0.4:
            # Try to add a random valid triple
            cand = random.randrange(num_triples)
            if cand not in current_set and (tpair[cand] & current_pair_union) == 0:
                # Check if adding it improves or maintains alpha
                new_packing = current_packing + [cand]
                new_alpha = compute_alpha_fast(new_packing)
                if new_alpha <= current_alpha:
                    current_packing = new_packing
                    current_set.add(cand)
                    current_pair_union |= tpair[cand]
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_packing = list(current_packing)
                        
        elif r < 0.8:
            # Swap: remove one, add one
            if not current_packing:
                continue
            rem = random.choice(current_packing)
            cand = random.randrange(num_triples)
            if cand in current_set or cand == rem:
                continue
            
            # Remove rem
            new_set = current_set.copy()
            new_set.discard(rem)
            new_set.add(cand)
            
            # Check linearity of new_set
            # We can recompute pair_union from scratch for the new set, or update it.
            # Updating is hard because we don't know if other triples share pairs with rem.
            # So we must check linearity against all others.
            # Since n_triples is small (~30-40), checking all pairs is O(k^2) which is fine.
            # But we can also just recompute the union and check if it matches the sum of bits?
            # No, pair_union is a bitmask. If we remove rem, we can't just clear its bits if others share them.
            # So we must verify linearity.
            
            ok = True
            # Simple check: for each pair in cand, ensure no other triple in new_set has that pair.
            # This is what tpair does.
            # Let's just build the new pair_union and check if it's valid?
            # No, building pair_union doesn't tell us if there's a conflict.
            # We need to check if any two triples in new_set share a pair.
            
            # Efficient check:
            # The new set is current_set - {rem} + {cand}.
            # We know current_set was linear.
            # So conflicts can only involve cand.
            # Check if cand shares a pair with any triple in (current_set - {rem}).
            
            conflict = False
            for other in current_set:
                if other == rem:
                    continue
                if (tpair[cand] & tpair[other]) != 0:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # If no conflict, the new set is linear.
            new_packing = list(new_set)
            new_alpha = compute_alpha_fast(new_packing)
            
            if new_alpha <= current_alpha:
                current_packing = new_packing
                current_set = new_set
                # Recompute pair_union
                current_pair_union = 0
                for idx in current_packing:
                    current_pair_union |= tpair[idx]
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_packing = list(current_packing)
                    
        else:
            # Remove one triple (to allow adding two later? or just to explore)
            # Removing a triple increases alpha or keeps it same.
            # This is usually bad for minimizing alpha, but might allow a better swap sequence.
            # Let's try removing a triple that is "redundant" or part of a large independent set.
            if len(current_packing) > 5:
                # Pick a triple to remove
                # Heuristic: remove one that is in many independent sets?
                # Or just random.
                rem = random.choice(current_packing)
                new_set = current_set.copy()
                new_set.discard(rem)
                
                # Check linearity: subset of a linear set is linear. So always valid.
                new_packing = list(new_set)
                new_alpha = compute_alpha_fast(new_packing)
                
                # We only accept if alpha doesn't increase?
                # Actually, removing a triple can only increase or keep alpha.
                # So new_alpha >= current_alpha.
                # We want to minimize alpha. So this move is only useful if it enables a future improvement.
                # In a simple hill-climbing, we might reject this.
                # But in simulated annealing or random walk, we might accept.
                # Let's accept with low probability or only if it doesn't increase alpha too much.
                # Actually, let's just reject if alpha increases, unless we are stuck.
                if new_alpha == current_alpha:
                    current_packing = new_packing
                    current_set = new_set
                    current_pair_union = 0
                    for idx in current_packing:
                        current_pair_union |= tpair[idx]
                    # current_alpha stays same
                # If alpha increased, we revert (do nothing)
                
        # Early stopping if we find a very good solution
        if best_alpha <= 5:
            # Keep searching for a bit more to be sure, or break if time is low
            if time.time() > deadline - 10:
                break

    # Convert best_packing (list of indices) to list of triples
    out = [list(all_triples[i]) for i in best_packing]
    return out

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Num triples: {len(res)}")
    # Verify linearity
    masks = set()
    for t in res:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        if m in masks:
            print("Duplicate triple found!")
        masks.add(m)
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            mi = (1 << res[i][0]) | (1 << res[i][1]) | (1 << res[i][2])
            mj = (1 << res[j][0]) | (1 << res[j][1]) | (1 << res[j][2])
            inter = mi & mj
            count = bin(inter).count('1')
            if count > 1:
                print(f"Non-linear: {res[i]} and {res[j]} share {count} points")
    
    # Compute alpha
    def check_alpha(triples):
        n = 16
        pop_arr = np.zeros(1 << n, dtype=np.int8)
        for i in range(1, 1 << n):
            pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        covered = np.zeros(1 << n, dtype=bool)
        subsets = np.arange(1 << n, dtype=np.int32)
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            covered |= (subsets & m) == m
        return int(np.max(pop_arr[~covered]))
    
    print(f"Alpha: {check_alpha(res)}")
