"""Search-based construction of dense increasing sequences with unique block products."""
import time
import random
import itertools

def construct():
    deadline = time.time() + 88.0  # safety margin before 90s kill
    best_seq = []
    best_len = 0

    def get_all_prods(seq):
        """Get set of all consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.add(prod)
        return prods

    def can_append(seq, suffix_prods, prods, x):
        """
        Check if x can be appended to seq.
        Returns (bool, new_suffix_prods, new_prods) if valid, else (False, None, None).
        """
        if x in prods:
            return False, None, None
        
        # Compute new suffix products
        # Current suffix_prods: [s_n, s_{n-1}s_n, ..., s_1...s_n]
        # New suffixes ending in x:
        # x
        # s_n * x
        # s_{n-1} * s_n * x
        # ...
        
        new_suffixes = [x]
        # Iterate through existing suffixes in reverse (longest to shortest? No, suffix_prods is usually [last, last-1*last, ...])
        # Let's define suffix_prods[k] = product of seq[-k-1:]
        # So suffix_prods[0] = seq[-1]
        # suffix_prods[1] = seq[-2]*seq[-1]
        
        # New suffixes:
        # 1. x
        # 2. seq[-1] * x
        # 3. seq[-2]*seq[-1] * x
        # ...
        
        # We need to check if any of these are in prods.
        # And if they are distinct from each other.
        
        seen_new = set()
        for p in new_suffixes:
            if p in seen_new:
                return False, None, None
            seen_new.add(p)
            
        for sp in suffix_prods:
            p = sp * x
            if p in seen_new:
                return False, None, None
            seen_new.add(p)
            if p in prods:
                return False, None, None
        
        # If we get here, it's valid.
        # Construct new_suffixes list for next step
        # new_suffixes[0] = x
        # new_suffixes[1] = suffix_prods[0] * x
        # ...
        new_suffixes_list = [x] + [sp * x for sp in suffix_prods]
        
        # Update prods
        new_prods = prods.union(new_suffixes_list)
        
        return True, new_suffixes_list, new_prods

    # Strategy 1: Efficient Greedy with incremental suffix products
    baseline = []
    baseline_prods = set()
    baseline_suffix = []
    
    for x in range(2, 129):
        ok, new_suffix, new_prods = can_append(baseline, baseline_suffix, baseline_prods, x)
        if ok:
            baseline.append(x)
            baseline_prods = new_prods
            baseline_suffix = new_suffix
            
    if len(baseline) > best_len:
        best_len = len(baseline)
        best_seq = baseline[:]

    # Strategy 2: Beam Search with incremental state
    # State: (seq, suffix_prods, prods_set)
    # To save memory, we might not store full prods_set for every beam item if it's huge, 
    # but for N~30-40, the set size is N(N+1)/2 ~ 600-800. This is manageable.
    
    beam = [(baseline, baseline_suffix, baseline_prods)]
    
    # We want to extend these sequences.
    # Since we are building increasing sequences, the next element must be > last.
    
    # To allow for different structures, we can also start from scratch with different heuristics.
    # But first, let's try to extend the baseline.
    
    # Actually, a pure greedy might get stuck. Let's use a Beam Search that keeps top K sequences.
    # We will perform "steps" where we try to add the next best candidate.
    
    # A better approach: Randomized Greedy with many restarts, using the fast can_append.
    random.seed(42)
    
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
            
        # Randomize the order of candidates slightly or use a different subset
        # We must maintain strictly increasing order, so we can't just shuffle all 2..128.
        # We can shuffle the *decision* of which candidate to pick from the valid ones > last.
        
        seq = []
        suffix = []
        prods = set()
        
        # Start with a random small number? No, must be increasing.
        # Let's pick a random starting point from 2..10?
        # Or just iterate 2..128 but with randomization in "tie-breaking" or "skipping".
        
        # Let's try a "Lookahead" greedy.
        # At each step, look at all x > last. Pick the one that allows the most future extensions?
        # Too expensive.
        
        # Let's try random restarts with random "skip" probabilities.
        # Or, simply shuffle the list of candidates 2..128? No, that breaks increasing order.
        
        # We can generate a random permutation of 2..128 and then filter to keep increasing?
        # That's just a random subset.
        
        # Let's try: Pick a random subset of candidates, sort them, and greedily add.
        if attempt < 100:
            # Dense subsets
            candidates = list(range(2, 129))
        else:
            # Random subsets
            k = random.randint(20, 60)
            candidates = random.sample(range(2, 129), k)
            candidates.sort()
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            ok, new_suffix, new_prods = can_append(seq, suffix, prods, x)
            if ok:
                seq.append(x)
                suffix = new_suffix
                prods = new_prods
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Strategy 3: Local Search on best_seq
    # Try to insert missing numbers or swap adjacent ones.
    if best_seq:
        best_prods = get_all_prods(best_seq)
        best_suffix = [1] # dummy
        # Recompute suffix for best_seq
        s = best_seq
        suf = []
        prod = 1
        for i in range(len(s)-1, -1, -1):
            prod *= s[i]
            suf.append(prod)
        best_suffix = suf
        
        in_seq = set(best_seq)
        missing = [x for x in range(2, 129) if x not in in_seq]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in missing[:]:
                if time.time() > deadline - 2:
                    break
                for pos in range(len(best_seq) + 1):
                    if time.time() > deadline - 2:
                        break
                    if pos > 0 and x <= best_seq[pos - 1]:
                        continue
                    if pos < len(best_seq) and x >= best_seq[pos]:
                        continue
                    
                    # Check validity of insertion
                    test_seq = best_seq[:pos] + [x] + best_seq[pos:]
                    # We can use can_append logic but it's for appending.
                    # For insertion, we need to check all blocks involving x.
                    # This is O(N^2) per insertion check.
                    # Given N~40, N^2=1600. 127 missing * 40 pos * 1600 ops = ~80M ops.
                    # In Python, this might be slow.
                    
                    # Let's just do a full validation for the candidate insertion.
                    valid = True
                    n = len(test_seq)
                    prods_check = set()
                    for i in range(n):
                        prod = 1
                        for j in range(i, n):
                            prod *= test_seq[j]
                            if prod in prods_check:
                                valid = False
                                break
                            prods_check.add(prod)
                        if not valid:
                            break
                    
                    if valid:
                        best_seq = test_seq
                        best_prods = prods_check
                        # Recompute suffix
                        s = best_seq
                        suf = []
                        prod = 1
                        for i in range(len(s)-1, -1, -1):
                            prod *= s[i]
                            suf.append(prod)
                        best_suffix = suf
                        best_len = len(best_seq)
                        if x in missing:
                            missing.remove(x)
                        improved = True
                        break
                if improved:
                    break

    return best_seq
