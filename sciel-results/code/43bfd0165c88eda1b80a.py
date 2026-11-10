import random
import time

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    all_masks = [1 << i for i in range(n)]
    # Actually, just range(2**n) is fine for n=12 (4096)
    
    # 1. Greedy Initialization
    family = set()
    # Start with empty set
    family.add(0)
    
    # Candidate pool: all masks
    candidates = list(range(1, 1 << n))
    random.shuffle(candidates)
    
    # Precompute subset check function for speed
    def is_subset(a, b):
        return (a & b) == a and a != b
        
    # Greedy fill
    for _ in range(m - 1):
        best_c = None
        best_gain = -1
        # To speed up, we might not check all 4096 every time?
        # But 4096 * 40 * 40 = 6.5M ops. It's okay.
        # Let's check all candidates.
        for c in candidates:
            if c in family:
                continue
            gain = 0
            for f in family:
                if (c & f) == c: # c subset of f
                    gain += 1
                elif (c & f) == f: # f subset of c
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c is not None:
            family.add(best_c)
        else:
            # Fallback: add random
            random_c = random.choice(candidates)
            family.add(random_c)
            
    # Convert to list for local search
    fam_list = list(family)
    
    # 2. Local Search / Simulated Annealing
    # Objective: Maximize comparable pairs
    def count_comparable(fam):
        # fam is a set of masks
        # Returns number of unordered pairs {a,b} with a subset b or b subset a
        count = 0
        fam_list_local = list(fam)
        for i in range(len(fam_list_local)):
            for j in range(i+1, len(fam_list_local)):
                a = fam_list_local[i]
                b = fam_list_local[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count
        
    # Initial score
    current_score = count_comparable(fam_list)
    
    # Optimization: Calculate gain of adding/removing a set from a specific family state
    # To make local search fast, we compute delta directly.
    # Delta when replacing A with B in family F:
    # New Score = Old Score - pairs(A, F\A) + pairs(B, F\A)
    # Note: pairs(A, F\A) is the number of sets in F\A comparable to A.
    
    def get_comparable_count(mask, fam_set):
        """Returns count of sets in fam_set (excluding mask if present? No, just in fam_set) comparable to mask."""
        cnt = 0
        for f in fam_set:
            if f == mask:
                continue
            if (mask & f) == mask or (mask & f) == f:
                cnt += 1
        return cnt

    # We will perform local search
    # Time limit 50s to be safe
    start_time = time.time()
    time_limit = 50.0
    
    # Temperature for SA
    T = 1.0
    cool = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick random index
        i = random.randint(0, m - 1)
        old_mask = fam_list[i]
        
        # Generate neighbor
        # Strategy: Flip 1-3 random bits
        new_mask = old_mask
        num_flips = random.randint(1, 3)
        for _ in range(num_flips):
            bit = random.randint(0, n - 1)
            new_mask ^= (1 << bit)
            
        # Check distinctness
        if new_mask in family:
            continue
            
        # Calculate Delta
        # Remove old_mask from consideration for new_mask's count?
        # Actually, we want:
        # Score_new = Score_old - count(old, F\old) + count(new, F\old)
        # Where count(X, Y) is # of pairs {x, y} in Y comparable to X.
        # Note: The set F for the calculation should be the current family minus the element being replaced?
        # Let F_rest = Family \ {old_mask}
        # old_contrib = count(old_mask, F_rest)
        # new_contrib = count(new_mask, F_rest)
        # Delta = new_contrib - old_contrib
        
        # We need to iterate F_rest for both.
        # Since m=40, this is fast.
        
        old_contrib = 0
        new_contrib = 0
        for f in fam_list:
            if f == old_mask:
                continue
            # Check old
            if (old_mask & f) == old_mask or (old_mask & f) == f:
                old_contrib += 1
            # Check new
            if (new_mask & f) == new_mask or (new_mask & f) == f:
                new_contrib += 1
                
        delta = new_contrib - old_contrib
        
        # Accept
        if delta >= 0 or random.random() < pow(2.718, delta / T):
            # Update
            family.remove(old_mask)
            family.add(new_mask)
            fam_list[i] = new_mask
            current_score += delta
            T *= cool
        else:
            T *= cool # Cool down even if rejected? Standard SA cools regardless or only on accept? Usually regardless.
            
    # Convert to lists of ints
    result = []
    for mask in family:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
