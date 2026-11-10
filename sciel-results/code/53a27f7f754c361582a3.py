import time
import random
import bisect

def construct():
    start_time = time.time()
    limit = 50.0  # Safety margin
    
    # Pre-compute primes
    primes = [x for x in range(2, 129) if all(x % i for i in range(2, int(x**0.5) + 1))]
    
    def is_square_free(n):
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True
    
    def count_distinct_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Categorize numbers
    all_nums = list(range(2, 129))
    primes_set = set(primes)
    non_sqfree = [x for x in all_nums if x not in primes_set and not is_square_free(x)]
    sqfree_comp = [x for x in all_nums if x not in primes_set and is_square_free(x)]
    
    def get_heuristic_score(x):
        if not is_square_free(x):
            return (0, 0)
        return (1, -count_distinct_prime_factors(x))

    candidates = non_sqfree + sqfree_comp
    candidates.sort(key=lambda x: (get_heuristic_score(x)[0], get_heuristic_score(x)[1], x))

    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def is_valid_insertion(S, x):
        """Check if inserting x into S (at sorted position) keeps all block products distinct."""
        idx = bisect.bisect_left(S, x)
        # Only need to check blocks that include x
        # These are: [S[i..idx-1], x], [x, S[idx..j]], [S[i..idx-1], x, S[idx..j]]
        # Actually, any block containing x: starts at i <= idx, ends at j >= idx
        # Block product = (prod S[i..idx-1]) * x * (prod S[idx..j])
        # We need to check these don't collide with existing products or each other
        
        n = len(S)
        # Precompute prefix products for S
        # prefix[i] = S[0]*S[1]*...*S[i-1], prefix[0]=1
        # suffix[i] = S[i]*S[i+1]*...*S[n-1], suffix[n]=1
        prefix = [1] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] * S[i]
        suffix = [1] * (n + 1)
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1] * S[i]
        
        # Existing products set
        existing = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= S[j]
                existing.add(prod)
        
        # New products involving x
        new_prods = set()
        for i in range(idx + 1):  # start from i to idx (0-indexed, so i=0..idx means S[i..idx-1])
            left_prod = prefix[idx] // prefix[i] if i < idx else 1
            for j in range(idx, n):  # end from idx to n-1
                right_prod = suffix[idx] // suffix[j+1] if j >= idx else 1
                # Wait, let me recompute more carefully
                # Block from i to j in new_S where new_S has x at position idx
                # If i <= idx <= j:
                #   product = (S[i]*...*S[idx-1]) * x * (S[idx]*...*S[j])
                # left part: S[i] to S[idx-1], product = prefix[idx] // prefix[i]
                # right part: S[idx] to S[j], product = suffix[idx] // suffix[j+1]
                pass
        
        # Let me redo this more carefully
        new_prods = set()
        for i in range(idx + 1):  # i is start index in new_S, 0..idx
            # left product: S[i]*...*S[idx-1] (empty if i==idx)
            if i < idx:
                left_prod = prefix[idx] // prefix[i]
            else:
                left_prod = 1
            for j in range(idx, n + 1):  # j is end index in new_S, idx..n
                # right product: S[idx]*...*S[j-1] (empty if j==idx)
                if j > idx:
                    right_prod = suffix[idx] // suffix[j]
                else:
                    right_prod = 1
                block_prod = left_prod * x * right_prod
                if block_prod in existing or block_prod in new_prods:
                    return False
                new_prods.add(block_prod)
        
        return True

    def is_valid_swap(S, x_remove, x_add):
        """Check if removing x_remove and inserting x_add keeps validity."""
        S_without = [v for v in S if v != x_remove]
        idx = bisect.bisect_left(S_without, x_add)
        new_S = S_without[:idx] + [x_add] + S_without[idx:]
        return is_valid(new_S)

    # 1. Initialize with Primes
    S = primes[:]
    S.sort()
    
    best_S = S[:]
    best_len = len(S)
    
    # 2. Greedy Insertion of Candidates
    for x in candidates:
        if time.time() - start_time > limit:
            break
        if is_valid_insertion(S, x):
            idx = bisect.bisect_left(S, x)
            S.insert(idx, x)
            if len(S) > best_len:
                best_S = S[:]
                best_len = len(S)
                
    # 3. Local Search with multiple move types
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < limit and no_improve_count < max_no_improve:
        improved = False
        
        # Get candidates for removal (non-primes in S, prioritizing those with more prime factors)
        removable = [x for x in S if x not in primes_set]
        removable.sort(key=lambda x: (-count_distinct_prime_factors(x), x))
        
        missing = [x for x in all_nums if x not in S]
        missing.sort(key=lambda x: (get_heuristic_score(x)[0], get_heuristic_score(x)[1], x))
        
        # Try 1-out/1-in swaps
        if removable and missing:
            random.shuffle(removable)
            for y in removable[:10]:  # Try up to 10 removals
                if time.time() - start_time > limit:
                    break
                for x in missing[:10]:  # Try up to 10 additions
                    if is_valid_swap(S, y, x):
                        S_without_y = [v for v in S if v != y]
                        idx = bisect.bisect_left(S_without_y, x)
                        new_S = S_without_y[:idx] + [x] + S_without_y[idx:]
                        if is_valid(new_S):
                            S = new_S
                            if len(S) > best_len:
                                best_S = S[:]
                                best_len = len(S)
                            improved = True
                            break
                if improved:
                    break
        
        # Try 1-out/2-in moves
        if not improved and removable and len(missing) >= 2:
            y = random.choice(removable)
            S_without_y = [v for v in S if v != y]
            
            added_count = 0
            new_S = S_without_y[:]
            for x in missing:
                if added_count >= 2:
                    break
                if is_valid_insertion(new_S, x):
                    idx = bisect.bisect_left(new_S, x)
                    new_S.insert(idx, x)
                    added_count += 1
            
            if added_count >= 2 and is_valid(new_S):
                S = new_S
                if len(S) > best_len:
                    best_S = S[:]
                    best_len = len(S)
                improved = True
        
        # Try 2-out/3-in moves
        if not improved and len(removable) >= 2 and len(missing) >= 3:
            y1, y2 = random.sample(removable, 2)
            S_without = [v for v in S if v != y1 and v != y2]
            
            added_count = 0
            new_S = S_without[:]
            for x in missing:
                if added_count >= 3:
                    break
                if is_valid_insertion(new_S, x):
                    idx = bisect.bisect_left(new_S, x)
                    new_S.insert(idx, x)
                    added_count += 1
            
            if added_count >= 3 and is_valid(new_S):
                S = new_S
                if len(S) > best_len:
                    best_S = S[:]
                    best_len = len(S)
                improved = True
        
        # Try 1-in insertion
        if not improved and missing:
            random.shuffle(missing)
            for x in missing[:15]:
                if time.time() - start_time > limit:
                    break
                if is_valid_insertion(S, x):
                    idx = bisect.bisect_left(S, x)
                    S.insert(idx, x)
                    if len(S) > best_len:
                        best_S = S[:]
                        best_len = len(S)
                    improved = True
                    break
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0

    return best_S
