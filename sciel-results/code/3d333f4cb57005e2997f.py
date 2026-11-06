import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def get_primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def is_valid(seq):
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def is_valid_fast(seq):
        """Faster validity check with early exit"""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
                # Early exit if product gets too large (won't help distinguish)
                if prod > 10**18:
                    break
        return True
    
    primes = get_primes(128)
    all_nums = list(range(2, 129))
    composites = [x for x in all_nums if x not in primes]
    
    # Precompute which numbers conflict with which
    # Two numbers conflict if they can't both be in a valid sequence
    # This is complex, so we'll use a simpler heuristic:
    # Numbers with many common factors are more likely to conflict
    
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Score numbers by how "conflict-prone" they are
    # Numbers with more prime factors tend to create more product collisions
    def conflict_score(n):
        return num_prime_factors(n)
    
    # Sort composites by conflict score (lower first to try less conflicting ones)
    composites_sorted = sorted(composites, key=lambda x: (conflict_score(x), x))
    
    best_seq = primes[:]
    best_len = len(primes)
    
    # Greedy Add with smarter ordering
    def greedy_add(seq, order=None):
        if order is None:
            order = composites_sorted
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for c in order:
                if c in seq:
                    continue
                new_seq = sorted(seq + [c])
                if is_valid_fast(new_seq):
                    seq = new_seq
                    improved = True
                    break
        return seq
    
    # Local Search with better moves
    def local_search(seq, max_iters=1000):
        seq_set = set(seq)
        candidates = [c for c in composites if c not in seq_set]
        
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Try to add
            for c in candidates:
                if c in seq_set:
                    continue
                new_seq = sorted(seq + [c])
                if is_valid_fast(new_seq):
                    seq = new_seq
                    seq_set.add(c)
                    candidates.remove(c)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add another
            found_swap = False
            for a in list(seq_set):
                for c in candidates:
                    if a == c:
                        continue
                    new_seq = sorted((seq_set - {a}) | {c})
                    if is_valid_fast(new_seq):
                        # Check if this swap enables further additions
                        new_seq_set = set(new_seq)
                        new_candidates = [x for x in candidates if x != c and x not in new_seq_set]
                        can_add = False
                        for d in new_candidates[:10]:  # Check a few
                            test_seq = sorted(new_seq + [d])
                            if is_valid_fast(test_seq):
                                can_add = True
                                break
                        
                        if can_add or random.random() < 0.3:
                            seq = new_seq
                            seq_set = new_seq_set
                            candidates = [x for x in candidates if x != c and x not in seq_set]
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            
            if improved:
                continue
            
            # Try remove (kick) to escape local optimum
            if len(seq) > 3:
                a = random.choice(list(seq_set))
                new_seq = sorted(seq_set - {a})
                if is_valid_fast(new_seq):
                    seq = new_seq
                    seq_set.remove(a)
                    candidates.append(a)
                    improved = True
            
            if not improved:
                break
        
        return seq
    
    # Main Loop
    current_seq = primes[:]
    current_seq = greedy_add(current_seq)
    if len(current_seq) > best_len:
        best_seq = current_seq
        best_len = len(current_seq)
    
    # Run local search from this starting point
    current_seq = local_search(current_seq)
    if len(current_seq) > best_len:
        best_seq = current_seq
        best_len = len(current_seq)
    
    while time.time() < deadline - 2:
        # Try to improve current_seq
        improved = False
        
        # 1. Try Greedy Add with different orderings
        for _ in range(3):
            random.shuffle(composites_sorted)
            new_seq = greedy_add(current_seq, composites_sorted)
            if len(new_seq) > len(current_seq):
                current_seq = new_seq
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(current_seq)
                improved = True
                break
        
        if not improved:
            # 2. Try Local Search
            new_seq = local_search(current_seq, max_iters=200)
            if len(new_seq) > len(current_seq):
                current_seq = new_seq
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(current_seq)
                improved = True
        
        if not improved:
            # 3. Try Random Swap
            seq_set = set(current_seq)
            candidates = [c for c in composites if c not in seq_set]
            if candidates:
                a = random.choice(current_seq)
                c = random.choice(candidates)
                if a != c:
                    new_seq = sorted((seq_set - {a}) | {c})
                    if is_valid_fast(new_seq):
                        current_seq = new_seq
                        improved = True
        
        if not improved:
            # 4. Try Random Remove (Kick)
            if len(current_seq) > 3:
                a = random.choice(current_seq)
                new_seq = sorted(set(current_seq) - {a})
                if is_valid_fast(new_seq):
                    current_seq = new_seq
                    improved = True
        
        if not improved:
            # 5. Restart with different random greedy
            random.shuffle(composites_sorted)
            current_seq = primes[:]
            improved_greedy = True
            while improved_greedy and time.time() < deadline - 2:
                improved_greedy = False
                for c in composites_sorted:
                    if c in current_seq:
                        continue
                    new_seq = sorted(current_seq + [c])
                    if is_valid_fast(new_seq):
                        current_seq = new_seq
                        improved_greedy = True
                        break
            
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(current_seq)
            
            # Run local search from this restart
            current_seq = local_search(current_seq, max_iters=100)
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(current_seq)
    
    return best_seq
