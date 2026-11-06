"""Sum-free subset maximizer with local search and deadline guard."""

def construct():
    import time
    from random import Random
    
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True
    
    def count_violations(S):
        """Count number of violating triples (a,b) where a+b in S."""
        Sset = set(S)
        count = 0
        for a in S:
            for b in S:
                if a + b in Sset:
                    count += 1
        return count
    
    def greedy_remove(S, rng, max_iters=5000):
        """Iteratively remove elements that are in the most violating triples."""
        S = list(S)
        Sset = set(S)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not S:
                break
            
            # Count how many violations each element participates in
            viol_count = {x: 0 for x in S}
            for a in S:
                for b in S:
                    if a + b in Sset:
                        viol_count[a] += 1
                        viol_count[b] += 1
                        viol_count[a + b] += 1
            
            max_viol = max(viol_count.values())
            if max_viol == 0:
                break  # sum-free achieved
            
            # Remove element with highest violation count (tie-break randomly)
            candidates = [x for x in S if viol_count[x] == max_viol]
            to_remove = rng.choice(candidates)
            S.remove(to_remove)
            Sset.remove(to_remove)
        
        return S
    
    def local_search(S, rng, max_iters=20000):
        """Try to add back removed elements or swap to improve size while staying sum-free."""
        S = list(S)
        Sset = set(S)
        all_elements = set()
        # We need the full instance to know what we can add back
        # This is handled by the caller passing in the full set
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try adding each element not in S
            improved = False
            for x in list(Sset):  # placeholder, actual logic below
                pass
        
        return S
    
    def optimize_instance(size, seed):
        """Find large sum-free subset of the instance."""
        S_full = instance(size, seed)
        Sset_full = set(S_full)
        rng = Random(seed + 1000)
        
        # Start with greedy removal to get a valid sum-free set
        S = greedy_remove(S_full, rng, max_iters=3000)
        
        # Local search: try to add back elements from S_full that aren't in S
        removed = [x for x in S_full if x not in set(S)]
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            if not removed:
                break
            
            # Try adding a random removed element
            idx = rng.randint(0, len(removed) - 1)
            x = removed[idx]
            S_test = list(S) + [x]
            Sset_test = set(S_test)
            
            # Check if adding x keeps sum-free
            valid = True
            for a in S_test:
                for b in S_test:
                    if a + b in Sset_test and (a == x or b == x or a + b == x):
                        # Check if this is a new violation involving x
                        if a + b == x:
                            # x = a + b, both a,b in S
                            valid = False
                            break
                        elif a == x:
                            # x + b = c in S
                            if b in Sset_test and (x + b) in Sset_test:
                                valid = False
                                break
                        elif b == x:
                            # a + x = c in S
                            if a in Sset_test and (a + x) in Sset_test:
                                valid = False
                                break
                if not valid:
                    break
            
            if valid:
                S = S_test
                Sset = Sset_test
                removed.remove(x)
            else:
                # Try swapping: remove one element from S, add x
                if len(S) > 0:
                    swap_idx = rng.randint(0, len(S) - 1)
                    y = S[swap_idx]
                    S_swap = [z for z in S if z != y] + [x]
                    Sset_swap = set(S_swap)
                    
                    valid_swap = True
                    for a in S_swap:
                        for b in S_swap:
                            if a + b in Sset_swap:
                                valid_swap = False
                                break
                        if not valid_swap:
                            break
                    
                    if valid_swap:
                        S = S_swap
                        Sset = Sset_swap
                        removed.remove(x)
                        removed.append(y)
                        removed = list(set(removed))
        
        return S
    
    # Optimize each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback: use odd numbers as baseline
            S_full = instance(size, seed)
            results.append([x for x in S_full if x % 2])
            continue
        
        S_opt = optimize_instance(size, seed)
        
        # Verify sum-free
        if not is_sum_free(S_opt):
            # Fallback to greedy removal
            S_full = instance(size, seed)
            rng = Random(seed + 1000)
            S_opt = greedy_remove(S_full, rng, max_iters=5000)
            if not is_sum_free(S_opt):
                # Last resort: odd numbers
                S_opt = [x for x in S_full if x % 2]
        
        results.append(S_opt)
    
    return results
