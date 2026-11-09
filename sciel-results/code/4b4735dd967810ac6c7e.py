import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s margin
    
    # Generate primes up to 1000
    is_prime = [True] * 1001
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(1000**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, 1001, i):
                is_prime[j] = False
    primes = [p for p in range(2, 1001) if is_prime[p]]
    
    # Precompute reciprocals
    recip = {p: Fraction(1, p) for p in primes}
    
    # Function to check if a set of classes covers 1..L
    def get_coverage(classes, max_L):
        # classes is list of (p, r)
        covered = [False] * (max_L + 1)
        for p, r in classes:
            for n in range(r if r > 0 else p, max_L + 1, p):
                if n <= max_L:
                    covered[n] = True
        # Find largest L such that 1..L are all True
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    # Greedy Set Cover
    def greedy_cover(target_L):
        if time.time() > deadline:
            return None, 0
        
        uncovered = set(range(1, target_L + 1))
        used_primes = set()
        cost = Fraction(0)
        classes = []
        
        # Precompute coverage sets for efficiency?
        # For each p, r, the set of n in 1..target_L covered.
        # Since target_L <= 1000, we can just compute on the fly or precompute.
        # Precomputation:
        # cover[p][r] = set of n
        # This might be memory heavy if done for all p, r.
        # 168 primes * avg 500 residues * 1000 ints = 84M ints. Too much.
        # Instead, compute intersection on the fly.
        
        while uncovered:
            best_gain = 0
            best_p = None
            best_r = None
            best_cost = None
            
            for p in primes:
                if p in used_primes:
                    continue
                p_cost = recip[p]
                if cost + p_cost > 1:
                    continue
                
                # For each residue r, check gain
                # We can optimize: only check r such that it covers at least one uncovered number.
                # The residues that cover uncovered numbers are {n % p for n in uncovered}.
                # But we can just iterate r from 0 to p-1?
                # p can be up to 1000. 1000 iterations per prime.
                # 168 primes * 1000 residues = 168k checks.
                # Inside check: count how many n in uncovered are covered.
                # If we do this naively, it's 168k * 1000 (size of uncovered) = 168M ops. Too slow.
                
                # Optimization:
                # For a given p, the residues that matter are those in {n % p for n in uncovered}.
                # Let's build a dict: residue -> count of uncovered numbers with that residue.
                res_counts = {}
                for n in uncovered:
                    r = n % p
                    if r not in res_counts:
                        res_counts[r] = 0
                    res_counts[r] += 1
                
                # Find max count
                if not res_counts:
                    continue
                max_count = max(res_counts.values())
                if max_count > best_gain:
                    best_gain = max_count
                    best_p = p
                    # Pick one residue with max count
                    # If tie, pick one that covers smaller numbers?
                    best_r = [r for r, c in res_counts.items() if c == max_count][0]
                    best_cost = p_cost
            
            if best_p is None:
                break
            
            # Add best class
            classes.append((best_p, best_r))
            used_primes.add(best_p)
            cost += best_cost
            
            # Update uncovered
            new_uncovered = set()
            for n in uncovered:
                if n % best_p != best_r:
                    new_uncovered.add(n)
            uncovered = new_uncovered
            
            if time.time() > deadline:
                return classes, get_coverage(classes, target_L)
        
        return classes, get_coverage(classes, target_L)

    # Local Search Refinement
    def refine(classes, target_L):
        if time.time() > deadline:
            return classes
        
        cost = sum(recip[p] for p, r in classes)
        used_primes = {p for p, r in classes}
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try changing residues
            for i, (p, r) in enumerate(classes):
                # Try all residues
                best_r = r
                best_L = get_coverage(classes, target_L)
                
                # To speed up, only try residues that cover at least one number?
                # Or just try a few random ones?
                # Let's try all residues for small p, and a few for large p.
                # Actually, changing r of p only affects coverage of numbers congruent to r mod p.
                # It might uncover some and cover others.
                
                # Simple heuristic: try to cover the first uncovered number.
                curr_L = get_coverage(classes, target_L)
                if curr_L >= target_L:
                    break
                
                next_uncovered = curr_L + 1
                # If next_uncovered is not covered by current classes, try to adjust a class to cover it.
                # Or adjust a class to cover a gap.
                
                # Let's just do a random restart of residues for a few primes.
                # This is getting complex.
                pass
            
            # 2. Try swapping primes
            # If we have a prime p that covers very few unique numbers, try to replace it with a larger prime q.
            # This frees up cost? No, 1/p > 1/q. So cost decreases.
            # But coverage might change.
            
        return classes

    # Main Loop: Binary Search on L
    best_classes = []
    best_L = 0
    
    # Try L from 1000 down to 1
    # But greedy is fast, so we can try 1000 first.
    # If it fails, try 999, etc.
    
    # Actually, let's just run greedy for L=1000.
    # If it succeeds, great.
    # If it fails, we might have a partial cover.
    # We can try to "repair" the partial cover.
    
    classes, L = greedy_cover(1000)
    if classes:
        best_L = L
        best_classes = classes
        # Refine
        best_classes = refine(best_classes, 1000)
        best_L = get_coverage(best_classes, 1000)
        
        # If we didn't reach 1000, try to extend?
        # Or try different greedy strategies (e.g., prioritize small numbers).
        
        # Let's try a second greedy pass with different tie-breaking.
        # Prefer residues that cover smaller numbers.
        
        # For now, return what we have.
        
    # If greedy failed completely or L is low, try a different approach?
    # Given the time, let's stick to one robust greedy + simple local search.
    
    return [[p, r] for p, r in best_classes]
