import time
import random
from fractions import Fraction
import math

START_TIME = time.time()
DEADLINE = START_TIME + 140  # Safety margin before 150s kill

def _sieve(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = _sieve(1000)
INV_PRIMES = {p: Fraction(1, p) for p in PRIMES}

def calc_L(classes, max_check=100000):
    """Calculate the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    
    # Use a boolean array for efficiency
    # We only need to check up to a reasonable limit.
    # The task says capped at 10^5, but usually L is small.
    # However, to be safe and fast, we can simulate up to a limit.
    # Let's find the LCM of the primes? No, that's huge.
    # The pattern repeats with period LCM(primes).
    # But we just need to find the first gap.
    
    # Optimization: If the set of classes covers 1..L, 
    # and the pattern has period P = LCM(p_i), then it covers everything 
    # if it covers 1..P? No, it covers all integers congruent to r_i mod p_i.
    # If it covers 1..L, does it cover L+1? Not necessarily.
    # We need to find the largest L such that 1..L are ALL covered.
    # This means we are looking for the first integer n >= 1 that is NOT covered.
    # L = n - 1.
    
    # Since primes are <= 1000, the period is huge. 
    # But typically, if we cover a decent range, we might hit a gap soon.
    # Let's check up to a limit. If we don't find a gap, we return the limit.
    # The task caps the score at 10^5.
    
    covered = [False] * (max_check + 1)
    
    # Mark covered numbers
    for p, r in classes:
        # n % p == r
        # First number >= 1:
        # If r == 0, first is p (since 0 is not in 1..L usually, but 1..L starts at 1)
        # Actually, residue r is 0 <= r < p.
        # Numbers are r, r+p, r+2p...
        # We need n >= 1.
        start = r
        if start == 0:
            start = p
        elif start < 1:
            # r is always >= 0, so if r==0 handled, r>=1 is fine.
            # If r < 1, i.e., r=0, handled.
            pass
        
        # Wait, if r=0, the numbers are p, 2p...
        # If r>0, the numbers are r, r+p... (since r>=1)
        
        for n in range(start, max_check + 1, p):
            covered[n] = True
            
    # Find first gap
    for n in range(1, max_check + 1):
        if not covered[n]:
            return n - 1
            
    return max_check

def greedy_residues(L_target, budget=1, max_iter=100000):
    """
    Greedy construction targeting coverage up to L_target.
    Returns a list of [p, r].
    """
    classes = []
    used_primes = set()
    current_sum = Fraction(0, 1)
    
    # We want to cover 1..L_target.
    # Greedy: At each step, pick a class that covers the most "uncovered" numbers in 1..L_target.
    # But "uncovered" depends on previous choices.
    
    # To speed up, we can track uncovered set.
    uncovered = set(range(1, L_target + 1))
    
    if not uncovered:
        return []
        
    while uncovered and current_sum < 1:
        best_p = None
        best_r = None
        best_count = -1
        best_cost = Fraction(1, 1) # Infinity
        
        # Iterate over available primes
        for p in PRIMES:
            if p in used_primes:
                continue
            if current_sum + INV_PRIMES[p] > budget:
                continue
                
            # For this prime, which residue covers the most uncovered numbers?
            # We can count for each residue r in 0..p-1.
            # But p can be up to 1000. L_target can be large.
            # Optimization: Only consider residues that cover at least one uncovered number?
            # Or just iterate residues and count.
            
            # To be faster, we can sample or use a smarter count.
            # Let's just iterate residues for now, but break early if we find a perfect cover?
            # Actually, for small p, this is fast. For large p, it's slow.
            
            # Heuristic: Try smaller primes first? They have fewer residues.
            # Or try all?
            
            # Let's optimize: 
            # For a given p, the count of uncovered numbers with n % p == r.
            # We can precompute counts for all r for this p?
            # That's O(L_target) per p. Too slow if L_target is large.
            
            # Alternative: 
            # Since we only care about the MAX count, we can stop counting a residue 
            # if it can't beat the current best? No, we don't know the max yet.
            
            # Let's restrict the search to primes p <= L_target? 
            # If p > L_target, each residue covers at most 1 number in 1..L_target.
            # So large primes are inefficient for dense coverage.
            
            # Let's only consider p up to some limit, or sort by p.
            
            # For this specific greedy step, let's evaluate candidates.
            # To keep it fast, we'll limit the number of primes we check per step 
            # or use a heuristic.
            
            # Actually, let's just check all primes, but efficiently.
            # If p is large, the count for any r is small.
            
            # Let's try to find the best (p, r) pair.
            # We'll iterate p. For each p, we find the best r.
            
            # Optimization: If p > L_target, max count is 1.
            # If we already have a best_count > 1, we can skip p > L_target.
            
            # Let's implement a faster count.
            # For a fixed p, count[r] = sum(1 for n in uncovered if n % p == r)
            # This is O(|uncovered|) per p.
            
            # To make this feasible, we should limit the number of iterations or primes.
            # Or, we can use a different strategy:
            # Pick the smallest uncovered number n_min.
            # The new class MUST cover n_min (otherwise n_min remains uncovered, 
            # and since we pick smallest, we are stuck unless we cover it later? 
            # No, if we don't cover it now, we might cover it later. 
            # But greedy usually covers the "hardest" or "most central" point.
            # Covering n_min is a safe bet to reduce the range from the left.
            
            # Let's stick to the "cover n_min" heuristic but choose the best p for n_min.
            # This reduces the search space to one residue per prime (r = n_min % p).
            
            n_min = min(uncovered)
            
            for p in PRIMES:
                if p in used_primes:
                    continue
                if current_sum + INV_PRIMES[p] > budget:
                    continue
                
                r = n_min % p
                # Count how many uncovered numbers are covered by (p, r)
                # This is still O(|uncovered|)
                
                # Optimization: If we just want to cover n_min, we know it's covered.
                # How many others?
                count = 0
                for n in uncovered:
                    if n % p == r:
                        count += 1
                        
                # Efficiency: count / (1/p) = count * p
                # We want to maximize coverage per budget.
                # Or just maximize count?
                # Maximizing count is good for reducing uncovered set size.
                
                if count > best_count:
                    best_count = count
                    best_p = p
                    best_r = r
                    best_cost = INV_PRIMES[p]
                    
                # Early break? If count == len(uncovered), we found a perfect cover for this step.
                # But we can't break the outer loop yet.
                
            if best_p is None:
                break
                
            # Add the class
            classes.append([best_p, best_r])
            used_primes.add(best_p)
            current_sum += best_cost
            
            # Update uncovered
            # Remove all n in uncovered where n % best_p == best_r
            new_uncovered = set()
            for n in uncovered:
                if n % best_p != best_r:
                    new_uncovered.add(n)
            uncovered = new_uncovered
            
            if time.time() > DEADLINE - 5:
                break

    return classes

def local_search(classes, L_target, max_steps=100):
    """
    Try to improve the solution by swapping or adding/removing classes.
    This is a placeholder for now, as greedy might be sufficient for the jump.
    """
    # Implement simple swap: remove one, add one?
    # Or just return classes as is for now.
    return classes

def construct():
    # Strategy:
    # 1. Use greedy to build a solution covering a target L.
    # 2. Start with a moderate target, increase if time permits.
    # 3. The "efficiency" insight suggests dynamic selection.
    #    My greedy picks the class covering n_min with the highest count.
    #    This is a form of efficiency (coverage per step).
    
    # Let's try to achieve L >= 999.
    # The seed achieves 999.
    # We want to beat it.
    
    # Let's run greedy for a few target Ls and pick the best result.
    
    best_classes = None
    best_L = 0
    
    # Targets to try
    # The budget is 1.
    # Sum of 1/p <= 1.
    # How many primes can we use?
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 ...
    # Harmonic series of primes grows slowly.
    # Sum of 1/p for p <= 100 is approx 1.6? No, sum 1/p diverges but slowly.
    # Sum 1/p for p <= 100 is ~ 1.9. So we can use many primes.
    
    # Let's try to cover up to 2000, 5000, etc.
    # But greedy might get stuck if it uses budget inefficiently.
    
    # Let's try multiple random restarts or different heuristics?
    # The prompt mentions "Dynamic greedy selection with efficiency scoring (cnt*p)".
    # My current greedy maximizes `count`. 
    # `count * p` is the "efficiency" (coverage per unit of budget 1/p).
    # Let's switch to maximizing `count * p` (or `count / (1/p)`).
    
    def greedy_efficient(L_target):
        classes = []
        used_primes = set()
        current_sum = Fraction(0, 1)
        uncovered = set(range(1, L_target + 1))
        
        while uncovered and current_sum < 1:
            best_p = None
            best_r = None
            best_score = -1
            
            n_min = min(uncovered)
            
            for p in PRIMES:
                if p in used_primes:
                    continue
                if current_sum + INV_PRIMES[p] > 1:
                    continue
                
                r = n_min % p
                
                # Count coverage
                count = sum(1 for n in uncovered if n % p == r)
                
                # Score: count * p (efficiency)
                # Or maybe just count?
                # Let's try count * p.
                score = count * p
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
                    
            if best_p is None:
                break
                
            classes.append([best_p, best_r])
            used_primes.add(best_p)
            current_sum += INV_PRIMES[best_p]
            
            # Update uncovered
            new_uncovered = set(n for n in uncovered if n % best_p != best_r)
            uncovered = new_uncovered
            
            if time.time() > DEADLINE - 5:
                break
                
        return classes

    # Try different targets
    # Start with 1000 (to match seed) and go up.
    targets = [1000, 1500, 2000, 3000, 5000]
    
    for L_target in targets:
        if time.time() > DEADLINE - 10:
            break
            
        # Run greedy
        classes = greedy_efficient(L_target)
        
        if not classes:
            continue
            
        # Verify sum
        s = sum(Fraction(1, p) for p, r in classes)
        if s > 1:
            continue
            
        # Calculate actual L
        # We can calc L up to a reasonable limit.
        # If the greedy covered 1..L_target, then L >= L_target.
        # But maybe it covers more?
        # Let's calc L up to L_target + 1000 or so.
        # Actually, calc_L checks for gaps.
        
        # To be fast, let's just check if 1..L_target are covered.
        # If yes, L >= L_target.
        # Then we can try to extend L.
        
        # Let's use calc_L with a cap.
        # If L_target is 5000, checking 5000 is fast.
        
        L_achieved = calc_L(classes, max_check=max(L_target + 100, 1000))
        
        if L_achieved > best_L:
            best_L = L_achieved
            best_classes = classes
            
            # If we beat 999, we are already good.
            # Can we do better?
            # Let's continue to higher targets if time permits.
            
    # If we didn't find anything, fallback to a simple valid solution
    if best_classes is None:
        # Simple fallback: cover 1..100 with small primes
        # 2,1 ; 3,2 ; 5,4 ...
        # Just return a trivial valid set, e.g. empty or single class.
        # But we want to maximize score.
        # Let's try a very small greedy.
        best_classes = greedy_efficient(100)
        best_L = calc_L(best_classes, 1000)
        
    return best_classes

# To ensure we don't timeout, we add a global check in construct?
# The greedy loops have internal checks.

if __name__ == "__main__":
    # Self-test
    classes = construct()
    L = calc_L(classes, 10000)
    s = sum(Fraction(1, p) for p, r in classes)
    print(f"Found L={L}, Sum={float(s)}, Classes={len(classes)}")
