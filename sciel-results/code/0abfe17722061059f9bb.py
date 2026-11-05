"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    n_max = 256
    
    # Build conflict graph: a and b (distinct) conflict if (a*b) % (a+b) == 0
    # Using the number theory: a=g*m, b=g*n, gcd(m,n)=1, conflict iff (m+n)|g
    
    # Start with all odd numbers
    selected = set(range(1, n_max + 1, 2))
    
    # Candidate even numbers to consider adding
    evens = list(range(2, n_max + 1, 2))
    
    # Precompute conflicts for each even with all selected (odds)
    # For efficiency, check conflict on the fly
    
    def has_conflict(num, selected_set):
        """Check if num conflicts with any element in selected_set."""
        for other in selected_set:
            if num == other:
                continue
            s = num + other
            p = num * other
            if p % s == 0:
                return True
        return False
    
    # Try adding even numbers greedily
    # Sort evens by how many conflicts they have with current set (fewer first)
    # Actually, let's try a few strategies
    
    # Strategy 1: try evens in increasing order
    for e in evens:
        if time.time() > deadline:
            break
        if not has_conflict(e, selected):
            selected.add(e)
    
    # Try another pass: maybe some evens that were skipped could now be added
    # if we removed some others? No, we only add.
    # Let's try a different ordering: evens with fewer conflicts first
    
    # Re-evaluate: compute conflict count for each even with the final odd set
    # and try adding in order of increasing conflict count
    # But we already added some. Let's try a fresh approach with ordering.
    
    # Actually, let's try multiple strategies and keep the best
    
    best_result = set(selected)
    
    # Strategy 2: sort evens by number of conflicts with odds only
    def count_conflicts_with_odds(e, odds_set):
        cnt = 0
        for o in odds_set:
            if e != o:
                s = e + o
                p = e * o
                if p % s == 0:
                    cnt += 1
        return cnt
    
    odds_only = set(range(1, n_max + 1, 2))
    
    evens_sorted = sorted(evens, key=lambda e: count_conflicts_with_odds(e, odds_only))
    
    selected2 = set(odds_only)
    for e in evens_sorted:
        if time.time() > deadline:
            break
        if not has_conflict(e, selected2):
            selected2.add(e)
    
    if len(selected2) > len(best_result):
        best_result = set(selected2)
    
    # Strategy 3: try evens in decreasing order of 2-adic valuation (higher powers of 2 first)
    # Numbers with higher 2-adic valuation might have fewer conflicts
    def v2(n):
        c = 0
        while n % 2 == 0:
            n //= 2
            c += 1
        return c
    
    evens_by_v2 = sorted(evens, key=lambda e: -v2(e))
    selected3 = set(odds_only)
    for e in evens_by_v2:
        if time.time() > deadline:
            break
        if not has_conflict(e, selected3):
            selected3.add(e)
    
    if len(selected3) > len(best_result):
        best_result = set(selected3)
    
    # Strategy 4: random restarts with random ordering of evens
    import random
    rng = random.Random(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        random_evens = list(evens)
        rng.shuffle(random_evens)
        selected4 = set(odds_only)
        for e in random_evens:
            if not has_conflict(e, selected4):
                selected4.add(e)
        if len(selected4) > len(best_result):
            best_result = set(selected4)
    
    # Final verification
    result = list(best_result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                # Invalid, should not happen
                return list(range(1, n_max + 1, 2))
    
    return result
# EVOLVE-BLOCK-END
