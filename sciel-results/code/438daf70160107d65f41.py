import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate primes up to n."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

# Precompute primes up to 1000
PRIMES = _sieve(1000)
PRIME_SET = set(PRIMES)

def _calculate_coverage(classes):
    """
    Calculate the largest L such that 1..L are covered by the given classes.
    classes: list of [p, r]
    Returns L (capped at 100000).
    """
    if not classes:
        return 0
    
    # We use a limit of 5000 for search speed. 
    # If we cover 5000, we likely cover more, but the score is capped at 10^5.
    # The judge will evaluate the exact L. 
    # For the search, we need a fast proxy that correlates with L.
    # If we find a gap before 5000, L is that gap - 1.
    # If we cover 1..5000, we return 5000.
    # Note: If the true L > 5000, this returns 5000. 
    # To distinguish solutions that cover >5000, we might need a higher limit or a secondary check.
    # However, the gap-targeting strategy works by extending the prefix. 
    # If we reach 5000, we can try to extend further in subsequent steps if we detect the gap is beyond our limit.
    # Let's use 10000 to be safer for the search phase. It's still fast with bytearray.
    
    limit = 10000
    covered = bytearray(limit + 1)
    
    for p, r in classes:
        # Mark numbers n in [1, limit] such that n % p == r
        # n = r + k*p. 
        # If r == 0, n = p, 2p, ...
        # If r > 0, n = r, r+p, ... (provided r >= 1)
        
        # We need the smallest n >= 1 such that n % p == r.
        if r == 0:
            start = p
        else:
            start = r
            
        if start < 1:
            start += p
            
        if start > limit:
            continue
            
        for n in range(start, limit + 1, p):
            covered[n] = 1
            
    # Find the largest L such that 1..L are all True
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return L

def _sum_reciprocals(classes):
    """Calculate sum of 1/p for given classes."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Initial solution: The seed program's solution
    # The i-th prime from 13 upward covers the integer i.
    # This is a known covering system for 1..(pi(1000)-pi(12)) roughly?
    # Actually, the seed covers 1..L where L is determined by the primes used.
    # Let's just start with a reasonable greedy or the seed itself.
    
    primes_from_13 = [p for p in PRIMES if p >= 13]
    initial_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_from_13)]
    
    # Check if initial is valid
    if _sum_reciprocals(initial_classes) > 1:
        # Fallback to empty if invalid
        initial_classes = []
        
    current_classes = [c[:] for c in initial_classes]
    current_score = _calculate_coverage(current_classes)
    best_classes = [c[:] for c in current_classes]
    best_score = current_score
    
    # Track used primes for quick lookup
    used_primes = set(p for p, r in current_classes)
    
    # Main search loop: Gap-targeting greedy
    # We repeatedly find the first gap (smallest n not covered) and try to fix it.
    
    while time.time() < deadline:
        # Find the first gap
        limit_check = 10000
        covered = bytearray(limit_check + 1)
        for p, r in current_classes:
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            if start > limit_check:
                continue
            for n in range(start, limit_check + 1, p):
                covered[n] = 1
                
        gap = -1
        for i in range(1, limit_check + 1):
            if not covered[i]:
                gap = i
                break
                
        if gap == -1:
            # Covered up to limit_check. 
            # We can try to extend the limit or just stop if we are near the cap.
            # If we cover 10000, the score is at least 10000.
            # We can break if we are confident, or try to optimize further.
            # Since the score is capped at 10^5, and we are at 10^4, we might have room.
            # But for now, let's assume 10000 is a good target.
            # If we want to be more aggressive, we can increase limit_check dynamically.
            # For simplicity, we break if we cover the search limit.
            break
            
        # We have a gap at 'gap'. We need to cover 'gap'.
        # We try to find a move that covers 'gap' with minimal cost or best improvement.
        
        # Move types:
        # 1. Add a new prime p such that gap % p == r.
        #    Cost: 1/p.
        #    Benefit: Covers gap and other numbers.
        # 2. Change residue of an existing prime p to r = gap % p.
        #    Cost: 0 (budget unchanged).
        #    Benefit: Covers gap, but might uncover other numbers.
        # 3. Swap an existing prime p_old with a new prime p_new.
        #    p_new must cover gap (gap % p_new == r_new).
        #    Cost: 1/p_new - 1/p_old.
        #    Benefit: Covers gap, removes p_old's coverage.
        
        best_move = None
        best_score_delta = -1
        best_new_classes = None
        
        # Option 2: Change residue of existing prime
        # This is usually the cheapest way to cover a specific number if a prime already in the set can cover it.
        # We check all existing primes.
        for idx, (p, r) in enumerate(current_classes):
            new_r = gap % p
            if new_r == r:
                continue # Already covers it? No, if gap is a gap, it's not covered. 
                         # Wait, if p is in classes, and gap % p == r, then gap IS covered.
                         # So this case shouldn't happen if 'gap' is truly uncovered by current_classes.
                         # Unless my coverage check is wrong? 
                         # No, if p,r is in classes, and n % p == r, n is covered.
                         # So if gap is uncovered, no existing p has gap % p == r.
                         # So changing r to gap % p will definitely cover gap.
                         
            # Create new classes
            new_classes = [c[:] for c in current_classes]
            new_classes[idx] = [p, new_r]
            
            # Evaluate
            new_score = _calculate_coverage(new_classes)
            delta = new_score - current_score
            
            if delta > best_score_delta:
                best_score_delta = delta
                best_move = 'change_residue'
                best_new_classes = new_classes
                
        # Option 1: Add a new prime
        # We need to find a prime p not in used_primes such that we can choose r = gap % p.
        # And sum + 1/p <= 1.
        current_sum = _sum_reciprocals(current_classes)
        
        # To make this efficient, we can just try a few small primes? 
        # Or iterate through available primes.
        # There are ~168 primes <= 1000.
        # We can filter by budget.
        
        # Optimization: Sort available primes by 1/p descending (smallest p first) to find cheap additions?
        # Actually, we want to maximize delta. 
        # Adding a small p is expensive (high 1/p) but covers many numbers.
        # Adding a large p is cheap but covers few.
        
        # Let's try a subset of candidates to keep it fast.
        # Candidates: primes not in used_primes.
        # We can prioritize primes that are "divisible" by gap? No, r = gap % p.
        
        # Let's just iterate through all available primes if time permits, or a random subset.
        # For determinism and speed, let's check all available primes.
        available_primes = [p for p in PRIMES if p not in used_primes]
        
        # Sort by p ascending? Small p covers more.
        # But small p costs more budget.
        
        for p in available_primes:
            if current_sum + Fraction(1, p) > 1:
                continue # Budget exceeded
                
            new_r = gap % p
            new_classes = [c[:] for c in current_classes]
            new_classes.append([p, new_r])
            
            new_score = _calculate_coverage(new_classes)
            delta = new_score - current_score
            
            if delta > best_score_delta:
                best_score_delta = delta
                best_move = 'add_prime'
                best_new_classes = new_classes
                
        # Option 3: Swap
        # Remove p_old, add p_new.
        # p_new must cover gap.
        # Budget: sum - 1/p_old + 1/p_new <= 1.
        
        # This is O(N^2) if we check all pairs. N is ~100. 10000 checks.
        # Each check involves _calculate_coverage which is O(N * Limit/p).
        # This might be too slow if done every iteration.
        # Let's restrict swaps to a few candidates or skip if time is tight.
        
        # For now, let's skip swaps in the main loop for speed, or only try them if no add/change works.
        # Or try a limited number of swaps.
        
        # If we found a move that improves the score
        if best_move and best_score_delta > 0:
            current_classes = best_new_classes
            current_score = best_score_delta + current_score # Wait, delta is relative to current.
            # Actually, best_score_delta is new_score - current_score.
            # So new_score = current_score + best_score_delta.
            # We already computed new_score in the loop, but didn't store it separately from delta.
            # Let's just re-calculate or store it.
            # In the loop above, I calculated new_score. I should store it.
            # Let's refactor slightly to store best_new_score.
            
            # Re-evaluate to be safe
            current_score = _calculate_coverage(current_classes)
            
            if current_score > best_score:
                best_score = current_score
                best_classes = [c[:] for c in current_classes]
                
            # Update used_primes
            used_primes = set(p for p, r in current_classes)
            
        else:
            # No improving move found for this gap.
            # We might be stuck. 
            # We could try a random perturbation or accept a worse move to escape local optimum.
            # Or just break if we can't improve.
            
            # Let's try a random move to escape.
            # Pick a random prime from current_classes and change its residue.
            if current_classes:
                idx = random.randint(0, len(current_classes) - 1)
                p, r = current_classes[idx]
                new_r = random.randint(0, p - 1)
                new_classes = [c[:] for c in current_classes]
                new_classes[idx] = [p, new_r]
                
                new_score = _calculate_coverage(new_classes)
                
                # Accept if it's not too bad? Or always accept to explore?
                # If we always accept, we might drift away from the best.
                # Let's accept if it's within some distance of best, or if we haven't improved in a while.
                # For simplicity, let's just accept random moves if we are stuck, but keep track of best.
                current_classes = new_classes
                current_score = new_score
                used_primes = set(p for p, r in current_classes)
            else:
                # No classes, can't do much.
                break

    # Final verification
    if _sum_reciprocals(best_classes) > 1:
        return initial_classes
        
    return best_classes

# EVOLVE-BLOCK-END
