import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    TIME_LIMIT = 110
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    reciprocals = {p: Fraction(1, p) for p in primes}
    
    MAX_L = 2000
    
    # State
    selected = [] # List of (p, r)
    count = [0] * (MAX_L + 1)
    covered = [False] * (MAX_L + 1)
    current_sum = Fraction(0, 1)
    first_gap = 1
    
    def update_first_gap():
        nonlocal first_gap
        # If first_gap is covered, find next
        while first_gap <= MAX_L and covered[first_gap]:
            first_gap += 1
            
    def add_class(p, r):
        nonlocal current_sum, first_gap
        current_sum += reciprocals[p]
        selected.append((p, r))
        
        # Update counts
        start = r
        if start == 0:
            start = p
        for x in range(start, MAX_L + 1, p):
            if count[x] == 0:
                covered[x] = True
            count[x] += 1
            
        # If we covered first_gap, update
        if covered[first_gap]:
            update_first_gap()

    def remove_class(p, r):
        nonlocal current_sum, first_gap
        current_sum -= reciprocals[p]
        selected.remove((p, r))
        
        # Update counts
        start = r
        if start == 0:
            start = p
        for x in range(start, MAX_L + 1, p):
            count[x] -= 1
            if count[x] == 0:
                covered[x] = False
                
        # If we uncovered first_gap, it stays the same (or becomes first_gap if it was covered? No, first_gap is the FIRST uncovered. If we uncover a number < first_gap, that's impossible because first_gap was the first uncovered. Wait.
        # If we remove a class, we might uncover a number.
        # If we uncover a number x < first_gap, that means x was covered before, so x < first_gap.
        # But first_gap is the smallest uncovered. So all x < first_gap were covered.
        # If we uncover x, then x becomes the new first_gap (if x < old first_gap).
        # So we need to check if any x < first_gap became uncovered.
        # Actually, if we uncover ANY x, and x < first_gap, then new first_gap <= x.
        # We can just re-scan from 1? No, that's slow.
        # We can check if the specific x we uncovered is < first_gap.
        # But we might uncover multiple.
        # Simplest: if we removed a class, first_gap might decrease.
        # We can just set first_gap = 1 and update_first_gap()?
        # update_first_gap only moves forward.
        # So we need a function to find first gap from scratch or move backwards.
        
        # Let's just re-scan if we uncovered something?
        # Or track min_uncovered.
        # For now, let's assume we re-scan from 1 if we uncovered something.
        # Optimization: Only re-scan if we uncovered a number.
        
        # Let's modify remove_class to return if any was uncovered
        uncovered_any = False
        start = r
        if start == 0:
            start = p
        for x in range(start, MAX_L + 1, p):
            count[x] -= 1
            if count[x] == 0:
                if covered[x]:
                    covered[x] = False
                    uncovered_any = True
        
        if uncovered_any:
            # Re-find first gap
            first_gap = 1
            update_first_gap()

    # Initialize
    # Greedy start
    random.shuffle(primes)
    for p in primes:
        if current_sum + reciprocals[p] <= 1:
            r = random.randint(0, p-1)
            add_class(p, r)
            
    best_L = first_gap - 1
    best_selected = selected.copy()
    
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < TIME_LIMIT:
        move_type = random.random()
        
        # We need to be able to rollback
        # Save state
        old_selected = selected.copy()
        old_count = count.copy()
        old_covered = covered.copy()
        old_sum = current_sum
        old_first_gap = first_gap
        
        new_selected = selected.copy()
        
        if move_type < 0.5:
            if selected:
                idx = random.randint(0, len(selected)-1)
                p, r = selected[idx]
                new_r = random.randint(0, p-1)
                # Remove old, add new
                remove_class(p, r)
                add_class(p, new_r)
        elif move_type < 0.8:
            if selected:
                idx = random.randint(0, len(selected)-1)
                p_old, r_old = selected[idx]
                p_new = random.choice(primes)
                if any(p == p_new for p, _ in selected):
                    # Rollback
                    selected = old_selected
                    count = old_count
                    covered = old_covered
                    current_sum = old_sum
                    first_gap = old_first_gap
                    continue
                new_sum = current_sum - reciprocals[p_old] + reciprocals[p_new]
                if new_sum > 1:
                    selected = old_selected
                    count = old_count
                    covered = old_covered
                    current_sum = old_sum
                    first_gap = old_first_gap
                    continue
                remove_class(p_old, r_old)
                new_r = random.randint(0, p_new-1)
                add_class(p_new, new_r)
        else:
            if random.random() < 0.5:
                if len(selected) < len(primes):
                    p_new = random.choice(primes)
                    if any(p == p_new for p, _ in selected):
                        selected = old_selected
                        count = old_count
                        covered = old_covered
                        current_sum = old_sum
                        first_gap = old_first_gap
                        continue
                    if current_sum + reciprocals[p_new] > 1:
                        selected = old_selected
                        count = old_count
                        covered = old_covered
                        current_sum = old_sum
                        first_gap = old_first_gap
                        continue
                    new_r = random.randint(0, p_new-1)
                    add_class(p_new, new_r)
            else:
                if selected:
                    idx = random.randint(0, len(selected)-1)
                    p, r = selected[idx]
                    remove_class(p, r)
        
        new_L = first_gap - 1
        new_extra = 0
        # Count extra coverage
        for i in range(first_gap, min(first_gap + 50, MAX_L + 1)):
            if covered[i]:
                new_extra += 1
        
        new_score = new_L * 1000 + new_extra
        old_L = old_first_gap - 1
        old_extra = 0
        for i in range(old_first_gap, min(old_first_gap + 50, MAX_L + 1)):
            if old_covered[i]:
                old_extra += 1
        old_score = old_L * 1000 + old_extra
        
        if new_score > old_score:
            if new_L > best_L:
                best_L = new_L
                best_selected = selected.copy()
            T *= 0.95
        else:
            delta = new_score - old_score
            if delta < 0:
                prob = pow(2.718, delta / T)
                if random.random() >= prob:
                    # Reject
                    selected = old_selected
                    count = old_count
                    covered = old_covered
                    current_sum = old_sum
                    first_gap = old_first_gap
            else:
                if random.random() < 0.5:
                    # Reject tie
                    selected = old_selected
                    count = old_count
                    covered = old_covered
                    current_sum = old_sum
                    first_gap = old_first_gap
            
        T *= cooling
        
    return [[p, r] for p, r in best_selected]

construct()
