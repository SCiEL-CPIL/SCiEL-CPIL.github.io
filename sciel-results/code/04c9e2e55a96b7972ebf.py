import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 125  # 25s safety margin for 150s limit

    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(math.isqrt(p)) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    
    # We want to maximize L such that every integer in [1, L] is covered by at least 2 congruences.
    # The score is capped at 10^5.
    # Let's aim for L up to 20000 initially, but we can expand if we find good solutions.
    # Actually, the "seed" score was 3.6471. If the seed was L=1000, 3.6471/1000 is not 1.0?
    # Wait, "search score is value/seed".
    # If the seed program returns L, and the score is L/seed_L.
    # The prompt says: "The evaluator measures the raw value (higher is better). The search score is value/seed, so 1.0 equals the seed program".
    # This implies the score returned by the evaluator is the raw L.
    # But the "score 3.6471" in the prompt header likely refers to the *normalized* score or a specific metric?
    # "The score is the largest L ... Maximize; L >= 1000 would realize the property... Scores are doubly covered lengths for n = 1000".
    # So the raw score IS L.
    # If the current program has "score 3.6471", that's confusing. L must be an integer.
    # Ah, maybe the "score 3.6471" is from a different context or I am misinterpreting "score".
    # Let's re-read: "INSPIRATIONS FROM OTHER SOLUTIONS: - inspiration 1 (score 3.6471)".
    # And "CURRENT PROGRAM (score 3.6471)".
    # If L is an integer, how can the score be 3.6471?
    # Perhaps the score is NOT L.
    # "The score is the largest L such that every integer 1..L satisfies at least two of the congruences".
    # Maybe the "score" in the prompt metadata is normalized?
    # "search score is value/seed".
    # If seed value is V_seed, and my value is V, score = V/V_seed.
    # If current score is 3.6471, and it's a ratio, then V = 3.6471 * V_seed.
    # But usually these tasks report the raw value.
    # Let's look at the instruction: "The evaluator measures the raw value (higher is better)."
    # So the function `construct` returns residues. The evaluator calculates L.
    # The "score 3.6471" might be a typo in the prompt description or refers to a different metric (like log L?).
    # Or maybe the "seed" value is small, e.g. 1000, and the current program achieves 3647?
    # If L=3647, that's a good score.
    # Let's assume the goal is to maximize L.
    # I will aim for L as large as possible.
    
    # Strategy:
    # 1. Greedy construction to get a decent initial L.
    # 2. Local search / Simulated Annealing to improve L.
    
    # To be fast, we should use a small L_max for the search, but the evaluator will check up to 10^5.
    # If we find a solution that works for L=5000, it might break at 5001.
    # We need to ensure the solution is robust.
    # However, optimizing for L=1000 is much faster than L=10000.
    # Let's try to optimize for L_target = 5000. If we succeed, we can try to extend.
    # But the evaluation is on the RAW L.
    # If I optimize for L=1000, I might get L=1000.
    # If I optimize for L=5000, I might get L=5000.
    # The cost of checking coverage for L=5000 is 5x higher than L=1000.
    # Given 150s, we can afford L=5000 or even 10000 with efficient updates.
    
    L_max = 10000
    
    # Initialize residues
    # A good heuristic: r_p = 0 for all p?
    # If r_p = 0, then x is covered if x is divisible by p.
    # For x to be covered by 2 primes, x must be divisible by at least 2 primes.
    # This fails for primes x < 1000 (only 1 divisor p=x).
    # So r_p=0 is bad.
    
    # Alternative: Random initialization.
    r = [random.randint(0, p - 1) for p in primes]
    
    # Coverage array
    cov = [0] * (L_max + 1)
    
    # bad_set: indices x in [1, L_max] where cov[x] < 2
    bad_set = set(range(1, L_max + 1))
    
    def update_bad_set(x, delta):
        # delta is +1 or -1
        cov[x] += delta
        if cov[x] < 2:
            bad_set.add(x)
        else:
            bad_set.discard(x)
    
    def apply_change(idx, new_r):
        p = primes[idx]
        old_r = r[idx]
        if old_r == new_r:
            return
        
        # Remove old coverage
        x = old_r
        while x <= L_max:
            if x > 0:
                update_bad_set(x, -1)
            x += p
            
        # Add new coverage
        r[idx] = new_r
        x = new_r
        while x <= L_max:
            if x > 0:
                update_bad_set(x, +1)
            x += p

    def get_L():
        if not bad_set:
            return L_max
        return min(bad_set) - 1

    # Initial coverage
    for i, p in enumerate(primes):
        x = r[i]
        while x <= L_max:
            if x > 0:
                update_bad_set(x, +1)
            x += p

    current_L = get_L()
    
    # Greedy Phase
    # Try to fix the first bad x.
    # If x is bad (cov < 2), we need to increase cov[x].
    # We can do this by changing a prime's residue to x % p.
    # We prefer primes that don't cover many other "critical" points (cov==2).
    
    start_time_greedy = time.time()
    greedy_deadline = start_time_greedy + 20
    
    while time.time() < greedy_deadline and current_L < L_max:
        x_fail = min(bad_set)
        
        best_idx = -1
        best_cost = float('inf')
        
        # Check all primes
        for i, p in enumerate(primes):
            new_r = x_fail % p
            old_r = r[i]
            if new_r == old_r:
                continue
            
            # Calculate cost: number of points in [1, x_fail] that are covered by old_r but not new_r,
            # and have cov == 2 (would break).
            # Actually, we just need to see if it breaks anything in [1, current_L].
            # Since x_fail is the first bad one, all x < x_fail have cov >= 2.
            # If we remove old_r coverage, any x < x_fail with cov[x]==2 and covered by old_r will break.
            
            break_count = 0
            x = old_r
            while x < x_fail:
                if x > 0 and cov[x] == 2:
                    break_count += 1
                x += p
            
            # We want to minimize break_count. Ideally 0.
            # If break_count > 0, this move is bad (unless we accept temporary degradation, but greedy usually doesn't).
            # Let's only accept if break_count == 0.
            
            if break_count == 0:
                # If multiple have 0, pick one with largest p? (less frequent collisions?)
                # Or random?
                # Let's pick the one with largest p to spread coverage.
                if best_idx == -1 or p > primes[best_idx]:
                    best_idx = i
                    best_cost = 0
        
        if best_idx != -1:
            new_r = x_fail % primes[best_idx]
            apply_change(best_idx, new_r)
            current_L = get_L()
        else:
            # No safe move found. Break to SA.
            break
    
    # SA Phase
    # T starts high, cools down.
    T = 50.0
    cooling = 0.999
    
    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = r[i]
        
        # Propose new residue
        # Bias towards residues that cover current bad spots?
        # For now, random.
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
        
        apply_change(i, new_r)
        new_L = get_L()
        
        delta_L = new_L - current_L
        
        if delta_L > 0 or (T > 0 and random.random() < math.exp(delta_L / T)):
            current_L = new_L
        else:
            apply_change(i, old_r)
            
        T *= cooling
        
        # If we hit L_max, we can stop or try to extend L_max?
        # The problem says capped at 10^5.
        # If current_L == L_max, we might want to increase L_max to search for more.
        # But increasing L_max requires rebuilding cov array and bad_set, which is expensive.
        # Given time constraints, let's stick to L_max=10000.
        # If we find L=10000, that's a very good score.
        
    return r

# To test locally
if __name__ == "__main__":
    res = construct()
    print("Residues:", res[:10])
    # Quick check
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(math.isqrt(p)) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    L_test = 2000
    cov = [0] * (L_test + 1)
    for i, p in enumerate(primes):
        r_p = res[i]
        x = r_p
        while x <= L_test:
            if x > 0:
                cov[x] += 1
            x += p
    
    L = 0
    for x in range(1, L_test + 1):
        if cov[x] >= 2:
            L = x
        else:
            break
    print("Local L:", L)
