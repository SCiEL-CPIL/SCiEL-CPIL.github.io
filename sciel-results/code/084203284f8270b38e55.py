import random
import math
import time

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We will search for a solution that covers up to M
    # Start with a reasonable M, expand if we cover it quickly
    M = 3000 
    max_M = 10000
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in primes]
    
    def compute_counts(res):
        counts = [0] * (M + 1)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            if r == 0:
                start = p
            else:
                start = r
            for x in range(start, M + 1, p):
                counts[x] += 1
        return counts

    def get_score(counts):
        for x in range(1, M + 1):
            if counts[x] < 2:
                return x - 1
        return M

    def count_good(counts):
        # Number of points with count >= 2
        return sum(1 for c in counts[1:M+1] if c >= 2)

    # Initial counts and score
    counts = compute_counts(residues)
    current_score = get_score(counts)
    best_score = current_score
    best_residues = residues[:]
    current_good = count_good(counts)
    best_good = current_good
    
    start_time = time.time()
    time_limit = 105.0 # Leave buffer
    
    # Greedy Hill Climbing with Perturbation
    # We try to fix the first failure. If stuck, we perturb.
    
    while (time.time() - start_time) < time_limit:
        # Check if we can extend M
        if current_score >= M:
            if M < max_M:
                M += 500
                # Extend counts array
                counts.extend([0] * 500)
                # Recompute counts for the new range? 
                # Actually, we just need to recompute the whole thing or update.
                # For simplicity, recompute.
                counts = compute_counts(residues)
                current_score = get_score(counts)
                current_good = count_good(counts)
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
            else:
                break
        
        x_fail = current_score + 1
        if x_fail > M:
            x_fail = M # Should not happen if score < M
        
        # Find primes that do NOT cover x_fail
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            if x_fail % p != residues[i]:
                candidates.append(i)
                
        if not candidates:
            # All primes cover x_fail, but count < 2? Impossible if n_primes >= 2 and all cover it.
            # This implies count >= n_primes. If count < 2, n_primes must be 1, which is false.
            # So this case is effectively impossible for x_fail with count < 2.
            # If we are here, it means logic error or x_fail is out of bounds.
            break

        # Evaluate each candidate
        best_delta = -1
        best_i = -1
        best_new_r = -1
        
        # Optimization: Don't evaluate ALL candidates if M is large and n_primes is large.
        # But 168 primes is small. Evaluating 168 changes per step is fine.
        
        for i in candidates:
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            if new_r == old_r:
                continue
                
            # Calculate delta in "good" count
            # We need to know how many points change from <2 to >=2 and vice versa.
            # This requires iterating over the progressions.
            # To make it fast, we can approximate or just do it.
            # M is up to 10000. Progression length is M/p.
            # Sum of M/p for p in primes is M * log(log(M)).
            # 10000 * log(log(10000)) ~ 10000 * 2.5 = 25000 operations per candidate.
            # 168 candidates * 25000 = 4.2 million operations per step.
            # This might be too slow if we do many steps.
            # Let's limit candidates to a random subset or just pick the best among a few.
            
            # Alternative: Just pick the candidate that minimizes p (largest frequency) or maximizes p?
            # Usually, small primes cover more points. Changing a small prime's residue has a larger impact.
            # Let's prioritize small primes.
            
            pass # We will implement a faster heuristic below

        # Faster Heuristic:
        # Instead of calculating exact delta for all, pick the candidate prime p 
        # that minimizes p (covers most points) among those that cover x_fail?
        # Or just pick one at random from the top 10 smallest primes in candidates?
        
        # Let's try: Pick the candidate with the smallest prime p.
        # Why? Because it affects the most points. If we fix x_fail, we likely fix others too.
        
        # Sort candidates by prime value (ascending)
        candidates.sort(key=lambda i: primes[i])
        
        # Try the top 5 candidates
        best_delta = -float('inf')
        best_i = -1
        best_new_r = -1
        
        for i in candidates[:5]:
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            if new_r == old_r:
                continue
                
            # Simulate change
            # Count points where count changes
            # We can compute the delta in "good" count
            delta_good = 0
            
            # Points losing coverage (old progression)
            start_old = old_r if old_r > 0 else p
            for x in range(start_old, M + 1, p):
                if counts[x] == 2:
                    delta_good -= 1 # Becomes 1, bad
                # If counts[x] > 2, it stays good (>=2), so no change in "good" count
                # If counts[x] < 2, it was bad, stays bad.
                
            # Points gaining coverage (new progression)
            start_new = new_r if new_r > 0 else p
            for x in range(start_new, M + 1, p):
                if counts[x] == 1:
                    delta_good += 1 # Becomes 2, good
                # If counts[x] == 0, becomes 1, still bad.
                # If counts[x] >= 2, stays good.

            if delta_good > best_delta:
                best_delta = delta_good
                best_i = i
                best_new_r = new_r
                
        if best_i == -1:
            # No good move found in top 5, pick random from candidates
            i = random.choice(candidates)
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
        else:
            i = best_i
            p = primes[i]
            old_r = residues[i]
            new_r = best_new_r
            
        # Apply move
        residues[i] = new_r
        
        # Update counts
        start_old = old_r if old_r > 0 else p
        for x in range(start_old, M + 1, p):
            counts[x] -= 1
        start_new = new_r if new_r > 0 else p
        for x in range(start_new, M + 1, p):
            counts[x] += 1
            
        new_score = get_score(counts)
        new_good = count_good(counts)
        
        # Accept if better score, or better good count, or random (to avoid sticking)
        if new_score > current_score:
            current_score = new_score
            current_good = new_good
            if new_score > best_score:
                best_score = new_score
                best_residues = residues[:]
        elif new_good > current_good:
            current_good = new_good
        elif random.random() < 0.1:
            # Accept worse move to escape local optima
            current_good = new_good
        else:
            # Revert
            residues[i] = old_r
            for x in range(start_new, M + 1, p):
                counts[x] -= 1
            for x in range(start_old, M + 1, p):
                counts[x] += 1
        
        # If we are stuck (no improvement in last many steps), perturb
        # Simple perturbation: randomize a few residues
        if random.random() < 0.01:
            for _ in range(5):
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                old_r = residues[idx]
                new_r = random.randint(0, p - 1)
                residues[idx] = new_r
                start_old = old_r if old_r > 0 else p
                for x in range(start_old, M + 1, p):
                    counts[x] -= 1
                start_new = new_r if new_r > 0 else p
                for x in range(start_new, M + 1, p):
                    counts[x] += 1
            current_score = get_score(counts)
            current_good = count_good(counts)

    return best_residues
