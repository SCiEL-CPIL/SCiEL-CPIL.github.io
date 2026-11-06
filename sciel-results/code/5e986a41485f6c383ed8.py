import time
import random
import math

def construct():
        # Deadline guard
        start_time = time.time()
        deadline = 110.0 # 10s margin
        
        # Primes up to 1000
        primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        n_primes = len(primes)
        M = 2000 # Window size for evaluation
        
        # Helper to calculate score
        def get_score(counts):
            L = 0
            for x in range(1, M + 1):
                if counts[x] >= 2:
                    L = x
                else:
                    break
            # Count deficiencies in the window
            def_count = sum(1 for x in range(1, M + 1) if counts[x] < 2)
            # Heavily weight L
            return L * 1000 - def_count
        
        best_score = -1
        best_residues = None
        
        # Try a few random restarts
        max_restarts = 50 # Adjust based on time
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize residues
            # Strategy: Random or Greedy
            # Let's try a semi-greedy init:
            # Start with random, then do a quick greedy pass to fix early holes
            residues = [random.randint(0, p - 1) for p in primes]
            
            # Build counts
            counts = [0] * (M + 1)
            for i, p in enumerate(primes):
                r = residues[i]
                # x in [1, M] such that x % p == r
                # First x >= 1
                start = r
                if start < 1:
                    start += p
                # Actually, x can be r, r+p, ...
                # If r==0, x are p, 2p...
                # If r>0, x are r, r+p...
                # Wait, congruence x = r (mod p).
                # If r=0, x is multiple of p.
                # If r>0, x is r, r+p...
                # Range 1..M
                
                # Efficient update
                # x = r + k*p
                # 1 <= r + k*p <= M
                # k >= (1-r)/p
                # k <= (M-r)/p
                
                # Simpler:
                first = r
                if first < 1:
                    first += p
                # If r==0, first=p (since x>=1). If r>0, first=r (if r>=1).
                # Wait, if r=0, x=0 is not in [1,M]. Next is p.
                # If r=1, x=1.
                # So:
                if r == 0:
                    first = p
                else:
                    first = r
                    if first < 1: # impossible since r>=1
                        first += p
                
                if first <= M:
                    for x in range(first, M + 1, p):
                        counts[x] += 1
            
            current_score = get_score(counts)
            
            # Quick Greedy Improvement
            # Try to fix holes from 1 upwards
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                L = 0
                for x in range(1, M + 1):
                    if counts[x] >= 2:
                        L = x
                    else:
                        break
                
                if L >= M:
                    break
                
                # Try to fix x = L + 1
                target_x = L + 1
                if counts[target_x] >= 2:
                    continue
                
                # Find a prime p that does NOT cover target_x, and changing it to cover target_x
                # doesn't break any y <= L.
                # This is complex to check exactly.
                # Instead, just try random changes on primes that don't cover target_x.
                # Pick a random prime index
                # We want to increase count[target_x].
                # So we need a prime p where residues[i] != target_x % p.
                
                candidates = []
                for i, p in enumerate(primes):
                    if residues[i] != target_x % p:
                        candidates.append(i)
                
                if not candidates:
                    break
                    
                # Try a few random candidates
                random.shuffle(candidates)
                for i in candidates[:5]:
                    p = primes[i]
                    old_r = residues[i]
                    new_r = target_x % p
                    
                    # Apply change
                    # Decrement old_r
                    first_old = old_r if old_r > 0 else p
                    if first_old <= M:
                        for x in range(first_old, M + 1, p):
                            counts[x] -= 1
                            
                    # Increment new_r
                    first_new = new_r if new_r > 0 else p
                    if first_new <= M:
                        for x in range(first_new, M + 1, p):
                            counts[x] += 1
                    
                    # Check if L improved
                    new_L = 0
                    for x in range(1, M + 1):
                        if counts[x] >= 2:
                            new_L = x
                        else:
                            break
                    
                    if new_L > L:
                        L = new_L
                        residues[i] = new_r
                        improved = True
                        break
                    else:
                        # Revert
                        first_old = old_r if old_r > 0 else p
                        if first_old <= M:
                            for x in range(first_old, M + 1, p):
                                counts[x] += 1
                        first_new = new_r if new_r > 0 else p
                        if first_new <= M:
                            for x in range(first_new, M + 1, p):
                                counts[x] -= 1
                        residues[i] = old_r

            current_score = get_score(counts)
            
            # Simulated Annealing
            T = 100.0
            cooling = 0.999
            step = 0
            while time.time() < deadline - 2:
                # Pick random prime
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                # Apply
                first_old = old_r if old_r > 0 else p
                if first_old <= M:
                    for x in range(first_old, M + 1, p):
                        counts[x] -= 1
                first_new = new_r if new_r > 0 else p
                if first_new <= M:
                    for x in range(first_new, M + 1, p):
                        counts[x] += 1
                
                new_score = get_score(counts)
                
                if new_score > current_score:
                    current_score = new_score
                    residues[i] = new_r
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    # Accept with prob
                    delta = current_score - new_score
                    if delta < 0:
                        prob = 1.0
                    else:
                        prob = math.exp(-delta / T)
                    if random.random() < prob:
                        residues[i] = new_r
                        current_score = new_score
                    else:
                        # Revert
                        first_old = old_r if old_r > 0 else p
                        if first_old <= M:
                            for x in range(first_old, M + 1, p):
                                counts[x] += 1
                        first_new = new_r if new_r > 0 else p
                        if first_new <= M:
                            for x in range(first_new, M + 1, p):
                                counts[x] -= 1
                        residues[i] = old_r
                
                T *= cooling
                step += 1
                if step % 10000 == 0:
                    # Reheat? Or just let it cool
                    pass
            
            # Check time
            if time.time() > deadline - 10:
                break

        # Final Evaluation of best_residues
        if best_residues is None:
            # Fallback to seed-like or random
            return [((i + 2) // 2) % p for i, p in enumerate(primes)]
        
        # Calculate actual L for best_residues (check up to 10000 just in case, or 10^5 cap)
        # The score is capped at 10^5, but we only optimized up to M=2000.
        # We should verify the actual L of the best solution found.
        # If L > M, we might have more.
        
        counts = [0] * (10001) # Check up to 10000
        for i, p in enumerate(primes):
            r = best_residues[i]
            first = r if r > 0 else p
            if first <= 10000:
                for x in range(first, 10001, p):
                    counts[x] += 1
        
        L = 0
        for x in range(1, 10001):
            if counts[x] >= 2:
                L = x
            else:
                break
        
        return best_residues

    # Wait, the score in the prompt is "value/seed".
    # Seed value is 84.
    # If I return residues that give L=1000, the evaluator calculates score = 1000/84 = 11.9.
    # My internal `best_score` is just for optimization.
    # The function `construct` just returns the residues.
