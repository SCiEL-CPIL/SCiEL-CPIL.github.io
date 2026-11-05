import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # 1. Generate Primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # 2. Helper: Calculate L for a given set of classes
    # We use a coverage array for speed.
    # MAX_L estimation: With budget 1, max L is likely around 1000-2000. 
    # The task says L >= 999 is the target. Let's check up to 5000 to be safe, 
    # but optimize for speed.
    MAX_L = 5000
    
    def calculate_L(active_classes):
        if not active_classes:
            return 0
        
        # Use numpy for fast coverage marking
        covered = np.zeros(MAX_L + 1, dtype=bool)
        
        for p, r in active_classes:
            # Mark indices i in [1, MAX_L] where i % p == r
            # Indices are r, r+p, r+2p...
            # If r == 0, it marks p, 2p... (0 is marked but ignored in L calc for 1..MAX_L)
            # If r > 0, it marks r, r+p...
            
            # Start index: 
            # We want indices >= 1.
            # If r == 0: first index is p.
            # If r > 0: first index is r.
            
            start = r if r > 0 else p
            if start <= MAX_L:
                # Slice: start, MAX_L+1, p
                covered[start::p] = True
            
        # Find first gap in 1..MAX_L
        # covered[1:] corresponds to 1..MAX_L
        # We want the first index in covered[1:] that is False.
        # np.where(~covered[1:]) gives indices of False in the slice.
        # Slice index k corresponds to number k+1.
        # If gap at slice index k, number k+1 is uncovered.
        # L is the number of consecutive covered integers starting from 1.
        # So if number k+1 is the first gap, L = k.
        
        gaps = np.where(~covered[1:])[0]
        if len(gaps) > 0:
            return gaps[0]
        else:
            return MAX_L

    def get_budget(active_classes):
        total = Fraction(0, 1)
        for p, r in active_classes:
            total += Fraction(1, p)
        return total

    # 3. Strategy: Greedy Frontier Coverage
    # State:
    # active_classes: list of [p, r]
    # budget: Fraction
    # covered_mask: numpy bool array (incremental update is complex, so we recalc or maintain carefully)
    # For simplicity and robustness, we maintain the list and recalc L when needed, 
    # but we use a "frontier" heuristic to pick the next move.
    
    # However, recalculating L from scratch for every candidate in a greedy step is expensive.
    # Greedy Step:
    # 1. Determine current L.
    # 2. Identify the "frontier" number N = L + 1.
    # 3. Find the (p, r) pair that covers N and maximizes the number of *additional* 
    #    covered numbers in [N, N+window] or simply maximizes the new L.
    # 
    # A more effective heuristic for this specific problem (covering 1..L):
    # Start with empty set.
    # At each step, pick the prime p (not yet used) and residue r such that:
    #   - Budget allows adding p.
    #   - The new L is maximized.
    # To speed this up, we don't try all r for all p. 
    # We observe that to cover the current frontier L+1, r must be (L+1) % p.
    # So for each available prime p, there is ONLY ONE residue r that covers L+1.
    # We can just test these candidates.
    
    def greedy_build():
        active_classes = []
        current_budget = Fraction(0, 1)
        current_L = 0
        
        # Sort primes? Small primes are "expensive" (high 1/p) but cover dense sets.
        # Large primes are "cheap" but cover sparse sets.
        # To cover a continuous range 1..L, we need dense coverage initially.
        # So small primes are crucial.
        
        # We iterate until no improvement is found or budget is exhausted.
        # Max iterations safeguard
        max_iters = 100
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            best_new_L = current_L
            best_new_class = None
            best_new_budget = None
            
            # Candidates: all primes not in active_classes
            active_p_set = {p for p, r in active_classes}
            
            # Optimization: Only consider primes that can fit in budget
            # And potentially improve L.
            
            for p in primes:
                if p in active_p_set:
                    continue
                    
                cost = Fraction(1, p)
                if current_budget + cost > Fraction(1, 1):
                    continue
                
                # To extend L, the new class MUST cover current_L + 1.
                # Let target = current_L + 1
                # r must be target % p
                r = (current_L + 1) % p
                
                # Check if this (p, r) is actually new (it is, since p is new)
                # Evaluate new L
                # To avoid full recalc, we can estimate? No, just recalc for candidates.
                # But there are ~168 primes. 168 * O(MAX_L) per step.
                # If we have 10 steps, 1680 * 5000 = 8.4M ops. Very fast.
                
                test_classes = active_classes + [[p, r]]
                new_L = calculate_L(test_classes)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_class = [p, r]
                    best_new_budget = current_budget + cost
            
            if best_new_class is None:
                break
                
            # Commit the best move
            active_classes.append(best_new_class)
            current_budget = best_new_budget
            current_L = best_new_L
            
            # If we've reached a high L, we might stop early if time is tight
            if current_L >= 4000:
                break
                
        return active_classes, current_L, current_budget

    # 4. Local Refinement (Hill Climbing on Residues)
    # The greedy approach fixes residues to cover the frontier. 
    # However, changing a residue of an *existing* prime might allow a better global L 
    # by shifting coverage to fill gaps later, or allowing a different prime to be added.
    # We perform a local search on the residues of the active primes.
    
    def local_refine(classes, budget, L):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try changing residue of each active prime
            for i in range(len(classes)):
                p, r = classes[i]
                # Try a few random residues or specific ones?
                # Since we want to maximize L, we should try residues that cover the current gap.
                # But changing r might break coverage of earlier numbers.
                # Simplest: Random restarts on residues for small subset?
                
                # Let's try all residues for the smallest few primes? No, too slow if p is large.
                # Let's try a few random residues.
                for _ in range(5):
                    new_r = random.randint(0, p - 1)
                    if new_r == r:
                        continue
                    new_classes = classes[:]
                    new_classes[i] = [p, new_r]
                    
                    # Check if L improves
                    new_L = calculate_L(new_classes)
                    if new_L > L:
                        classes = new_classes
                        L = new_L
                        improved = True
                        break # Restart loop as classes changed
                    
        return classes, L

    # 5. Main Execution
    # Phase 1: Greedy Build
    classes, L, budget = greedy_build()
    
    # Phase 2: Local Refinement
    classes, L = local_refine(classes, budget, L)
    
    # Phase 3: Multi-start Randomized Greedy
    # Run greedy a few more times with different random seeds or slight variations
    # to escape local optima.
    best_classes = classes
    best_L = L
    best_budget = budget
    
    # Use remaining time for randomized restarts
    # Each restart: Greedy build with random tie-breaking or random prime order?
    # Actually, the greedy logic is deterministic given the "maximize L" rule.
    # To vary it, we can shuffle the order of primes considered? No, we pick best L.
    # We can vary the initial state?
    # Or, we can perform a "Simulated Annealing" lite on the *set* of primes?
    
    # Let's do a few random restarts of the greedy process where we pick randomly 
    # among the top 3 candidates for L extension.
    
    def randomized_greedy(num_random_choices=3):
        active_classes = []
        current_budget = Fraction(0, 1)
        current_L = 0
        
        for _ in range(100):
            if time.time() > deadline - 5:
                break
                
            candidates = []
            active_p_set = {p for p, r in active_classes}
            
            for p in primes:
                if p in active_p_set:
                    continue
                cost = Fraction(1, p)
                if current_budget + cost > Fraction(1, 1):
                    continue
                
                r = (current_L + 1) % p
                test_classes = active_classes + [[p, r]]
                new_L = calculate_L(test_classes)
                candidates.append((new_L, p, r, cost))
            
            if not candidates:
                break
                
            # Sort by new_L descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick from top N
            top_n = min(num_random_choices, len(candidates))
            if random.random() < 0.5:
                # Deterministic pick (best)
                chosen = candidates[0]
            else:
                # Random pick from top N
                chosen = random.choice(candidates[:top_n])
            
            new_L, p, r, cost = chosen
            if new_L <= current_L:
                # No improvement, stop
                break
                
            active_classes.append([p, r])
            current_budget += cost
            current_L = new_L
            
        # Refine
        active_classes, current_L = local_refine(active_classes, current_budget, current_L)
        return active_classes, current_L, current_budget

    # Run a few randomized restarts
    for i in range(10):
        if time.time() > deadline - 10:
            break
        r_classes, r_L, r_budget = randomized_greedy()
        if r_L > best_L:
            best_classes = r_classes
            best_L = r_L
            best_budget = r_budget
            
    # Final Check
    final_budget = get_budget(best_classes)
    if final_budget > Fraction(1, 1):
        return []
        
    return best_classes

if __name__ == "__main__":
    classes = construct()
    # Verify
    L = 0
    for i in range(1, 10000):
        covered = False
        for p, r in classes:
            if i % p == r:
                covered = True
                break
        if covered:
            L = i
        else:
            break
    print(f"Covered up to L={L}")
    budget = sum(Fraction(1, p) for p, r in classes)
    print(f"Budget: {budget}")
