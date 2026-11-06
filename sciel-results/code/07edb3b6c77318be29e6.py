"""Rewritten program using beam search + local search for pairwise coprime set."""
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 150.0  # seconds total
    SAFETY = 5.0      # safety margin
    
    N = 512
    # Precompute scores: 1/(N - a) for a in 1..N-1
    # Higher a gives higher score, so we prefer larger a's
    # But we need pairwise coprime
    
    # Precompute gcd checks efficiently
    # For each a, precompute which other values share a factor with it
    # Actually, let's just use gcd directly but cache it
    
    # Beam search approach:
    # State: (frozenset of chosen numbers, current score as Fraction)
    # We want to maximize sum of 1/(512-a)
    
    # Key insight: larger a values give larger 1/(512-a), so we prefer large a's
    # But we need pairwise coprime
    
    # Strategy:
    # 1. Start with greedy: pick largest available a that's coprime to all chosen
    # 2. Beam search: maintain multiple partial solutions, expand each
    # 3. Local search: try to improve by swapping
    
    # Precompute prime factorization for fast gcd-like checks
    # Actually, let's just use gcd but be smart about ordering
    
    # Greedy initialization (largest first)
    def greedy_largest():
        chosen = []
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    # Greedy initialization (smallest first) - different local optimum
    def greedy_smallest():
        chosen = []
        for a in range(1, 512):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    # Score function using Fraction for exactness
    def score(chosen):
        s = Fraction(0, 1)
        for a in chosen:
            s += Fraction(1, 512 - a)
        return s
    
    # Beam search
    def beam_search(initial_solutions, beam_width=20, max_depth=60):
        """
        Beam search over partial solutions.
        Each state is a tuple of chosen numbers (sorted).
        We expand by adding one more number at a time.
        """
        # Sort initial solutions by score descending
        initial_solutions = [tuple(sorted(s)) for s in initial_solutions]
        
        # Deduplicate
        seen = set()
        unique_initials = []
        for sol in initial_solutions:
            if sol not in seen:
                seen.add(sol)
                unique_initials.append(sol)
        initial_solutions = unique_solutions = unique_initials
        
        # Score each
        def sol_score(sol):
            s = Fraction(0, 1)
            for a in sol:
                s += Fraction(1, 512 - a)
            return s
        
        # Initial beam
        beam = []
        for sol in initial_solutions:
            beam.append((sol_score(sol), sol))
        beam.sort(key=lambda x: -x[0])
        beam = beam[:beam_width]
        
        best_score, best_sol = beam[0]
        
        for depth in range(max_depth):
            if time.time() - start_time > DEADLINE - SAFETY:
                break
            
            next_beam = []
            seen_next = set()
            
            for cur_score, sol in beam:
                # Try adding each candidate a not in sol
                sol_set = set(sol)
                for a in range(511, 0, -1):
                    if a in sol_set:
                        continue
                    if time.time() - start_time > DEADLINE - SAFETY:
                        break
                    # Check coprime with all in sol
                    compatible = True
                    for b in sol:
                        if gcd(a, b) != 1:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    new_sol = tuple(sorted(sol + (a,)))
                    if new_sol in seen_next:
                        continue
                    seen_next.add(new_sol)
                    new_score = cur_score + Fraction(1, 512 - a)
                    next_beam.append((new_score, new_sol))
            
            if not next_beam:
                break
            
            next_beam.sort(key=lambda x: -x[0])
            next_beam = next_beam[:beam_width]
            
            if next_beam[0][0] > best_score:
                best_score, best_sol = next_beam[0]
            
            beam = next_beam
            
            # If beam has converged (all same length or no improvement), break
            lengths = set(len(s) for _, s in beam)
            if len(lengths) == 1 and depth > 5:
                # Check if we can still improve
                max_len = max(lengths)
                if depth > max_len * 2:
                    break
        
        return best_sol, best_score
    
    # Local search: try swaps to improve
    def local_search(sol, max_iters=5000):
        sol = list(sol)
        sol_set = set(sol)
        
        def sol_score_fn(lst):
            s = Fraction(0, 1)
            for a in lst:
                s += Fraction(1, 512 - a)
            return s
        
        cur_score = sol_score_fn(sol)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() - start_time > DEADLINE - SAFETY:
                break
            
            # Try: remove one, add one different
            for i, a in enumerate(sol):
                if time.time() - start_time > DEADLINE - SAFETY:
                    break
                remaining = sol[:i] + sol[i+1:]
                rem_set = set(remaining)
                
                # Try adding each candidate not in sol
                for b in range(511, 0, -1):
                    if b in sol_set:
                        continue
                    # Check b is coprime with all in remaining
                    compatible = True
                    for c in remaining:
                        if gcd(b, c) != 1:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    new_sol = remaining + [b]
                    new_score = cur_score - Fraction(1, 512 - a) + Fraction(1, 512 - b)
                    if new_score > cur_score:
                        sol = new_sol
                        sol_set = set(sol)
                        cur_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        return sol, cur_score
    
    # Run greedy initializations
    sol1 = greedy_largest()
    sol2 = greedy_smallest()
    
    # Also try a few random-ish initializations
    import random
    random.seed(42)
    
    def greedy_random_order():
        chosen = []
        candidates = list(range(1, 512))
        random.shuffle(candidates)
        for a in candidates:
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    sol3 = greedy_random_order()
    sol4 = greedy_random_order()
    sol5 = greedy_random_order()
    
    initial_solutions = [sol1, sol2, sol3, sol4, sol5]
    
    # Beam search
    best_sol, best_score = beam_search(initial_solutions, beam_width=15, max_depth=80)
    
    # Local search polish
    best_sol, best_score = local_search(best_sol, max_iters=3000)
    
    # Convert to list
    return list(best_sol)

# Quick test
if __name__ == "__main__":
    result = construct()
    s = sum(1.0 / (512 - a) for a in result)
    print(f"Score: {s}")
    print(f"Count: {len(result)}")
    # Verify pairwise coprime
    from math import gcd
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert gcd(result[i], result[j]) == 1, f"Not coprime: {result[i]}, {result[j]}"
    print("Valid: pairwise coprime, all in 1..511, distinct")
