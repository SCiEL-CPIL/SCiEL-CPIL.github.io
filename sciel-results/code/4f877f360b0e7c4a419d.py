"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def construct():
    # Deadline guard: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We want to maximize L such that every integer 1..L has coverage >= 2.
    # Let's aim for L = 200 initially, then increase if fast enough.
    # Actually, let's aim for a dynamic target.
    # The seed score 0.188 corresponds to L=84 (since 84/446 ~ 0.188? No, score is L. 
    # Wait, the task says "score is the largest L". 
    # The seed program returns residues for L=84? 
    # "The i-th prime covers the integer ceil(i/2)..." -> L=84.
    # Score 0.1882 might be normalized? 
    # "The evaluator measures the raw value (higher is better)."
    # "search score is value/seed".
    # If seed L=84, and my score is 0.188, then raw value is 0.188 * 84? No.
    # "score 0.1882" is the SEARCH SCORE. 
    # If seed is 1.0, and seed L=84, then raw value = 84? 
    # Then 84/84 = 1.0.
    # Current program score 0.1882 means raw value = 0.1882 * 84 = 15.8?
    # Or maybe the seed program in the prompt is the one with score 1.0 (inspiration 1).
    # Inspiration 1: "The i-th prime covers the integer ceil(i/2)..." -> L=84.
    # So Seed Raw Value = 84.
    # Current Program Raw Value = 0.1882 * 84 = 15.8 -> L=15?
    # This implies the current program (SA) is bad.
    
    # Strategy: Constraint Propagation / Backtracking
    # We assign residues r_p one by one.
    # Each assignment r_p = k affects the coverage of numbers x where x % p == k.
    # We want to ensure that for all x in 1..L, count(p: x % p == r_p) >= 2.
    
    # Let's try to build L incrementally.
    # Start with L=1. Try to extend to L+1.
    # For a fixed L, we have constraints on residues.
    # This is a CSP.
    
    # Variables: r_2, r_3, ..., r_997
    # Domains: 0..p-1
    # Constraints: For all x in 1..L, sum_{p} [x % p == r_p] >= 2.
    
    # We can solve this via backtracking.
    # Order variables by "impact" or "constrainedness".
    # Small primes have high impact (small period).
    
    # To make it efficient, we can use a "Greedy + Repair" or "Dancing Links" style?
    # Simpler: Backtracking with Forward Checking.
    
    # Let's aim for L = 100 first. If we solve it fast, we go higher.
    target_L = 100
    
    # Pre-compute for each x in 1..target_L, the list of primes p such that 
    # we can potentially cover x.
    # Actually, for a specific x, the primes that CAN cover it are ALL primes.
    # But the RESIDUE r_p is shared.
    
    # Let's formulate:
    # We assign r_p.
    # For each x, we track how many primes currently cover it (coverage[x]).
    # We need coverage[x] >= 2 for all x <= L.
    
    # Backtracking Search:
    # State: current assignment of r_p for first k primes.
    # Check: for all x <= L, is it still POSSIBLE to reach coverage 2?
    # A simple heuristic: 
    # If we have assigned k primes, the remaining primes can add at most (n_primes - k) to any x.
    # So if coverage[x] + (n_primes - k) < 2, we are stuck.
    
    # Better Heuristic:
    # For each unassigned prime p, it can cover at most 1 specific residue class mod p.
    # For a specific x, only 1 residue of p covers x.
    # So each unassigned prime can contribute at most 1 to coverage[x].
    # Thus, if coverage[x] + (n_primes - k) < 2, prune.
    
    # This is a very weak constraint.
    # Stronger: 
    # Group unassigned primes.
    # Actually, let's just use a simple backtracking with good variable ordering.
    # Variable ordering: Smallest primes first? Or most constrained?
    # Small primes cover many numbers in 1..L.
    
    # Let's try to find a solution for L=100.
    
    def solve_L(L):
        # Returns residues or None
        # n_primes = 168
        # We assign r for primes[0], primes[1], ...
        
        # To speed up, we can use symmetry breaking?
        # Not obvious.
        
        # We'll use a recursive backtracking.
        # To avoid recursion limit, we'll use an explicit stack or set recursion limit.
        sys.setrecursionlimit(10000)
        
        # Precompute: for each prime index i, and each possible residue r,
        # which numbers x in 1..L does it cover?
        # Actually, we just need to update coverage[x] incrementally.
        
        coverage = [0] * (L + 1)
        residues = [None] * n_primes
        
        # Order of assignment:
        # Let's try assigning small primes first.
        # Primes: 2, 3, 5, ...
        
        # Optimization:
        # If we fail to find a solution, we return None.
        
        def backtrack(k):
            if time.time() > deadline:
                return False
            
            if k == n_primes:
                # Check if all x in 1..L have coverage >= 2
                for x in range(1, L + 1):
                    if coverage[x] < 2:
                        return False
                return True
            
            p = primes[k]
            
            # Try all residues r in 0..p-1
            # Heuristic: Try residues that cover numbers with currently low coverage first?
            # Or just random?
            # Let's try residues in a random order to break symmetry.
            
            # Identify "critical" numbers: those with coverage < 2.
            # If a number x has coverage 0, we MUST cover it with some future prime.
            # If we assign r_p, we cover x if x % p == r.
            
            # Let's try all residues, but maybe prioritize those that help "tight" spots.
            
            # Simple loop:
            residues_to_try = list(range(p))
            random.shuffle(residues_to_try)
            
            for r in residues_to_try:
                # Assign r
                residues[k] = r
                
                # Update coverage
                # x = r, r+p, r+2p... <= L
                # If r == 0, x = p, 2p...
                
                start = r
                if start == 0:
                    start = p
                # Actually, if r==0, x=p is the first positive multiple.
                # If r>0, x=r is the first.
                
                # Apply change
                x = start
                while x <= L:
                    coverage[x] += 1
                    x += p
                
                # Forward Checking:
                # For all x in 1..L, check if it's still possible to reach coverage 2.
                # Remaining primes: n_primes - k - 1
                # Each remaining prime can add at most 1 to coverage[x].
                # So we need coverage[x] + (n_primes - k - 1) >= 2.
                
                valid = True
                remaining = n_primes - k - 1
                for x in range(1, L + 1):
                    if coverage[x] + remaining < 2:
                        valid = False
                        break
                
                if valid:
                    if backtrack(k + 1):
                        return True
                else:
                    pass # Pruned
                
                # Undo change
                x = start
                while x <= L:
                    coverage[x] -= 1
                    x += p
                
            return False
        
        return backtrack(0)
        
    # Try increasing L
    # Start with a known good L? 
    # The seed L=84.
    # Let's try L=84 first to ensure we don't crash.
    # Then try L=90, 100, etc.
    
    best_residues = None
    current_L = 84
    
    # Try to solve for current_L
    # Note: Backtracking on 168 variables is huge. 
    # 168! is impossible.
    # But constraints prune heavily?
    # "coverage[x] + remaining >= 2" is weak.
    
    # Alternative: Local Search / Hill Climbing on the CSP?
    # Or: Greedy Construction?
    
    # Let's try a different structural approach:
    # **Greedy with Lookahead**
    
    # 1. Initialize r_p randomly.
    # 2. Calculate coverage.
    # 3. Find first x with coverage < 2.
    # 4. Find a prime p that covers x (i.e., x % p == r_p is false, but we can change r_p to x % p).
    #    Wait, if we change r_p to x % p, p now covers x.
    #    But p might have stopped covering other numbers.
    # 5. This is basically the hill climbing in the current program, but we can be smarter.
    
    # Let's try **Min-Conflicts / Local Search** with a specific objective.
    # Objective: Minimize the number of "violations" (x with cov < 2) in 1..L.
    # Then maximize L.
    
    # Actually, the previous "Inspiration 2" (score 0.2118) used a greedy constraint satisfaction.
    # Let's try **Simulated Annealing** but with a much better initial state and a stronger move set.
    
    # Better Idea: **Exact Cover via DLX (Dancing Links)**?
    # Too complex to implement in 150s?
    
    # Let's try **Backtracking with Stronger Pruning**.
    # The weak pruning "cov + remaining >= 2" is not enough.
    # We can use: "For any subset of unassigned primes, can they cover the deficit?"
    # This is a Set Cover problem, which is NP-hard.
    
    # Let's stick to **Hill Climbing / SA** but optimize the evaluation and move generation.
    # The current SA is slow because it recalculates coverage from scratch.
    # We can update coverage incrementally.
    
    # Let's implement a fast incremental SA.
    
    MAX_L_TARGET = 200 # We want to see if we can reach this.
    # We will search for residues that maximize L.
    
    # Representation:
    # residues: list of 168 ints.
    # coverage: array of size MAX_L_TARGET + 1.
    
    def get_primes():
        return [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    
    primes = get_primes()
    n_primes = len(primes)
    
    # Precompute for each prime p, the list of x in 1..MAX_L_TARGET that are in residue class r mod p.
    # Actually, we just need to update x, x+p, x+2p...
    
    # Initialize residues
    # Heuristic initialization:
    # Try to make 1..10 covered by many primes.
    # Set r_p = 1 for all p? No.
    # Let's try the seed's strategy for initialization:
    # Seed: r_i = ceil(i/2) mod p.
    # Let's use that as the start.
    
    residues = [((i + 2) // 2) % primes[i] for i in range(n_primes)]
    
    # Calculate initial coverage
    coverage = [0] * (MAX_L_TARGET + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start == 0:
            start = p
        x = start
        while x <= MAX_L_TARGET:
            coverage[x] += 1
            x += p
            
    def get_L(cov):
        L = 0
        for i in range(1, MAX_L_TARGET + 1):
            if cov[i] >= 2:
                L = i
            else:
                break
        return L
        
    current_L = get_L(coverage)
    best_L = current_L
    best_residues = residues[:]
    
    # SA Parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.9995
    
    while time.time() < deadline and T > T_min:
        # Move: Change one residue
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            T *= alpha
            continue
            
        # Update coverage
        # Remove old_r contributions
        start_old = old_r
        if start_old == 0:
            start_old = p
        x = start_old
        while x <= MAX_L_TARGET:
            coverage[x] -= 1
            x += p
            
        # Add new_r contributions
        start_new = new_r
        if start_new == 0:
            start_new = p
        x = start_new
        while x <= MAX_L_TARGET:
            coverage[x] += 1
            x += p
            
        residues[idx] = new_r
        
        new_L = get_L(coverage)
        
        if new_L >= best_L:
            best_L = new_L
            best_residues = residues[:]
            
        # Acceptance
        delta = new_L - current_L
        if delta > 0:
            current_L = new_L
        else:
            prob = math.exp(delta / T)
            if random.random() < prob:
                current_L = new_L
            else:
                # Reject: Revert
                # Undo new_r, restore old_r
                x = start_new
                while x <= MAX_L_TARGET:
                    coverage[x] -= 1
                    x += p
                    
                x = start_old
                while x <= MAX_L_TARGET:
                    coverage[x] += 1
                    x += p
                    
                residues[idx] = old_r
                
        T *= alpha
        
    # Return best residues found
    # The task asks for 168 residues.
    # If best_L is small, it's fine.
    return best_residues

# EVOLVE-BLOCK-END
