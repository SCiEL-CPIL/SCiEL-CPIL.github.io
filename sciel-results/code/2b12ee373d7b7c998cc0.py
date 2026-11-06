import random
import math
import time
from fractions import Fraction

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # We want to maximize L such that 1..L is covered.
    # Constraint: sum(1/p) <= 1.
    
    # State: dictionary mapping prime p to residue r.
    # Initial state: Greedy selection.
    
    def get_coverage(state, max_L=10000):
        """
        Returns the largest L such that 1..L is covered by the state.
        state: dict {p: r}
        """
        if not state:
            return 0
        covered = [False] * (max_L + 1)
        for p, r in state.items():
            start = r if r > 0 else p
            for n in range(start, max_L + 1, p):
                covered[n] = True
        
        L = 0
        for n in range(1, max_L + 1):
            if covered[n]:
                L = n
            else:
                break
        return L

    def get_budget_used(state):
        return sum(Fraction(1, p) for p in state.keys())

    # 1. Greedy Initialization
    # We want to cover small numbers first.
    # Start with no primes.
    state = {}
    current_budget = Fraction(0)
    max_L_target = 10000
    
    # Greedy: At each step, find the prime p and residue r that covers the most new numbers
    # in the range 1..current_L + some_margin, with the lowest cost 1/p.
    # This is expensive to do perfectly. 
    # Simpler Greedy: 
    # Iterate through primes. For each prime, try all residues.
    # Pick the one that extends L the most.
    
    # Actually, a known good starting point for this problem is using primes >= 13 
    # and optimizing residues.
    # Let's start with a random set of primes that fits the budget, then optimize.
    
    # Better Init: 
    # Use the fact that small primes are "expensive" (1/p is large) but cover frequently.
    # Large primes are "cheap" but cover sparsely.
    # To cover 1..L, we need a mix.
    
    # Let's use a simple greedy:
    # Start with empty state.
    # While budget allows:
    #   For each unused prime p:
    #     For each residue r:
    #       Calculate new L if we add (p, r).
    #       Keep the one that maximizes L.
    #   Add the best (p, r) if it improves L.
    #   Stop if no improvement.
    
    # This is O(N_primes * N_residues * L) which is too slow for N_primes=168, N_residues=avg 500, L=10000.
    # 168 * 500 * 10000 = 8.4 * 10^8 operations. Too slow.
    
    # Faster Init:
    # Use a random set of primes with sum(1/p) <= 1.
    # Then use Simulated Annealing to optimize residues and set.
    
    # Let's try a specific heuristic:
    # Use primes 13, 17, 19, 23, ...
    # The sum of reciprocals of primes >= 13 is roughly 0.4-0.5?
    # Sum(1/p for p in primes) ~ log(log(1000)) ~ 2.
    # Sum(1/p for p >= 13) is smaller.
    
    # Let's just pick a random subset of primes that fits the budget.
    random.seed(42)
    available_primes = primes[:]
    random.shuffle(available_primes)
    
    state = {}
    budget = Fraction(0)
    for p in available_primes:
        cost = Fraction(1, p)
        if budget + cost <= Fraction(1):
            # Pick a random residue
            r = random.randint(0, p-1)
            state[p] = r
            budget += cost
        else:
            # Try to swap? No, just skip for now.
            pass
            
    # If we didn't get enough primes, we might have a small L.
    # Let's ensure we have a decent number of primes.
    # If L is small, we need more primes.
    
    # Let's try a different init: 
    # Use all primes >= 13.
    P_init = [p for p in primes if p >= 13]
    sum_recip = sum(Fraction(1, p) for p in P_init)
    if sum_recip <= Fraction(1):
        for p in P_init:
            state[p] = random.randint(0, p-1)
    else:
        # Remove largest primes until budget is met? 
        # Or smallest? 
        # Removing largest primes saves little budget. Removing smallest saves a lot but loses coverage density.
        # We want to keep small primes for coverage density.
        # So remove largest primes first.
        P_init_sorted = sorted(P_init, reverse=True)
        P_final = []
        budget = Fraction(0)
        for p in P_init:
            cost = Fraction(1, p)
            if budget + cost <= Fraction(1):
                P_final.append(p)
                budget += cost
        for p in P_final:
            state[p] = random.randint(0, p-1)

    current_L = get_coverage(state, 10000)
    
    # 2. Simulated Annealing
    # Parameters
    start_temp = 10.0
    end_temp = 0.01
    alpha = 0.99
    steps = 50000 # Adjust based on time limit
    
    # Precompute primes list for random selection
    all_primes = primes
    
    t = start_temp
    best_state = state.copy()
    best_L = current_L
    
    start_time = time.time()
    max_time = 110 # Leave 10s for overhead
    
    while time.time() - start_time < max_time:
        # Generate a neighbor
        new_state = state.copy()
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Change residue of an existing prime
            if state:
                p = random.choice(list(state.keys()))
                old_r = state[p]
                new_r = random.randint(0, p-1)
                if new_r != old_r:
                    new_state[p] = new_r
            else:
                continue
        elif move_type < 0.85:
            # Add a new prime
            # Find a prime not in state that fits in budget
            used_primes = set(state.keys())
            available = [p for p in all_primes if p not in used_primes]
            if available:
                # Sort by cost (1/p) ascending? No, random.
                p = random.choice(available)
                cost = Fraction(1, p)
                current_budget = get_budget_used(new_state)
                if current_budget + cost <= Fraction(1):
                    r = random.randint(0, p-1)
                    new_state[p] = r
                else:
                    # Can't add, try removing one to make space? 
                    # For now, skip.
                    continue
            else:
                continue
        else:
            # Remove a prime
            if state:
                p = random.choice(list(state.keys()))
                del new_state[p]
        
        # Evaluate new state
        new_L = get_coverage(new_state, 10000)
        
        # Acceptance criterion
        delta_L = new_L - current_L
        
        if delta_L >= 0:
            # Accept
            state = new_state
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_state = state.copy()
        else:
            # Worse move
            # Probability of acceptance: exp(delta_L / T)
            # delta_L is negative.
            # We need a scaling factor. L is around 1000.
            # Let's use exp(delta_L * 0.01 / T)
            prob = math.exp((delta_L * 0.01) / t)
            if random.random() < prob:
                state = new_state
                current_L = new_L
        
        # Cool down
        t *= alpha
        if t < end_temp:
            t = end_temp
            # If cooled down, maybe restart with higher temp?
            # Or just stop.
            # Let's do a few restarts.
            # Break inner loop to restart?
            # Simpler: just keep going with low temp.
            
    # Final check
    if best_L > current_L:
        state = best_state
    
    # Return classes
    return [[p, state[p]] for p in state.keys()]
