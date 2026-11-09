import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target_size = 12
    max_val = 10000
    
    # 1. Precompute prime factors for all sums (2 to 20000)
    # This allows O(1) lookup of prime factors for any sum.
    sum_primes = [set() for _ in range(2 * max_val + 1)]
    for p in range(2, 2 * max_val + 1):
        # Check if p is prime
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            for multiple in range(p, 2 * max_val + 1, p):
                sum_primes[multiple].add(p)
                
    def evaluate_set(A):
        """Calculate the union of prime factors of all pairwise sums."""
        all_primes = set()
        for i in range(target_size):
            for j in range(i + 1, target_size):
                s = A[i] + A[j]
                all_primes.update(sum_primes[s])
        return all_primes

    # 2. Strategy: 
    # First, try to find a set where all sums are 7-smooth (primes {2,3,5,7}).
    # If successful, the answer is exactly 4.
    # If not, use local search to minimize the count.
    
    # --- Phase 1: Clique Search for 7-smooth sums ---
    # We want 12 distinct integers a_1..a_12 <= 10000 such that a_i + a_j is 7-smooth for all i!=j.
    # 7-smooth numbers are of the form 2^a * 3^b * 5^c * 7^d.
    # Let's generate all 7-smooth numbers up to 20000.
    smooth_7 = set()
    for a in range(15): # 2^14 > 16384, 2^15 > 32768
        p2 = 2**a
        if p2 > 20000: break
        for b in range(10): # 3^9 > 19683
            p3 = 3**b
            if p2 * p3 > 20000: break
            for c in range(7): # 5^6 > 15625
                p5 = 5**c
                if p2 * p3 * p5 > 20000: break
                for d in range(5): # 7^4 = 2401, 7^5 > 16807
                    p7 = 7**d
                    val = p2 * p3 * p5 * p7
                    if val > 20000: break
                    smooth_7.add(val)
    
    # Build graph: nodes are 1..10000. Edge (u,v) if u+v is in smooth_7.
    # To save memory/time, we only need to check if a candidate x can be added to a set S.
    # We will use a backtracking search with forward checking.
    # Heuristic: Order candidates by degree (number of smooth sums) or just iterate.
    # Since 10000 is large, we should restrict candidates.
    # Numbers that are "central" to smooth sums are better.
    # Let's just pick candidates randomly or sorted by value?
    # Actually, let's try a greedy construction first.
    
    def find_7_smooth_set():
        if time.time() > deadline - 5.0:
            return None
            
        # We want to find 12 numbers.
        # Greedy:
        # 1. Pick a random start.
        # 2. Pick next number that is smooth-sum compatible with all previous.
        # 3. If stuck, backtrack or restart.
        
        # To make it faster, precompute for each number x, the set of y such that x+y is smooth_7.
        # This is too big (10000 * 10000).
        # Instead, for a given x, we can generate potential y's by checking s in smooth_7: y = s - x.
        
        def get_compatible(x, current_set):
            compatible = []
            for s in smooth_7:
                y = s - x
                if 1 <= y <= max_val and y not in current_set:
                    # Check if y is compatible with all in current_set
                    ok = True
                    for cx in current_set:
                        if (y + cx) not in smooth_7:
                            ok = False
                            break
                    if ok:
                        compatible.append(y)
            return compatible

        # Randomized Greedy with Restarts
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
                
            current_set = []
            # Start with a random number? Or a small one?
            # Small numbers might have more small smooth sums?
            # Let's try starting with 1, 2, 3... or random.
            # Let's try random start from 1..10000
            start = random.randint(1, max_val)
            current_set.append(start)
            
            success = True
            for _ in range(11):
                candidates = get_compatible(current_set[-1], current_set)
                # Wait, get_compatible checks against current_set, but we need to check against ALL in current_set.
                # My helper above does check against all in current_set.
                # But I passed current_set[-1] as x? No, I need to pick a NEW x.
                # Let's pick a candidate y from the universe such that y is compatible with ALL in current_set.
                
                # Better approach:
                # Maintain a set of "valid candidates".
                # Initially all 1..10000.
                # When we add x, we filter valid_candidates to those y such that x+y is smooth.
                
                pass # The logic above is slightly off for efficiency. Let's use a simpler backtracking.
                
            # Let's implement a proper backtracking with bitsets or sets.
            
        # Alternative: Use the structure of 7-smooth numbers.
        # If we pick numbers that are multiples of 2,3,5,7?
        # If a_i are 7-smooth numbers themselves?
        # Sum of two 7-smooth numbers is not necessarily 7-smooth.
        
        # Let's try a specific known set or a very constrained search.
        # Since 4 is the target, and 2,3,5,7 are small, maybe we can find a set.
        # Let's try to find a clique in the graph of 7-smooth sums.
        
        # Given the complexity, if we don't find it quickly, we fall back to local search.
        return None

    # --- Phase 2: Local Search ---
    # This is the robust fallback and likely the main solver.
    
    best_set = None
    best_score = float('inf')
    
    # Initial sets
    strategies = [
        lambda: list(range(1, 13)),
        lambda: [i * 100 for i in range(1, 13)],
        lambda: random.sample(range(1, max_val + 1), target_size),
        lambda: random.sample(range(1, 1000), target_size),
    ]
    
    # Precompute prime factors for initial evaluation speed?
    # We already have sum_primes.
    
    def get_initial_set(strategy_idx):
        if strategy_idx == 0:
            return list(range(1, 13))
        elif strategy_idx == 1:
            return [i * 100 for i in range(1, 13)]
        elif strategy_idx == 2:
            return random.sample(range(1, max_val + 1), target_size)
        else:
            return random.sample(range(1, 1000), target_size)
    
    strategy_idx = 0
    
    while time.time() < deadline - 2.0:
        A = get_initial_set(strategy_idx % len(strategies))
        strategy_idx += 1
        
        # Calculate initial score and counts
        counts = {}
        active = set()
        for i in range(target_size):
            for j in range(i + 1, target_size):
                s = A[i] + A[j]
                for p in sum_primes[s]:
                    if p not in counts:
                        counts[p] = 0
                        active.add(p)
                    counts[p] += 1
        
        current_score = len(active)
        
        if current_score < best_score:
            best_score = current_score
            best_set = A[:]
            if best_score <= 4:
                break # Optimal found
        
        no_improve_steps = 0
        max_no_improve = 5000
        
        # Local Search Loop
        while time.time() < deadline - 2.0:
            if no_improve_steps > max_no_improve:
                break
                
            i = random.randint(0, target_size - 1)
            x = random.randint(1, max_val)
            
            if x == A[i] or x in A:
                continue
                
            # Calculate delta
            new_counts = counts.copy()
            new_active = set(active)
            
            # Remove contributions of A[i]
            for j in range(target_size):
                if i == j: continue
                s_old = A[i] + A[j]
                for p in sum_primes[s_old]:
                    new_counts[p] -= 1
                    if new_counts[p] == 0:
                        new_active.discard(p)
                        
            # Add contributions of x
            for j in range(target_size):
                if i == j: continue
                s_new = x + A[j]
                for p in sum_primes[s_new]:
                    if p not in new_counts:
                        new_counts[p] = 0
                        new_active.add(p)
                    new_counts[p] += 1
                    
            new_score = len(new_active)
            
            if new_score < current_score:
                # Accept
                A[i] = x
                counts = new_counts
                active = new_active
                current_score = new_score
                no_improve_steps = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_set = A[:]
                    if best_score <= 4:
                        break
            else:
                # Simulated Annealing style: accept equal or slightly worse with prob
                if new_score == current_score:
                    if random.random() < 0.5:
                        A[i] = x
                        counts = new_counts
                        active = new_active
                else:
                    no_improve_steps += 1
                    
        if best_score <= 4:
            break

    if best_set is None:
        return list(range(1, 13))
        
    return best_set
