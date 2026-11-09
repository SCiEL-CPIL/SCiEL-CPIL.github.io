import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We aim to construct a valid prefix. 
    # The score is the largest L such that 1..L is doubly covered.
    # We will try to build this prefix incrementally using a matching-based constraint solver.
    
    # Strategy:
    # 1. Maintain a list of "slots" for the current prefix [1, L].
    #    Each integer x in [1, L] requires 2 covers.
    #    So we have 2*L slots.
    # 2. We assign primes to these slots. A prime p can cover slot (x, k) if x % p == r_p.
    #    Wait, the residue r_p is the variable.
    #    If we assign prime p to cover x, then r_p must be x % p.
    #    If prime p is assigned to cover x1 and x2, then x1 % p must equal x2 % p.
    #    
    # This is equivalent to:
    # We want to find residues r_p such that for each x in 1..L, count_p(x % p == r_p) >= 2.
    #
    # Let's reformulate as a bipartite matching problem for a fixed L.
    # Left nodes: The 2*L "coverage requirements" (2 for each x in 1..L).
    # Right nodes: The n_primes primes.
    # Edge (x_req, p) exists if we can set r_p to satisfy x. 
    # But r_p is global. If p covers x1, r_p is fixed to x1%p.
    # If p also covers x2, we must have x2%p == x1%p.
    #
    # This is a "List Coloring" or "Constraint Satisfaction" problem.
    # A simpler approach: 
    # Iterate L from 1 upwards.
    # Maintain current residues.
    # When adding L+1, we need to ensure it gets 2 covers.
    # If it doesn't, we need to modify residues to add covers for L+1 without breaking 1..L.
    #
    # The "Matching" idea from inspiration 1:
    # For a target L, can we assign residues?
    # This is NP-hard in general, but for small L (up to ~1000-2000) and 168 primes,
    # we can use a heuristic repair that is guided by matching logic.
    
    # Let's try a different structural approach:
    # **Simulated Annealing with a smarter move set and a "Conflict Graph" repair.**
    # The current program uses random kicks. 
    # New Strategy: **Greedy Construction with Backtracking-Style Repair via Matching.**
    
    # Actually, let's look at the "Maximum Bipartite Matching" for *deficits*.
    # If we have a current configuration with score L, and we want to reach L+1.
    # Let x = L+1.
    # Current coverage of x is c_x.
    # If c_x >= 2, we are good, move to x+1.
    # If c_x < 2, we need to "steal" coverage or "add" coverage.
    #
    # Let's try a **Deterministic Greedy Construction** first, then optimize.
    # Start with L=1.
    # For each x from 1 to MAX_LIMIT:
    #   We want x to be covered by at least 2 primes.
    #   We pick 2 primes p1, p2 that are "least constrained" or "most flexible" for x.
    #   Set r_{p1} = x % p1, r_{p2} = x % p2.
    #   If this breaks a previous y < x (i.e., y was covered by p1 or p2 and now isn't),
    #   we need to fix y.
    #
    # This is essentially the same as the current repair loop, but the order of prime selection
    # can be smarter.
    
    # Let's try **Local Search with a "Coverage Matrix" cache and efficient delta updates**.
    # The current program recomputes coverage from scratch (O(N*M)) every step.
    # We can update coverage in O(M) or O(N) if we track which primes cover which x.
    #
    # New Structural Change: **Iterative Deepening A* / Best-First Search on the Prefix Length**
    # No, that's too slow.
    
    # Let's try **Greedy Set Cover with Conflict Resolution**.
    # 1. Initialize residues randomly.
    # 2. Compute coverage. Find first failure L.
    # 3. Identify "Critical Primes" for the prefix 1..L.
    #    A prime p is critical for x if x is covered by exactly 2 primes, and p is one of them.
    #    If we change p, x might fail.
    # 4. To fix L+1, we must change a prime p that does NOT cover L+1.
    #    We prefer primes that are NOT critical for any x in 1..L.
    #    If all candidates are critical, we pick the one that is critical for the fewest x's.
    # 5. Change the residue.
    # 6. Check if any x in 1..L drops to coverage < 2.
    #    If yes, we have a conflict. 
    #    We can try to "repair" the conflict by changing another prime to cover the broken x.
    #    This is a 2-step repair.
    
    # Let's implement this "Smart Greedy with 2-Step Repair".
    
    MAX_EVAL = 20000
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def compute_coverage(residues):
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    def evaluate(residues):
        coverage = compute_coverage(residues)
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        first_fail_idx = np.argmax(mask)
        return int(xs[first_fail_idx] - 1)

    # Initialize
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    best_residues = residues.copy()
    best_score = evaluate(residues)
    
    # Precompute prime indices for quick access
    prime_indices = {p: i for i, p in enumerate(primes)}
    
    # Main Loop
    # We will use a "Time-Budgeted" search.
    # Phase 1: Greedy Repair
    # Phase 2: Local Optimization (Hill Climbing) on the best found so far.
    
    current_score = best_score
    residues = best_residues.copy()
    
    # Data structure for fast criticality check
    # coverage[x] = count
    # critical_primes[x] = set of primes p such that x is covered by exactly 2 primes, and p is one of them.
    # We don't need the full set, just the count of critical x's for each prime.
    
    def get_critical_counts(coverage):
        # Returns an array of length n_primes, where critical_counts[i] is the number of x
        # in 1..L such that coverage[x] == 2 and prime i covers x.
        # We only care about x in 1..current_score.
        critical_counts = np.zeros(n_primes, dtype=np.int64)
        # Find x where coverage is exactly 2
        # This is a bit expensive to do from scratch every time.
        # Let's just use the coverage array.
        # For each x in 1..L with coverage[x]==2, increment critical_counts for the 2 primes covering it.
        # We need to know WHICH primes cover x.
        # We can recompute hits for the prefix 1..L.
        
        L = current_score
        if L == 0:
            return critical_counts
        
        prefix_xs = xs[:L]
        diff = prefix_xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        
        # hits is (n_primes, L)
        # coverage_prefix = np.sum(hits, axis=0)
        # We know coverage_prefix == coverage[:L]
        
        # Find columns where sum is 2
        # And for those columns, add 1 to the rows that are 1.
        # This can be done with matrix multiplication or boolean ops.
        # mask_two = (np.sum(hits, axis=0) == 2)
        # critical_counts += np.sum(hits * mask_two[None, :], axis=1)
        
        # Efficient version:
        # We want to sum hits[i, x] for all x where sum(hits[:, x]) == 2.
        # Let M be the matrix of hits.
        # Let S be the vector of sums.
        # We want C_i = sum_x M_{i,x} * [S_x == 2]
        
        sums = np.sum(hits, axis=0)
        mask_two = (sums == 2)
        # Multiply rows by mask and sum
        critical_counts = np.sum(hits * mask_two[None, :], axis=1)
        
        return critical_counts

    stuck_count = 0
    MAX_STUCK = 20
    
    while time.time() < deadline:
        # Update current score
        current_score = evaluate(residues)
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            stuck_count = 0
            
        if current_score >= MAX_EVAL:
            break
            
        target_x = current_score + 1
        
        # Find candidates for change: primes that do NOT cover target_x
        candidates = []
        for i, p in enumerate(primes):
            if (target_x - residues[i]) % p != 0:
                candidates.append(i)
        
        if not candidates:
            break
            
        # Score candidates based on criticality
        # We want to change a prime that is "safe" (low criticality)
        # Compute critical counts for the current prefix
        critical_counts = get_critical_counts(compute_coverage(residues))
        
        # Sort candidates by critical_counts[i] (ascending)
        # If critical_counts[i] is 0, changing it is "safe" (won't break any x with coverage 2)
        # Note: Changing a prime might break an x with coverage 3 (making it 2, which is fine)
        #       or coverage 2 (making it 1, which is bad).
        #       So critical_counts[i] counts the "bad" breaks.
        
        candidates.sort(key=lambda i: critical_counts[i])
        
        found_repair = False
        # Try the top candidates
        for idx in candidates:
            old_res = residues[idx]
            new_res = target_x % primes[idx]
            
            if old_res == new_res:
                continue
                
            residues[idx] = new_res
            
            # Check if prefix 1..current_score is still valid
            new_score = evaluate(residues)
            
            if new_score >= current_score:
                found_repair = True
                break
            else:
                residues[idx] = old_res
                
        if found_repair:
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > MAX_STUCK:
                # Try a 2-step repair:
                # Change prime A to cover target_x.
                # This breaks some x_bad.
                # Change prime B to cover x_bad.
                # Check if this fixes everything.
                
                # Pick a candidate A (low criticality)
                if candidates:
                    A = candidates[0]
                    old_res_A = residues[A]
                    new_res_A = target_x % primes[A]
                    residues[A] = new_res_A
                    
                    # Find what broke
                    coverage_after_A = compute_coverage(residues)
                    # Find first x in 1..current_score where coverage < 2
                    mask_broken = coverage_after_A[:current_score] < 2
                    if np.any(mask_broken):
                        broken_idx = np.argmax(mask_broken)
                        x_bad = int(xs[broken_idx])
                        
                        # Find a prime B that can cover x_bad
                        # B must not be A.
                        # B should not cover x_bad currently (or we can change it to cover x_bad)
                        # Actually, if B covers x_bad currently, coverage[x_bad] would be >= 1.
                        # We need to ADD a cover. So B must NOT cover x_bad currently.
                        # Wait, if coverage[x_bad] is 1, it is covered by one prime.
                        # We need to add a second one.
                        # So we need a prime B such that (x_bad - r_B) % p_B != 0.
                        # And after changing r_B to x_bad % p_B, we must not break anything else.
                        
                        # Let's try all primes B != A
                        found_B = False
                        for B in range(n_primes):
                            if B == A:
                                continue
                            if (x_bad - residues[B]) % primes[B] == 0:
                                continue # B already covers x_bad, changing it won't increase coverage (unless we change to a different residue that also covers it? No, residue is unique mod p).
                            
                            old_res_B = residues[B]
                            new_res_B = x_bad % primes[B]
                            residues[B] = new_res_B
                            
                            # Check if this fixes x_bad and doesn't break others
                            # Specifically, check prefix 1..current_score
                            new_score_2 = evaluate(residues)
                            if new_score_2 >= current_score:
                                found_B = True
                                found_repair = True
                                break
                            else:
                                residues[B] = old_res_B
                        
                        if not found_B:
                            # Revert A
                            residues[A] = old_res_A
                    else:
                        # A didn't break anything? Then why did the single step fail?
                        # Maybe evaluate() was slow or there's a bug.
                        # If A didn't break anything, the single step should have worked.
                        # So this case shouldn't happen if logic is correct.
                        # Revert A to be safe? No, if it didn't break, keep it.
                        # Actually, if single step failed, it means new_score < current_score.
                        # That implies something broke. So mask_broken should be non-empty.
                        pass
            
            if not found_repair:
                # Random kick
                kick_size = random.randint(3, 10)
                kick_indices = random.sample(range(n_primes), kick_size)
                for i in kick_indices:
                    residues[i] = random.randint(0, primes[i] - 1)
                stuck_count = 0

    # Final Polish: Local Optimization
    # Try to improve the best_residues by hill climbing
    if time.time() < deadline - 5.0:
        residues = best_residues.copy()
        current_score = evaluate(residues)
        
        # Try changing each prime to all possible residues
        # This is O(N * P * Eval). N=168, P=1000, Eval=1ms. 168 * 1000 * 1ms = 168s. Too slow.
        # We need a faster evaluation or fewer tries.
        
        # Let's try a few random restarts of the hill climbing
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            # Random perturbation
            for _ in range(5):
                i = random.randint(0, n_primes - 1)
                residues[i] = random.randint(0, primes[i] - 1)
            
            # Hill climb
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(n_primes):
                    if time.time() > deadline - 5.0:
                        break
                    old_res = residues[i]
                    # Try a few random residues
                    for _ in range(5):
                        new_res = random.randint(0, primes[i] - 1)
                        if new_res == old_res:
                            continue
                        residues[i] = new_res
                        new_score = evaluate(residues)
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            break
                        elif new_score < current_score:
                            residues[i] = old_res
                        else:
                            # Keep if equal? Maybe explore
                            pass
                    if improved:
                        break # Restart inner loop if improved
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()

    return best_residues.tolist()
