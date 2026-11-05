"""Combined program: conflict-aware greedy + improved local search with merge moves + B&B."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    N = 512
    
    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    
    # For each number 1..511, compute its set of prime factors
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if is_prime[p]:
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)
    
    # Score for each a: 1/(512-a)
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512 - a)
    
    # Precompute sorted candidates by score descending for fast lookup of best add
    candidates_by_score = sorted(range(511, 0, -1), key=lambda x: scores[x], reverse=True)
    
    def get_score(s):
        return sum(scores[a] for a in s)
    
    def valid(s):
        used = set()
        for a in s:
            if prime_factors[a] & used:
                return False
            used |= prime_factors[a]
        return True
    
    def find_best_add(used_primes, current_set, exclude=None):
        """Find the best number to add that is coprime to used_primes and not in current_set."""
        if exclude is None:
            exclude = set()
        for a in candidates_by_score:
            if a in current_set or a in exclude:
                continue
            if not (prime_factors[a] & used_primes):
                return a
        return None

    def hill_climb(initial_set, max_iters=5000):
        current = list(initial_set)
        current_score = get_score(current)
        current_set = set(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding any a not in current that is coprime
            used_primes = set()
            for a in current:
                used_primes |= prime_factors[a]
            
            best_add = find_best_add(used_primes, current_set)
            if best_add is not None:
                current.append(best_add)
                current_set.add(best_add)
                current_score += scores[best_add]
                improved = True
                continue

            # 2. Try 1-for-1 swap: remove one, add one
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                best_add = find_best_add(rem_used, rem_set)
                if best_add is not None:
                    new_score = rem_score + scores[best_add]
                    if new_score > current_score + 1e-15:
                        current = remaining + [best_add]
                        current_set = rem_set | {best_add}
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue

            # 3. Try 1-for-2 swap: remove one, add two
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                # Find best pair to add - search more thoroughly
                best_pair = None
                best_pair_score = 0
                
                # Get list of valid candidates
                valid_cands = []
                for a in candidates_by_score:
                    if a in rem_set:
                        continue
                    if prime_factors[a] & rem_used:
                        continue
                    valid_cands.append(a)
                    if len(valid_cands) > 80: # Limit search space for pairs
                        break
                
                for idx_a in range(len(valid_cands)):
                    a = valid_cands[idx_a]
                    for idx_b in range(idx_a + 1, len(valid_cands)):
                        b = valid_cands[idx_b]
                        if prime_factors[a] & prime_factors[b]:
                            continue
                        s = rem_score + scores[a] + scores[b]
                        if s > best_pair_score:
                            best_pair_score = s
                            best_pair = (a, b)
                
                if best_pair is not None and best_pair_score > current_score + 1e-15:
                    current = remaining + list(best_pair)
                    current_set = rem_set | set(best_pair)
                    current_score = best_pair_score
                    improved = True
                    break
            
            if improved:
                continue

            # 4. Try 2-for-1 swap: remove two, add one (MERGE) - try multiple candidates
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        break
                    removed1 = current[i]
                    removed2 = current[j]
                    remaining = [x for k, x in enumerate(current) if k != i and k != j]
                    rem_set = set(remaining)
                    rem_score = current_score - scores[removed1] - scores[removed2]
                    
                    rem_used = set()
                    for a in remaining:
                        rem_used |= prime_factors[a]
                    
                    # Try multiple add candidates, not just the best
                    best_new_score = 0
                    best_add_candidate = None
                    for a in candidates_by_score:
                        if a in rem_set:
                            continue
                        if prime_factors[a] & rem_used:
                            continue
                        new_score = rem_score + scores[a]
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_add_candidate = a
                        # Early termination if we've found a good enough candidate
                        if best_new_score > current_score + 1e-15:
                            break
                    
                    if best_add_candidate is not None and best_new_score > current_score + 1e-15:
                        current = remaining + [best_add_candidate]
                        current_set = rem_set | {best_add_candidate}
                        current_score = best_new_score
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue

            # 5. Try 2-for-2 swap: remove two, add two
            if len(current) >= 2:
                for i in range(len(current)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() > deadline:
                            break
                        removed1 = current[i]
                        removed2 = current[j]
                        remaining = [x for k, x in enumerate(current) if k != i and k != j]
                        rem_set = set(remaining)
                        rem_score = current_score - scores[removed1] - scores[removed2]
                        
                        rem_used = set()
                        for a in remaining:
                            rem_used |= prime_factors[a]
                        
                        # Find best pair to add
                        best_pair = None
                        best_pair_score = 0
                        
                        valid_cands = []
                        for a in candidates_by_score:
                            if a in rem_set:
                                continue
                            if prime_factors[a] & rem_used:
                                continue
                            valid_cands.append(a)
                            if len(valid_cands) > 60:
                                break
                        
                        for idx_a in range(len(valid_cands)):
                            a = valid_cands[idx_a]
                            for idx_b in range(idx_a + 1, len(valid_cands)):
                                b = valid_cands[idx_b]
                                if prime_factors[a] & prime_factors[b]:
                                    continue
                                s = rem_score + scores[a] + scores[b]
                                if s > best_pair_score:
                                    best_pair_score = s
                                    best_pair = (a, b)
                        
                        if best_pair is not None and best_pair_score > current_score + 1e-15:
                            current = remaining + list(best_pair)
                            current_set = rem_set | set(best_pair)
                            current_score = best_pair_score
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current, current_score
    
    def branch_and_bound(initial_set, time_budget=30.0):
        """
        Custom B&B exploiting sparse conflict graph.
        Upper bound: current_score + sum of scores of all candidates coprime to used_primes.
        This is valid because any feasible completion can only use a subset of those.
        """
        bb_start = time.time()
        best_score_bb = get_score(initial_set)
        best_set_bb = list(initial_set)
        
        # Work on a copy
        current = list(initial_set)
        current_score = best_score_bb
        current_set = set(current)
        
        # Precompute: for each candidate, which candidates conflict with it
        # conflict[a] = set of b such that gcd(a,b) > 1
        conflict = [set() for _ in range(512)]
        for a in range(1, 512):
            for b in range(a+1, 512):
                if prime_factors[a] & prime_factors[b]:
                    conflict[a].add(b)
                    conflict[b].add(a)
        
        # Fractional knapsack upper bound: sort available candidates by score desc,
        # take all that are coprime to used_primes AND coprime to each other in greedy order.
        # But simpler: just sum scores of all candidates coprime to used_primes (loose but fast).
        # Better: greedy fractional upper bound respecting conflicts.
        
        def compute_upper_bound(used_primes, excluded_set):
            """
            Upper bound: greedy fractional knapsack on candidates coprime to used_primes,
            respecting pairwise conflicts in a greedy manner.
            """
            ub = 0.0
            remaining_excluded = set(excluded_set)
            
            for a in candidates_by_score:
                if a in remaining_excluded:
                    continue
                if prime_factors[a] & used_primes:
                    continue
                # Check if a conflicts with any already "taken" in the UB
                # For simplicity, use a loose bound: just add the score
                # This is valid as an upper bound since we're not enforcing conflicts in the UB
                ub += scores[a]
            
            return ub
        
        # More precise UB: use a greedy independent set approximation
        def greedy_ub(used_primes, excluded_set):
            """
            Greedy: process candidates in score order, add if coprime to used_primes 
            and not conflicting with previously added in this UB computation.
            This gives a valid upper bound since we're computing a feasible independent set
            in the remaining graph, which is an upper bound on any feasible completion.
            """
            ub = 0.0
            taken = set()
            taken_primes = set(used_primes)
            for a in candidates_by_score:
                if a in excluded_set or a in taken:
                    continue
                if prime_factors[a] & taken_primes:
                    continue
                # Check conflict with taken using precomputed conflict sets
                if conflict[a] & taken:
                    continue
                ub += scores[a]
                taken.add(a)
                taken_primes |= prime_factors[a]
            return ub
        
        # B&B state: (current_set, current_score, used_primes, next_candidate_index)
        # We'll do a depth-first search with pruning.
        
        # Start with the initial set, and try to add more candidates.
        # Sort candidates by score descending.
        
        def dfs(used_primes, current_score, current_set, candidate_list, depth=0):
            nonlocal best_score_bb, best_set_bb
            
            if time.time() - bb_start > time_budget:
                return
            
            # Limit depth to prevent stack overflow and excessive branching
            if depth > 50:
                return
            
            # Upper bound: current_score + greedy_ub of remaining candidates
            ub = current_score + greedy_ub(used_primes, current_set)
            
            if ub <= best_score_bb + 1e-15:
                return
            
            # Filter candidates to only those coprime to used_primes
            valid_candidates = [a for a in candidate_list if a not in current_set and not (prime_factors[a] & used_primes)]
            
            if not valid_candidates:
                return
            
            # Try candidates in order of decreasing score (already sorted)
            for idx, a in enumerate(valid_candidates):
                if time.time() - bb_start > time_budget:
                    return
                
                new_used = used_primes | prime_factors[a]
                new_score = current_score + scores[a]
                new_set = current_set | {a}
                
                # Update best if better
                if new_score > best_score_bb + 1e-15:
                    best_score_bb = new_score
                    best_set_bb = list(new_set)
                
                # Prune: if even the upper bound from here can't beat best, skip
                remaining_after = valid_candidates[idx+1:]
                ub_after = new_score + greedy_ub(new_used, new_set)
                if ub_after <= best_score_bb + 1e-15:
                    continue
                
                # Recurse with remaining candidates after a
                dfs(new_used, new_score, new_set, remaining_after, depth + 1)
        
        # Start DFS from the initial set
        # Candidate list: all candidates not in initial_set, sorted by score desc
        initial_used = set()
        for a in initial_set:
            initial_used |= prime_factors[a]
        
        candidate_list = [a for a in candidates_by_score if a not in current_set]
        
        dfs(initial_used, current_score, current_set, candidate_list)
        
        return best_set_bb, best_score_bb
    
    # Initial strategy: Start with all primes
    initial_set = [p for p in range(2, 512) if is_prime[p]]
    
    # Hill climb from all primes
    best_set, best_score = hill_climb(initial_set, max_iters=3000)
    
    # Try B&B from the current best
    if time.time() < deadline - 50:
        bb_set, bb_score = branch_and_bound(best_set, time_budget=40.0)
        if bb_score > best_score + 1e-15:
            best_set = bb_set
            best_score = bb_score
    
    # Polish after B&B
    if time.time() < deadline - 20:
        result, score = hill_climb(best_set, max_iters=5000)
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
    
    # Random restarts with more aggressive perturbation
    rng = random.Random(42)
    
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        # Perturb best set more aggressively
        perturbed = best_set[:]
        k = rng.randint(1, min(5, len(perturbed)))
        for _ in range(k):
            if len(perturbed) > 1:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
        
        # Also try adding some random high-score elements
        if len(perturbed) < len(best_set) - 1:
            for a in candidates_by_score[:10]:
                if a not in perturbed:
                    used = set()
                    for x in perturbed:
                        used |= prime_factors[x]
                    if not (prime_factors[a] & used):
                        perturbed.append(a)
                        break
        
        result, score = hill_climb(perturbed, max_iters=1500)
        
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
    
    # Final polish with more iterations
    if time.time() < deadline - 5:
        result, score = hill_climb(best_set, max_iters=3000)
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
            
    return best_set
# EVOLVE-BLOCK-END
