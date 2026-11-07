import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    N = 256
    
    # Precompute prime factors for 1..N
    # F[x] is a frozenset of prime factors of x
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        # Simple factorization
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)
    
    # Precompute gcd matrix? No, just use F intersection.
    # gcd(x, y) == 1 iff F[x].isdisjoint(F[y])
    
    def greedy_initialization(descending=True, randomize=False):
        candidates = list(range(1, N + 1))
        if descending:
            candidates.sort(reverse=True)
        else:
            candidates.sort()
        if randomize:
            random.shuffle(candidates)
        
        selected = []
        selected_factors = set()
        
        for a in candidates:
            if time.time() > deadline:
                break
            fa = F[a]
            # Check if disjoint from all selected factors
            # Since we only need pairwise coprime, we can just check against the union of factors?
            # No, if a shares factor 2 with b, and c shares factor 3 with b, a and c might be coprime.
            # But we need a to be coprime to ALL selected.
            # So a must be disjoint from union of F[b] for all b in selected?
            # Yes. If a shares factor p with any b, then gcd(a,b) > 1.
            # So checking against the union of prime factors of selected numbers is sufficient.
            if selected_factors.isdisjoint(fa):
                selected.append(a)
                selected_factors.update(fa)
        
        return selected

    def solve_subproblem(candidates):
        """
        Find max sum subset of candidates that is pairwise coprime.
        candidates: list of ints
        Returns: (max_sum, subset_list)
        """
        if not candidates:
            return 0, []
        
        # Sort descending to find high sums early
        cands = sorted(candidates, reverse=True)
        n = len(cands)
        
        # Prefix sums for pruning
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + cands[i]
            
        best_sum = 0
        best_subset = []
        
        def dfs(idx, current_sum, used_primes, current_subset):
            nonlocal best_sum, best_subset
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_subset = current_subset[:]
                
            if idx == n:
                return
                
            # Pruning: even if we take all remaining, we can't beat best
            if current_sum + suffix_sums[idx] <= best_sum:
                return
            
            # Option 1: Exclude cands[idx]
            dfs(idx + 1, current_sum, used_primes, current_subset)
            
            # Option 2: Include cands[idx] if coprime to used_primes
            fa = F[cands[idx]]
            if used_primes.isdisjoint(fa):
                current_subset.append(cands[idx])
                new_used = used_primes.union(fa)
                dfs(idx + 1, current_sum + cands[idx], new_used, current_subset)
                current_subset.pop()
                
        dfs(0, 0, set(), [])
        return best_sum, best_subset

    def local_search(S):
        """
        Perform 1-exchange local search.
        S: list of ints
        """
        S_set = set(S)
        S_factors = set()
        for x in S_set:
            S_factors.update(F[x])
            
        improved = True
        while improved:
            improved = False
            for a in list(S_set):
                if time.time() > deadline:
                    return S_set
                
                # Remove a
                rest = S_set - {a}
                
                # Calculate factors of rest
                # We can update S_factors efficiently, but for simplicity and safety:
                rest_factors = set()
                for x in rest:
                    rest_factors.update(F[x])
                
                # Find candidates: x not in rest, coprime to all in rest
                # x is coprime to rest iff F[x].isdisjoint(rest_factors)
                candidates = []
                for x in range(1, N + 1):
                    if x in rest:
                        continue
                    if F[x].isdisjoint(rest_factors):
                        candidates.append(x)
                
                # Solve subproblem on candidates
                # Note: 'a' is in candidates.
                # We want to replace 'a' with a subset T of candidates.
                # Gain = sum(T) - a
                best_sum, best_T = solve_subproblem(candidates)
                
                gain = best_sum - a
                if gain > 0:
                    # Apply change
                    S_set = rest.union(best_T)
                    # Update S_factors for next iteration?
                    # The while loop restarts, so we recalc next time.
                    improved = True
                    break # Restart loop with new set
        
        return S_set

    best_S = set()
    best_sum = 0
    
    # 1. Greedy Descending
    S1 = greedy_initialization(descending=True)
    S1_set = set(S1)
    sum1 = sum(S1_set)
    if sum1 > best_sum:
        best_S = S1_set
        best_sum = sum1
        
    # 2. Greedy Ascending
    S2 = greedy_initialization(descending=False)
    S2_set = set(S2)
    sum2 = sum(S2_set)
    if sum2 > best_sum:
        best_S = S2_set
        best_sum = sum2

    # 3. Randomized Greedy
    random.seed(42)
    for i in range(10):
        if time.time() > deadline - 5:
            break
        Sr = greedy_initialization(descending=True, randomize=True)
        Sr_set = set(Sr)
        sumr = sum(Sr_set)
        if sumr > best_sum:
            best_S = Sr_set
            best_sum = sumr

    # 4. Local Search on Best
    if time.time() < deadline - 5:
        best_S = local_search(best_S)
        best_sum = sum(best_S)

    # 5. If time remains, try local search on other initializations
    if time.time() < deadline - 5:
        # Re-evaluate S1 and S2 with local search if they were close
        # Or just run local search on a few random ones
        for i in range(5):
            if time.time() > deadline - 2:
                break
            Sr = greedy_initialization(descending=True, randomize=True)
            Sr_set = local_search(set(Sr))
            sumr = sum(Sr_set)
            if sumr > best_sum:
                best_S = Sr_set
                best_sum = sumr

    # Ensure return type is list of ints
    return list(best_S)
