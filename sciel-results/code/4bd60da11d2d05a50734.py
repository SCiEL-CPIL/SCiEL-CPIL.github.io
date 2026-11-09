import time
import random
import numpy as np

def construct():
    deadline = time.time() + 58.0
    n_max = 512
    target_size = 48

    def build_conflict_graph():
        """
        Build a conflict graph where vertices are integers 1..n_max.
        Two vertices a, b are connected if one is the arithmetic mean of a subset
        of size >= 2 containing the other (plus possibly other elements).
        
        For the 3-term AP case: a, b, c with b = (a+c)/2 means b is the mean of {a,c}.
        So b conflicts with the pair {a,c}. But in a pairwise conflict graph for
        independent set, we need: if we select b, we cannot select both a and c.
        
        Actually, the problem is more complex than just 3-AP. An element x conflicts
        with any subset of size >= 2 whose mean is x. This is a hypergraph, not a graph.
        
        However, the directive suggests using a conflict graph for 3-AP. Let's start
        with the 3-AP constraint as a relaxation (necessary but not sufficient),
        then verify the full condition.
        
        For 3-AP: x is the mean of {a, b} iff a + b = 2x iff x = (a+b)/2.
        So x, a, b form a 3-term AP with x as the middle term.
        In a conflict graph for independent set: if we pick x, we cannot pick BOTH a and b.
        This is a 3-uniform hyperedge, not a pairwise edge.
        
        To convert to a graph: we can say x conflicts with a AND x conflicts with b,
        but that's too restrictive. Instead, we can use a different approach.
        
        Let's use the direct approach: greedily build a set, checking the FULL condition
        (not just 3-AP) using efficient bitset DP.
        """
        pass

    def check_full_validity(S, x):
        """
        Check if adding x to set S (where S is already valid) keeps the set valid.
        Returns True if S U {x} is valid.
        """
        if x in S:
            return False
        
        S_list = list(S)
        n = len(S_list)
        
        # Check 1: x is not the mean of any subset of S of size >= 2
        # sum(T) = k * x for some T subset of S, |T| = k >= 2
        if n >= 2:
            max_sum = sum(S_list)
            max_k = min(n, max_sum // x)
            if max_k >= 2:
                dp = [0] * (n + 1)
                dp[0] = 1
                for val in S_list:
                    for k in range(n - 1, -1, -1):
                        if dp[k]:
                            dp[k+1] |= (dp[k] << val)
                for k in range(2, max_k + 1):
                    target = k * x
                    if (dp[k] >> target) & 1:
                        return False
        
        # Check 2: No element a in S is the mean of any subset of (S \ {a}) U {x} of size >= 2
        for a in S_list:
            others = [s for s in S_list if s != a]
            m = len(others)
            if m < 1:
                continue
            
            # Case A: subset doesn't contain x. Already valid since S is valid.
            # Case B: subset contains x. M = {x} U T, T subset of others, |T| >= 1.
            # sum(M) = x + sum(T) = (|T| + 1) * a
            # sum(T) = (j + 1) * a - x where j = |T|, j >= 1
            
            if m >= 1:
                dp = [0] * (m + 1)
                dp[0] = 1
                for val in others:
                    for k in range(m - 1, -1, -1):
                        if dp[k]:
                            dp[k+1] |= (dp[k] << val)
                
                max_sum_o = sum(others)
                for j in range(1, m + 1):
                    target = (j + 1) * a - x
                    if target < 0:
                        continue
                    if target > max_sum_o:
                        break
                    if (dp[j] >> target) & 1:
                        return False
        
        return True

    def greedy_conflict_aware(order, S_set):
        """
        Greedy independent set with conflict-aware ordering.
        Prioritize elements that have fewer conflicts with already-selected elements.
        """
        S = []
        S_set = set(S_set) if S_set else set()
        
        for x in order:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if check_full_validity(S_set, x):
                S.append(x)
                S_set.add(x)
        
        return S, S_set

    # Build a conflict-aware ordering: elements with fewer "potential conflicts" first
    # A potential conflict for x is any pair (a, b) in [1, n_max] such that x = (a+b)/2
    # i.e., a + b = 2x. The number of such pairs is roughly x - 1 (for a from 1 to x-1, b = 2x-a).
    # Actually, count = number of pairs (a, b) with a < b, a + b = 2x, 1 <= a < b <= n_max.
    # a ranges from max(1, 2x - n_max) to x-1, and b = 2x - a.
    # So count = (x - 1) - max(1, 2x - n_max) + 1 if x - 1 >= max(1, 2x - n_max).
    
    def conflict_degree(x):
        """Count the number of pairs (a, b) with a < b, a + b = 2x, 1 <= a, b <= n_max."""
        a_min = max(1, 2 * x - n_max)
        a_max = x - 1
        if a_min > a_max:
            return 0
        return a_max - a_min + 1

    # Strategy: Greedy with conflict-aware ordering (lowest degree first)
    best_set = []
    best_size = 0
    
    # Order by conflict degree (ascending) - fewer conflicts first
    ordered_by_degree = sorted(range(1, n_max + 1), key=lambda x: conflict_degree(x))
    S, S_set = greedy_conflict_aware(ordered_by_degree, set())
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Also try descending degree
    ordered_by_degree_desc = sorted(range(1, n_max + 1), key=lambda x: -conflict_degree(x))
    S, S_set = greedy_conflict_aware(ordered_by_degree_desc, set())
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Random orders with conflict-aware refinement
    rng = random.Random(42)
    for i in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        S, S_set = greedy_conflict_aware(order, set())
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search: try to add or swap elements
    if best_size < target_size and time.time() < deadline:
        S = list(best_set)
        S_set = set(S)
        available = [x for x in range(1, n_max + 1) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try to add any available element
            for x in available:
                if len(S) >= target_size:
                    break
                if check_full_validity(S_set, x):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            
            if len(S) >= target_size:
                break
            
            # If no add, try swaps
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    old = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp_set = set(S_temp)
                    
                    for x in available:
                        if x in S_temp_set:
                            continue
                        if check_full_validity(S_temp_set, x):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(old)
                            improved = True
                            break
                    if improved:
                        break
    
    return best_set
