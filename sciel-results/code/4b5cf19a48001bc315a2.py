"""
Strategy: Maximum Independent Set on a Conflict Graph.
1. Precompute a conflict graph G on vertices {1, ..., 24}.
   An edge (u, v) exists if 1/u equals the sum of reciprocals of a subset of {1,...,24} \ {u, v}.
   Note: The condition is "no reciprocal of a selected element may equal the sum of reciprocals 
   of a subset of the OTHER selected elements".
   This implies that for any pair (u, v) in the selected set S, if there exists a subset T of S \ {u} 
   such that sum_{t in T} 1/t = 1/u, then S is invalid.
   
   Crucially, the subset T can contain any number of elements from S \ {u}.
   This makes the constraint global, not just pairwise.
   However, we can simplify:
   If 1/u is a sum of reciprocals of some set T, and T is a subset of S, then S is invalid.
   This is not a simple graph property (it's hypergraph or set-based).
   
   Wait, let's re-read carefully:
   "no reciprocal of a selected element may equal the sum of reciprocals of a subset of the OTHER selected elements."
   
   This means: For all u in S, and for all subsets T of S \ {u}, sum_{t in T} (1/t) != 1/u.
   
   This is a global constraint on the set S. It is NOT reducible to a simple pairwise graph conflict 
   in the sense that if (u,v) are in S, they don't necessarily conflict, but adding w might create a conflict 
   where 1/u = 1/v + 1/w.
   
   So, we cannot just build a static conflict graph of pairs. We need to search for a set S that is "reciprocal-subsum-free".
   
   Let's re-evaluate the "Conflict Graph" idea.
   If we define a conflict between u and a set T, it's not a graph.
   However, we can use a branch-and-bound search over subsets of {1..24}.
   Since N=24 is small, we can potentially enumerate or use a smart search.
   2^24 is ~16 million. This is feasible if the check is fast.
   
   How to check if a set S is valid?
   For each u in S, we need to check if 1/u can be formed by a subset sum of reciprocals of S\{u}.
   Let R = {1/x for x in S\{u}}. We need to know if 1/u is in the subset sums of R.
   Subset sum with rational numbers.
   Since the numbers are reciprocals of integers <= 24, we can use exact arithmetic (Fraction).
   The number of elements in S is at most 24.
   Checking subset sum for each u in S takes O(2^(|S|-1)) in worst case, which is too slow if done naively for every step of a search.
   
   Alternative:
   Precompute all possible "target" sums?
   The targets are 1/u for u in 1..24.
   For a given set S, we check validity.
   
   Let's try a different angle:
   The condition is equivalent to: The set of reciprocals R_S = {1/x | x in S} is such that no element is a sum of a subset of the others.
   This is exactly the definition of a "sum-free" set in the additive group of rationals, restricted to these specific values.
   Actually, it's "subset-sum-free" for the specific targets.
   
   Let's use a Branch and Bound search to maximize |S|.
   We can order elements.
   We can prune if the current set is already invalid.
   We can also use an upper bound on the maximum possible size.
   
   Is there a structural property?
   If we pick large numbers, their reciprocals are small.
   1/1 = 1. Sum of others must be 1.
   1/2 = 0.5.
   
   Let's look at the seed solution: range(13, 25) -> {13, ..., 24}. Size 12.
   Score 1.0.
   Can we do better?
   
   Let's try to include smaller numbers.
   If we include 1, 1/1 = 1. We need a subset of others to sum to 1.
   If we include 2, 1/2 = 0.5.
   
   Let's implement a search that tries to build a set S.
   We will use a recursive backtracking algorithm.
   State: current set S, next candidate index i.
   We want to maximize len(S).
   
   Optimization:
   Check validity incrementally?
   When adding a new element x to S, we must ensure:
   1. 1/x is not a sum of a subset of S (old S).
   2. For every y in S (old S), 1/y is not a sum of a subset of S ∪ {x} \ {y}.
      This second part is tricky because the subset can now include x.
      1/y = sum(T) where T ⊆ S ∪ {x} \ {y}.
      This implies either:
      a) sum(T) = 1/y where T ⊆ S \ {y} (already checked when y was added or previously)
      b) sum(T \ {x}) + 1/x = 1/y => sum(T \ {x}) = 1/y - 1/x.
         So we need to check if (1/y - 1/x) is a subset sum of S \ {y, x}.
   
   So, to check if we can add x to S:
   For each y in S:
       target = 1/y - 1/x
       Check if target is a subset sum of {1/z for z in S if z != y and z != x}.
   Also check if 1/x is a subset sum of {1/z for z in S}.
   
   Subset sum check with Fractions.
   Since |S| <= 24, the subset sum check can be done with a set of reachable sums.
   Number of sums can be 2^k. If k=10, 1024 sums. If k=20, 1 million sums.
   This might be too slow for deep recursion.
   
   However, we can prune the search space.
   The maximum size is likely not 24.
   Let's estimate.
   If we take all odd numbers?
   1, 3, 5, ... 23.
   1/1 = 1. Others: 1/3+1/5+... < 1?
   Sum of 1/odd from 3 to 23.
   1/3+1/5+1/7+1/9+1/11+1/13+1/15+1/17+1/19+1/21+1/23 ≈ 0.33+0.2+0.14+0.11+0.09+0.07+0.06+0.05+0.05+0.04+0.04 ≈ 1.18.
   So 1/1 could potentially be formed.
   
   Let's write the code with a time limit and a robust subset sum checker using a set of Fractions, but with a cap on the number of sums to avoid explosion, or just rely on the small N.
   Actually, for N=24, the worst case subset sum set size is 2^23 which is huge.
   But we are checking specific targets.
   We can use a dictionary or set of reachable sums, but prune sums that are >= target?
   Since all terms are positive, if sum > target, we can prune.
   Also, if sum + max_possible_remaining < target, prune.
   
   Let's implement a DFS with pruning.
   
   To make it faster, we can precompute the reciprocal values.
   
   One more optimization:
   If we find a valid set of size K, we can try to extend it.
   We can use a greedy approach to find a good initial solution, then use local search or branch and bound to improve.
   
   Given the 90s limit, and N=24, a well-pruned B&B should find the optimum.
   The optimum size is likely around 10-15.
   
   Let's refine the "Can we add x to S" check.
   S is a list of selected integers.
   To check if x can be added:
   1. Check if 1/x is a subset sum of reciprocals of S.
   2. For each y in S:
      Check if (1/y - 1/x) is a subset sum of reciprocals of S \ {y, x}.
      Note: 1/y - 1/x must be positive. If 1/y <= 1/x, then 1/y - 1/x <= 0.
      Since reciprocals are positive, the sum of a non-empty subset is positive.
      The empty subset sums to 0.
      So if 1/y - 1/x == 0, i.e., x=y, impossible since distinct.
      If 1/y < 1/x, then 1/y - 1/x < 0. No subset of positive numbers sums to a negative number. So this case is automatically safe.
      So we only need to check if x < y (so 1/x > 1/y, then 1/y - 1/x < 0, safe).
      Wait, 1/y - 1/x. If x < y, then 1/x > 1/y, so 1/y - 1/x is negative. Safe.
      If x > y, then 1/x < 1/y, so 1/y - 1/x is positive. We need to check if this value is a subset sum of S \ {y, x}.
      
   So, for each y in S such that y < x (i.e., 1/y > 1/x):
       target = 1/y - 1/x
       Check if target is a subset sum of {1/z for z in S if z != y and z != x}.
       
   And check if 1/x is a subset sum of {1/z for z in S}.
   
   This reduces the number of checks.
   
   Implementation of subset sum check:
   def is_subset_sum(target, items):
       # items are Fractions
       # use DFS or DP with set
       # since items are small count, DFS with pruning is good
       # sort items descending to find large sums quickly? Or ascending?
       # Pruning: if current_sum > target, return False.
       # if current_sum + sum(remaining) < target, return False.
       
   Let's code this.
"""

import time

_START_TIME = None
_TIME_LIMIT = 80.0  # Leave 10s buffer for 90s limit
_EPS = 1e-9

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

def _subset_sum_possible(target, items):
    """
    Check if 'target' (float) can be formed by a subset sum of 'items' (list of floats).
    items are all positive.
    """
    if target < 0:
        return False
    if abs(target) < _EPS:
        return True
    if not items:
        return False
    
    # Sort items descending to find large sums quickly? 
    # Actually, for DP set, order doesn't matter for correctness, but for pruning sum < target, 
    # we just need total sum.
    total_sum = sum(items)
    if total_sum + _EPS < target:
        return False
        
    # Use a set of reachable sums
    # We can prune sums that exceed target + EPS since all items are positive
    reachable = {0.0}
    for item in items:
        new_reachable = set()
        for s in reachable:
            ns = s + item
            if ns <= target + _EPS:
                new_reachable.add(ns)
        reachable.update(new_reachable)
        
    # Check if target is in reachable
    for s in reachable:
        if abs(s - target) < _EPS:
            return True
            
    return False

def _can_add(x, S):
    """
    Check if x can be added to set S (list of ints) without violating constraints.
    """
    if x in S:
        return False
        
    # Reciprocals of S
    recips_S = [1.0 / y for y in S]
    recip_x = 1.0 / x
    
    # Check 1: 1/x is not a subset sum of recips_S
    if _subset_sum_possible(recip_x, recips_S):
        return False
        
    # Check 2: For each y in S, if 1/y - 1/x > 0, check if it is a subset sum of S \ {y, x}
    for y in S:
        if y < x:
            # 1/y > 1/x, so 1/y - 1/x > 0
            target = (1.0 / y) - recip_x
            # Items are reciprocals of S excluding y and x
            items = [1.0 / z for z in S if z != y and z != x]
            if _subset_sum_possible(target, items):
                return False
        # If y > x, 1/y < 1/x, target is negative, impossible with positive items.
        # If y == x, not possible as x not in S.
        
    return True

def _greedy_initial():
    S = []
    for x in range(1, 25):
        if _can_add(x, S):
            S.append(x)
    return S

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Get a good initial solution
    best_set = _greedy_initial()
    
    # Now do B&B to improve
    # We can restart the search with the knowledge of best_set size.
    
    def search(i, S):
        nonlocal best_set
        if _time_exceeded():
            return
            
        # Prune
        if len(S) + (24 - i) <= len(best_set):
            return
            
        if i == 24:
            if len(S) > len(best_set):
                best_set = list(S)
            return
            
        x = i + 1
        
        # Try include
        if _can_add(x, S):
            S.append(x)
            search(i + 1, S)
            S.pop()
            
        # Try exclude
        search(i + 1, S)
        
    search(0, [])
    
    return best_set
