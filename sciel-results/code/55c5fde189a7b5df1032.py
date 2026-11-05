import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    TARGET = 200
    UNIVERSE = list(range(1, 101))
    
    def can_add(reach, x):
        """Check if adding x to a set with reachable sums 'reach' creates sum 200."""
        if x > TARGET:
            return False
        if (TARGET - x) in reach:
            return False
        return True
    
    def update_reach(reach, x):
        """Update reachable sums after adding x."""
        if x > TARGET:
            return reach
        new_reach = set(reach)
        for s in reach:
            ns = s + x
            if ns <= TARGET:
                new_reach.add(ns)
        return new_reach
    
    def recompute_reach(S):
        reach = {0}
        for x in sorted(S):
            reach = update_reach(reach, x)
        return reach
    
    def greedy_add(S, reach, candidates):
        """Add as many candidates as possible to S."""
        for x in candidates:
            if x in S:
                continue
            if can_add(reach, x):
                S.add(x)
                reach = update_reach(reach, x)
        return S, reach
    
    def compute_conflict_density(S, reach):
        """
        Compute conflict density for each candidate not in S.
        A candidate x has conflict density = number of elements y in S such that
        there exists a subset T of S with sum(T) + x + y = TARGET, i.e.,
        (TARGET - x - y) is reachable by a subset of S\{y}.
        We approximate: count how many y in S have (TARGET - x - y) in reach.
        This is a proxy for how many hyperedges x participates in.
        """
        densities = {}
        for x in UNIVERSE:
            if x in S:
                continue
            if x > TARGET:
                densities[x] = 0
                continue
            count = 0
            for y in S:
                if y == x:
                    continue
                if y >= TARGET:
                    continue
                rem = TARGET - x - y
                if rem >= 0 and rem in reach:
                    count += 1
            densities[x] = count
        return densities
    
    def greedy_conflict_density(S, reach, candidates):
        """
        Greedy heuristic using conflict density:
        Iteratively select the candidate with the lowest conflict density
        that can still be added without creating sum 200.
        """
        remaining = [x for x in candidates if x not in S]
        if not remaining:
            return S, reach
        
        while remaining:
            if time.time() > deadline - 5:
                break
            
            # Compute conflict densities for remaining candidates
            densities = compute_conflict_density(S, reach)
            
            # Filter to only addable candidates
            addable = [x for x in remaining if can_add(reach, x)]
            if not addable:
                break
            
            # Select candidate with lowest conflict density
            addable.sort(key=lambda x: (densities.get(x, 0), x))
            best_candidate = addable[0]
            
            S.add(best_candidate)
            reach = update_reach(reach, best_candidate)
            remaining.remove(best_candidate)
        
        return S, reach
    
    def hypergraph_local_search(S, reach):
        """
        Local search using hypergraph conflict density:
        Try removing one element and re-adding using conflict density greedy.
        """
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            for y in S_list:
                if time.time() > deadline - 5:
                    break
                S_temp = set(S)
                S_temp.discard(y)
                reach_temp = recompute_reach(S_temp)
                candidates = [x for x in UNIVERSE if x not in S_temp]
                S_new, reach_new = greedy_conflict_density(S_temp, reach_temp, candidates)
                if len(S_new) > len(S):
                    S = S_new
                    reach = reach_new
                    improved = True
                    break
        
        return S, reach
    
    best_S = set()
    best_size = 0
    
    # Strategy 1: Start with {67,...,100} which is guaranteed valid
    S = set(range(67, 101))
    reach = recompute_reach(S)
    
    # Try to add smaller numbers using conflict density greedy
    small_numbers = list(range(1, 67))
    S, reach = greedy_conflict_density(S, reach, small_numbers)
    
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    # Local search: try removing one element and re-adding using conflict density
    S, reach = hypergraph_local_search(S, reach)
    
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    # Try random initializations with conflict density greedy
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        S_rand = set()
        reach_rand = {0}
        candidates = UNIVERSE[:]
        random.shuffle(candidates)
        S_rand, reach_rand = greedy_conflict_density(S_rand, reach_rand, candidates)
        if len(S_rand) > best_size:
            best_S = set(S_rand)
            best_size = len(S_rand)
    
    # Final pass: try to add any remaining valid elements using conflict density
    if best_S:
        best_reach = recompute_reach(best_S)
        remaining = [x for x in UNIVERSE if x not in best_S]
        best_S, best_reach = greedy_conflict_density(best_S, best_reach, remaining)
    
    return list(best_S) if best_S else [1]
