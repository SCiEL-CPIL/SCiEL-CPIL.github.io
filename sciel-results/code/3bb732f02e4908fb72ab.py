import time
from math import gcd
from itertools import combinations
import random

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    Approach: Optimized greedy deletion with incremental triple counting,
    followed by aggressive local search with efficient conflict checks.
    """
    DEADLINE = time.time() + 85.0  # safety margin before 90s kill
    
    N = 128
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # Precompute conflict pairs: for each element x, the set of (u,v) pairs in S
    # such that gcd(x,u) == gcd(x,v) == gcd(u,v)
    # This is expensive to precompute for all S, so we compute on-the-fly efficiently.
    
    # Precompute for each pair (u,v), the gcd value
    # Already in gcd_mat
    
    def count_bad_triples_fast(S_set):
        """
        For each element in S, count how many bad triples it participates in.
        Uses O(n^2) per element but with early termination.
        Returns dict: element -> count
        """
        S_list = sorted(S_set)
        n = len(S_list)
        count = {}
        
        for idx_a in range(n):
            a = S_list[idx_a]
            cnt = 0
            # For each pair (b, c) with b, c > a in S_list,
            # check if gcd(a,b) == gcd(a,c) == gcd(b,c)
            # Group by gcd(a, *)
            gcd_groups = {}
            for idx_b in range(idx_a + 1, n):
                b = S_list[idx_b]
                g_ab = gcd_mat[a][b]
                if g_ab not in gcd_groups:
                    gcd_groups[g_ab] = []
                gcd_groups[g_ab].append(b)
            
            # For each group of elements with same gcd to a,
            # count pairs (b,c) in group where gcd(b,c) == g_ab
            for g_val, group in gcd_groups.items():
                glen = len(group)
                if glen >= 2:
                    for ii in range(glen):
                        for jj in range(ii + 1, glen):
                            if gcd_mat[group[ii]][group[jj]] == g_val:
                                cnt += 1
            
            if cnt > 0:
                count[a] = cnt
        
        return count
    
    def can_add_to_set(x, S_set):
        """Check if x can be added to S_set without creating a bad triple."""
        S_list = sorted(S_set)
        n = len(S_list)
        
        # Group elements in S by gcd with x
        gcd_groups = {}
        for u in S_list:
            g = gcd_mat[x][u]
            if g not in gcd_groups:
                gcd_groups[g] = []
            gcd_groups[g].append(u)
        
        # For each group, check if any pair (u,v) has gcd(u,v) == g
        for g_val, group in gcd_groups.items():
            glen = len(group)
            if glen >= 2:
                for ii in range(glen):
                    for jj in range(ii + 1, glen):
                        if gcd_mat[group[ii]][group[jj]] == g_val:
                            return False
        
        return True
    
    def greedy_delete_fast(randomize=False, seed=None):
        """
        Optimized greedy: maintain triple counts incrementally.
        """
        if randomize and seed is not None:
            rng = random.Random(seed)
        elif randomize:
            rng = random.Random()
        else:
            rng = None
        
        S = set(range(1, N + 1))
        removed = []
        
        # Initial count
        counts = count_bad_triples_fast(S)
        
        while counts:
            if time.time() > DEADLINE:
                break
            
            # Find element with max count
            max_count = max(counts.values())
            candidates = [x for x, c in counts.items() if c == max_count]
            
            if rng:
                to_remove = rng.choice(candidates)
            else:
                # Tie-break by global constraint
                to_remove = max(candidates, key=lambda x: sum(1 for y in S if y != x and gcd_mat[x][y] > 1))
            
            S.discard(to_remove)
            removed.append(to_remove)
            
            # Update counts: recompute only for affected elements
            # An element's count changes only if it was in a triple with to_remove
            # Recompute counts for elements that had to_remove in their triples
            # Simpler: recompute all (still O(n^2) per step, but fewer steps)
            counts = count_bad_triples_fast(S)
        
        return S, removed
    
    def local_search_aggressive(S, removed, max_rounds=200):
        """
        Aggressive local search: try additions and swaps with efficient checks.
        """
        S = set(S)
        removed = list(removed)
        best_size = len(S)
        
        for round_num in range(max_rounds):
            if time.time() > DEADLINE:
                break
            improved = False
            
            # Phase A: Try adding each removed element
            for x in removed[:]:
                if time.time() > DEADLINE:
                    break
                if x in S:
                    continue
                if can_add_to_set(x, S):
                    S.add(x)
                    removed.remove(x)
                    improved = True
                    best_size = len(S)
            
            # Phase B: Try 1-for-1 swaps
            if not improved:
                found_swap = False
                for x in sorted(S, key=lambda v: -sum(1 for y in S if y != v and gcd_mat[x][y] > 1)):
                    if time.time() > DEADLINE:
                        break
                    if found_swap:
                        break
                    temp_S = S - {x}
                    for y in removed:
                        if time.time() > DEADLINE:
                            break
                        if can_add_to_set(y, temp_S):
                            S = temp_S
                            S.add(y)
                            removed.remove(y)
                            removed.append(x)
                            improved = True
                            found_swap = True
                            best_size = len(S)
                            break
            
            # Phase C: Try 2-for-1 swaps (remove 2 from S, add 1 from removed)
            if not improved and len(S) > 10:
                found_2swap = False
                S_list = sorted(S)
                n = len(S_list)
                # Limit to a subset of pairs for speed
                step = max(1, n // 40)
                for i in range(0, n - 1, step):
                    if time.time() > DEADLINE or found_2swap:
                        break
                    x1 = S_list[i]
                    for j in range(i + 1, min(i + step + 1, n)):
                        if time.time() > DEADLINE or found_2swap:
                            break
                        x2 = S_list[j]
                        temp_S = S - {x1, x2}
                        for y in removed:
                            if can_add_to_set(y, temp_S):
                                S = temp_S
                                S.add(y)
                                removed.remove(y)
                                removed.extend([x1, x2])
                                improved = True
                                found_2swap = True
                                best_size = len(S)
                                break
            
            if not improved:
                break
        
        return S, removed, best_size
    
    # Phase 1: Deterministic greedy deletion
    S, removed = greedy_delete_fast(randomize=False)
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local search on greedy result
    S, removed, _ = local_search_aggressive(S, removed, max_rounds=150)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Phase 3: Multiple randomized greedy starts
    for trial in range(30):
        if time.time() > DEADLINE:
            break
        
        S, removed = greedy_delete_fast(randomize=True, seed=42 + trial * 7)
        S, removed, _ = local_search_aggressive(S, removed, max_rounds=80)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Phase 4: Final aggressive local search on best
    final_removed = [x for x in range(1, N + 1) if x not in best_S]
    S, removed, _ = local_search_aggressive(best_S, final_removed, max_rounds=300)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Phase 5: One more round of randomized starts with more local search
    for trial in range(15):
        if time.time() > DEADLINE:
            break
        S, removed = greedy_delete_fast(randomize=True, seed=1000 + trial * 13)
        S, removed, _ = local_search_aggressive(S, removed, max_rounds=120)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    if not best_S:
        best_S = {1}
    
    return sorted(best_S)
