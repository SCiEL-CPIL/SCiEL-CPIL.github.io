"""Optimized construction for large pairwise noncoprime sets containing 255."""
import math
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Find largest subset of [1,255] containing 255 with gcd(a,b)>1 for all distinct pairs.
    
    # Generate all numbers in [1,255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency using bitsets
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def time_check():
        return time.time() < deadline
    
    def is_clique(clique_set):
        for i in clique_set:
            for j in clique_set:
                if i == j:
                    continue
                if not (adj[i] >> j) & 1:
                    return False
        return True
    
    def try_add_vertex(clique_set, v):
        """Try to add vertex v to clique. Return True if successful."""
        for u in clique_set:
            if not (adj[u] >> v) & 1:
                return False
        return True
    
    def greedy_expand(clique_set):
        """Greedily add vertices that are compatible with all current clique members."""
        improved = True
        while improved and time_check():
            improved = False
            for v in range(n):
                if v in clique_set:
                    continue
                if try_add_vertex(clique_set, v):
                    clique_set.add(v)
                    improved = True
        return clique_set
    
    def swap_local_search(clique_set):
        """Try swap moves: remove one vertex, add one or two others."""
        improved = True
        while improved and time_check():
            improved = False
            cur_size = len(clique_set)
            # Try single swaps: remove u, add v
            for u in list(clique_set):
                if u == idx_255:
                    continue  # Must keep 255
                for v in range(n):
                    if v in clique_set or v == u:
                        continue
                    # Check if v is compatible with all clique members except u
                    ok = True
                    for w in clique_set:
                        if w == u:
                            continue
                        if not (adj[w] >> v) & 1:
                            ok = False
                            break
                    if ok:
                        clique_set.remove(u)
                        clique_set.add(v)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try double swaps: remove u, add v and w
                for u in list(clique_set):
                    if u == idx_255:
                        continue
                    for v in range(n):
                        if v in clique_set or v == u:
                            continue
                        # Check v compatible with all except u
                        ok_v = True
                        for w in clique_set:
                            if w == u:
                                continue
                            if not (adj[w] >> v) & 1:
                                ok_v = False
                                break
                        if not ok_v:
                            continue
                        for w2 in range(n):
                            if w2 in clique_set or w2 == u or w2 == v:
                                continue
                            # Check w2 compatible with all except u, and with v
                            ok_w = True
                            if not (adj[v] >> w2) & 1:
                                ok_w = False
                            else:
                                for w in clique_set:
                                    if w == u:
                                        continue
                                    if not (adj[w] >> w2) & 1:
                                        ok_w = False
                                        break
                            if ok_w:
                                clique_set.remove(u)
                                clique_set.add(v)
                                clique_set.add(w2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            # After any swap, try to greedily expand again
            if improved:
                clique_set = greedy_expand(clique_set)
        return clique_set
    
    # Multiple initialization strategies
    init_strategies = []
    
    # Strategy 1: all multiples of 3
    s1 = set(i for i in range(n) if candidates[i] % 3 == 0)
    init_strategies.append(s1)
    
    # Strategy 2: all multiples of 5
    s2 = set(i for i in range(n) if candidates[i] % 5 == 0)
    init_strategies.append(s2)
    
    # Strategy 3: all multiples of 17
    s3 = set(i for i in range(n) if candidates[i] % 17 == 0)
    init_strategies.append(s3)
    
    # Strategy 4: multiples of 3 that are also divisible by 5 or 17 (intersection)
    s4 = set(i for i in range(n) if candidates[i] % 3 == 0 and (candidates[i] % 5 == 0 or candidates[i] % 17 == 0))
    if len(s4) > 1:
        init_strategies.append(s4)
    
    # Strategy 5: all numbers divisible by 3 or 5 (larger base)
    s5 = set(i for i in range(n) if candidates[i] % 3 == 0 or candidates[i] % 5 == 0)
    init_strategies.append(s5)
    
    # Strategy 6: all numbers divisible by 3 or 17
    s6 = set(i for i in range(n) if candidates[i] % 3 == 0 or candidates[i] % 17 == 0)
    init_strategies.append(s6)
    
    # Strategy 7: all numbers divisible by 5 or 17
    s7 = set(i for i in range(n) if candidates[i] % 5 == 0 or candidates[i] % 17 == 0)
    init_strategies.append(s7)
    
    best_clique = set()
    best_size = 0
    
    for init in init_strategies:
        if not time_check():
            break
        # Ensure 255 is in the clique
        if idx_255 not in init:
            init = init | {idx_255}
        
        # Verify it's actually a clique before proceeding
        if not is_clique(init):
            continue
        
        # Greedy expand
        cur = greedy_expand(init.copy())
        
        # Swap local search
        cur = swap_local_search(cur)
        
        # Final greedy expand
        cur = greedy_expand(cur)
        
        if len(cur) > best_size:
            best_size = len(cur)
            best_clique = cur.copy()
    
    # Additional: try more targeted initializations based on factor patterns
    # Numbers with specific prime factorizations
    if time_check():
        # Try: all numbers divisible by 3, plus some that bridge to 5/17
        # Start with multiples of 3, then try to add numbers that share factor 5 or 17
        # with many existing members
        base = set(i for i in range(n) if candidates[i] % 3 == 0)
        if idx_255 not in base:
            base.add(idx_255)
        
        # Score each non-base vertex by how many base members it's compatible with
        scores = []
        for v in range(n):
            if v in base:
                continue
            count = 0
            for u in base:
                if (adj[u] >> v) & 1:
                    count += 1
            if count > 0:
                scores.append((count, v))
        scores.sort(reverse=True)
        
        # Try adding top-scoring vertices one by one
        for count, v in scores:
            if not time_check():
                break
            if try_add_vertex(base, v):
                base.add(v)
        
        base = greedy_expand(base)
        base = swap_local_search(base)
        base = greedy_expand(base)
        
        if len(base) > best_size:
            best_size = len(base)
            best_clique = base.copy()
    
    # Convert to result
    result = [candidates[i] for i in best_clique]
    result.sort()
    
    # Verify
    assert 255 in result
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert math.gcd(result[i], result[j]) > 1
    
    return result

# EVOLVE-BLOCK-END
