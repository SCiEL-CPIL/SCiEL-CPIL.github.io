import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute GCDs and Conflicts
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # conflicts[v] is a set of frozensets {u, w} such that {v, u, w} is a bad triple
    # A triple {a,b,c} is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    conflicts = [set() for _ in range(N + 1)]
    
    # Also precompute the set of all bad triples for potential use
    bad_triples = []
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))
            bad_triples.append((a, b, c))
    
    # Precompute degrees (number of pairs that conflict with v)
    degrees = [len(conflicts[v]) for v in range(N + 1)]
    
    best_size = 0
    best_set = set()
    
    # Helper function to check if a set is valid
    def is_valid(S):
        S_list = list(S)
        for a, b, c in combinations(S_list, 3):
            if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                return False
        return True

    # Improved Greedy + Local Search
    # We will use multiple restarts with different random seeds/orders
    
    num_restarts = 50
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Strategy: Start with a random subset of "easy" numbers (low degree)
        # Or start with a greedy construction.
        
        # Let's try a greedy construction first, then local search.
        # Order: Randomize, but bias towards low degree?
        # Or just random shuffle?
        
        candidates = numbers[:]
        rng.shuffle(candidates)
        
        # Greedy Add
        S = set()
        active_pairs = set()
        # pair_count[p] = number of x in S such that p is in conflicts[x]
        # Actually, we just need to know if a pair is "active" (both in S)
        # And for a new v, it conflicts if any pair {u,w} in conflicts[v] is active.
        
        # To check if v can be added:
        # For each pair {u,w} in conflicts[v], if u in S and w in S, then v cannot be added.
        
        def can_add(v, S, active_pairs):
            for p in conflicts[v]:
                if p in active_pairs:
                    return False
            return True

        def add(v):
            S.add(v)
            # Update active_pairs
            for u in S:
                if u != v:
                    p = frozenset((u, v))
                    active_pairs.add(p)

        def remove(v):
            # Update active_pairs
            for u in S:
                if u != v:
                    p = frozenset((u, v))
                    if p in active_pairs:
                        active_pairs.remove(p)
            S.remove(v)

        # Phase 1: Greedy Initialization
        # We can try different orders.
        # Let's try sorting by degree ascending, then shuffling ties?
        # Or just random.
        
        # Let's use a "weighted" random order based on degree?
        # Lower degree is better.
        # Let's just shuffle for now, maybe try sorted by degree in some restarts.
        
        if restart % 2 == 0:
            candidates = sorted(numbers, key=lambda x: (degrees[x], rng.random()))
        else:
            rng.shuffle(candidates)

        for v in candidates:
            if can_add(v, S, active_pairs):
                add(v)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = set(S)
        
        # Phase 2: Local Search
        # 1. Add any valid element
        # 2. Swap 1 for 1 (if it allows adding more later? or just size increase is 0, so we need 1-for-2 or 2-for-3)
        # Actually, a swap of 1 for 1 keeps size same. We need to find a swap that allows future additions.
        # Standard local search for independent set:
        # - Try to add a vertex.
        # - If not possible, try to remove a vertex u, then try to add a vertex v.
        #   If removing u allows adding v, and v has "better" properties (e.g., allows more additions later), do it.
        #   But simple 1-1 swap doesn't increase size.
        #   We need 1-2 swap (remove 1, add 2) to increase size by 1.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try to add
            remaining = set(numbers) - S
            for v in remaining:
                if can_add(v, S, active_pairs):
                    add(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-2 swap
            # Remove u, then try to add v1, v2.
            # This is expensive. O(|S| * |R|^2 * check)
            # Let's be smart.
            # For each u in S:
            #   Remove u
            #   Find all v in R that can be added.
            #   If there are at least 2, add them.
            #   If there is 1, maybe it's not worth it unless that v enables more?
            #   For now, just look for 2.
            
            found_swap = False
            for u in list(S):
                if time.time() > deadline - 2:
                    break
                
                remove(u)
                
                valid_v = []
                for v in remaining:
                    if can_add(v, S, active_pairs):
                        valid_v.append(v)
                
                # Check pairs in valid_v
                if len(valid_v) >= 2:
                    # Try to add two that don't conflict with each other
                    # v1, v2 are both valid wrt S\{u}.
                    # Do they conflict with each other?
                    # i.e., is there a w in S\{u} such that {v1, v2, w} is bad?
                    # This is equivalent to: is frozenset({v1, v2}) in conflicts[w] for any w in S\{u}?
                    # Or simpler: is frozenset({v1, v2}) in active_pairs? No, active_pairs contains pairs from S\{u}.
                    # We need to check if adding v1 and v2 creates a bad triple with any w in S\{u}.
                    # Bad triple {v1, v2, w} means gcd(v1,v2)==gcd(v1,w)==gcd(v2,w).
                    # This is equivalent to frozenset({v1, v2}) being in conflicts[w].
                    
                    # Let's just check all pairs in valid_v
                    added = False
                    for i in range(len(valid_v)):
                        for j in range(i + 1, len(valid_v)):
                            v1 = valid_v[i]
                            v2 = valid_v[j]
                            p12 = frozenset((v1, v2))
                            
                            # Check if p12 conflicts with any w in S
                            conflict = False
                            for w in S:
                                if p12 in conflicts[w]:
                                    conflict = True
                                    break
                            
                            if not conflict:
                                # Add v1 and v2
                                add(v1)
                                add(v2)
                                found_swap = True
                                improved = True
                                added = True
                                break
                        if added:
                            break
                    
                    if added:
                        break
                
                if not found_swap:
                    # Restore u
                    add(u)
            
            if improved:
                continue
            
            # Try 1-1 swap? (Remove u, add v, size same, but maybe better structure)
            # This is harder to justify without a heuristic.
            # Let's skip for now or implement a simple "improvement" metric.
            # For example, sum of degrees of neighbors?
            
            # Let's try a random restart of the local search state?
            # Or just break and do a new global restart.
            
            # If we are stuck, maybe do a small random perturbation?
            # Remove 2 random elements, add 2 random valid ones?
            # This is risky.
            
            # Let's just break the local search and do a new restart.
            break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    return list(best_set)
