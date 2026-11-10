"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    deadline = time.time() + 85  # 5s safety margin
    N = 128
    
    # Precompute bad_with[u][v] = list of w such that (u, v, w) is a bad triple
    # u, v, w are 1-based indices.
    # bad_with is a 129x129 list of lists (index 0 unused)
    bad_with = [[[] for _ in range(N+1)] for _ in range(N+1)]
    
    for u in range(1, N+1):
        for v in range(u+1, N+1):
            g = gcd(u, v)
            u_prime = u // g
            v_prime = v // g
            # w = k * g
            # 1 <= k*g <= 128
            max_k = N // g
            for k in range(1, max_k + 1):
                w = k * g
                if w == u or w == v:
                    continue
                # Check gcd(k, u_prime) == 1 and gcd(k, v_prime) == 1
                if gcd(k, u_prime) == 1 and gcd(k, v_prime) == 1:
                    bad_with[u][v].append(w)
                    bad_with[v][u].append(w)
                    bad_with[w][u].append(v)
                    bad_with[w][v].append(u)
                    bad_with[u][w].append(v)
                    bad_with[v][w].append(u)
                    # Wait, bad_with[x][y] should contain z such that (x,y,z) is bad.
                    # So for triple (u,v,w):
                    # bad_with[u][v] gets w
                    # bad_with[v][u] gets w
                    # bad_with[u][w] gets v
                    # bad_with[w][u] gets v
                    # bad_with[v][w] gets u
                    # bad_with[w][v] gets u
                    
    # Correcting the precomputation logic above:
    # The loop above adds to bad_with[u][v] etc correctly if we just do:
    # bad_with[u][v].append(w)
    # bad_with[v][u].append(w)
    # bad_with[u][w].append(v)
    # bad_with[w][u].append(v)
    # bad_with[v][w].append(u)
    # bad_with[w][v].append(u)
    # This is symmetric.
    
    # Greedy construction
    def greedy_init():
        S = set()
        # Sort by degree? Or random?
        # Let's try random shuffle with a fixed seed for reproducibility in trials, 
        # but multiple trials with different seeds.
        # For init, let's use a heuristic: prefer numbers with fewer conflicts?
        # Degree of v = sum(len(bad_with[v][u]) for u in 1..128 if u!=v) / 2? 
        # Actually, just count how many pairs (u,w) are bad with v.
        # This is sum(len(bad_with[v][u]) for u in 1..128) // 2? No.
        # Each bad triple (v,u,w) contributes 1 to bad_with[v][u] (w) and 1 to bad_with[v][w] (u).
        # So sum(len(bad_with[v][u])) counts each bad triple containing v twice?
        # No, bad_with[v][u] contains w. So for a fixed v, the number of bad triples containing v is 
        # sum_{u != v} len(bad_with[v][u]) / 2? No.
        # If (v,u,w) is bad, w is in bad_with[v][u]. u is in bad_with[v][w].
        # So for fixed v, each bad triple {v,u,w} appears in bad_with[v][u] and bad_with[v][w].
        # So total count is 2 * (# bad triples containing v).
        # So degree = sum(len(bad_with[v][u]) for u in range(1, N+1) if u != v) // 2
        
        degrees = [0] * (N+1)
        for v in range(1, N+1):
            cnt = 0
            for u in range(1, N+1):
                if u != v:
                    cnt += len(bad_with[v][u])
            degrees[v] = cnt // 2
            
        # Sort by degree ascending (least constrained first)
        order = sorted(range(1, N+1), key=lambda x: degrees[x])
        
        # We need to check if adding v is valid.
        # v is valid if no pair (u,w) in S is bad with v.
        # i.e., for all u in S, w is not in bad_with[v][u] intersect S?
        # Actually, if (u,w,v) is bad, then w is in bad_with[u][v] and u is in bad_with[w][v].
        # So we need to check if there exists u in S such that bad_with[u][v] intersects S.
        # Or simpler: maintain conflict_count.
        
        S_set = set()
        conflict_count = [0] * (N+1) # conflict_count[v] = # pairs (u,w) in S such that (u,w,v) is bad
        
        for v in order:
            if conflict_count[v] == 0:
                S_set.add(v)
                # Update conflict_count for all z not in S_set
                for u in S_set:
                    if u == v: continue
                    for w in bad_with[v][u]:
                        if w not in S_set:
                            conflict_count[w] += 1
        return S_set

    # Local search with swaps
    def local_search(S_set, rng):
        # Compute initial conflict_count
        conflict_count = [0] * (N+1)
        for v in S_set:
            for u in S_set:
                if u == v: continue
                for w in bad_with[v][u]:
                    if w not in S_set:
                        conflict_count[w] += 1
        
        best_size = len(S_set)
        best_set = set(S_set)
        
        # Add phase: greedily add any valid vertex
        improved = True
        while improved:
            improved = False
            for v in range(1, N+1):
                if v not in S_set and conflict_count[v] == 0:
                    S_set.add(v)
                    improved = True
                    for u in S_set:
                        if u == v: continue
                        for w in bad_with[v][u]:
                            if w not in S_set:
                                conflict_count[w] += 1
                    if len(S_set) > best_size:
                        best_size = len(S_set)
                        best_set = set(S_set)
            
        # Swap phase
        # We iterate until no improvement or time limit
        while time.time() < deadline:
            # Shuffle S to try different removals
            elements = list(S_set)
            rng.shuffle(elements)
            
            found_improvement = False
            for rem in elements:
                if time.time() > deadline:
                    break
                
                # Simulate removal of rem
                S_sim = set(S_set)
                S_sim.remove(rem)
                
                # Recompute conflict_count for S_sim
                # This is O(|S|^2 * K) which might be slow if done from scratch.
                # But |S| ~ 60, N=128.
                # Let's try to update incrementally?
                # Given the complexity of reverting, and the small N, 
                # recomputing from scratch for the simulation might be acceptable 
                # if we limit the number of swap attempts.
                # However, the previous code was slow. 
                # Let's use the copy-conflict-count approach.
                
                cc_sim = conflict_count[:] # Copy is O(N)
                
                # Remove rem from cc_sim
                # For each u in S_sim (which is S_set - {rem}):
                #   For each w in bad_with[rem][u]:
                #      if w not in S_sim: cc_sim[w] -= 1
                for u in S_sim:
                    for w in bad_with[rem][u]:
                        if w not in S_sim:
                            cc_sim[w] -= 1
                
                # Try to add vertices to S_sim
                sim_improved = True
                while sim_improved:
                    sim_improved = False
                    for v in range(1, N+1):
                        if v not in S_sim and cc_sim[v] == 0:
                            S_sim.add(v)
                            sim_improved = True
                            for u in S_sim:
                                if u == v: continue
                                for w in bad_with[v][u]:
                                    if w not in S_sim:
                                        cc_sim[w] += 1
                
                if len(S_sim) > len(S_set):
                    # Accept
                    S_set = S_sim
                    conflict_count = cc_sim
                    found_improvement = True
                    if len(S_set) > best_size:
                        best_size = len(S_set)
                        best_set = set(S_set)
                    break # Restart swap loop with new set
                
            if not found_improvement:
                # No improvement in this round of swaps
                # Try a random perturbation?
                # Or break if we've done many rounds?
                # Let's try one random perturbation: remove 2 random elements, re-add
                # This helps escape local minima.
                if len(S_set) > 2:
                    elems = list(S_set)
                    rng.shuffle(elems)
                    rem1, rem2 = elems[0], elems[1]
                    S_sim = set(S_set)
                    S_sim.remove(rem1)
                    S_sim.remove(rem2)
                    
                    cc_sim = conflict_count[:]
                    
                    # Remove rem1
                    for u in S_sim:
                        for w in bad_with[rem1][u]:
                            if w not in S_sim:
                                cc_sim[w] -= 1
                    # Remove rem2
                    for u in S_sim:
                        for w in bad_with[rem2][u]:
                            if w not in S_sim:
                                cc_sim[w] -= 1
                    
                    # Add back greedily
                    sim_improved = True
                    while sim_improved:
                        sim_improved = False
                        for v in range(1, N+1):
                            if v not in S_sim and cc_sim[v] == 0:
                                S_sim.add(v)
                                sim_improved = True
                                for u in S_sim:
                                    if u == v: continue
                                    for w in bad_with[v][u]:
                                        if w not in S_sim:
                                            cc_sim[w] += 1
                                        
                    if len(S_sim) > len(S_set):
                        S_set = S_sim
                        conflict_count = cc_sim
                        if len(S_set) > best_size:
                            best_size = len(S_set)
                            best_set = set(S_set)
                    else:
                        # Perturbation failed, try next trial or break?
                        # We can break the while loop if we want to try a new random init
                        break
                else:
                    break

        return best_set

    rng = random.Random(42)
    best_global = set()
    
    # Multiple restarts
    for i in range(20):
        if time.time() > deadline:
            break
        
        # Different init strategies
        if i % 3 == 0:
            S_init = greedy_init()
        elif i % 3 == 1:
            # Random init
            S_init = set(random.sample(range(1, N+1), N//2))
            # Clean up invalid triples? 
            # Greedy add will fix it, but starting with invalid set is bad.
            # Better: start with empty and greedy add with random order
            order = list(range(1, N+1))
            rng.shuffle(order)
            S_init = set()
            conflict_count = [0] * (N+1)
            for v in order:
                if conflict_count[v] == 0:
                    S_init.add(v)
                    for u in S_init:
                        if u == v: continue
                        for w in bad_with[v][u]:
                            if w not in S_init:
                                conflict_count[w] += 1
        else:
            # Degree based but shuffled
            degrees = [0] * (N+1)
            for v in range(1, N+1):
                cnt = 0
                for u in range(1, N+1):
                    if u != v:
                        cnt += len(bad_with[v][u])
                degrees[v] = cnt // 2
            order = sorted(range(1, N+1), key=lambda x: degrees[x])
            rng.shuffle(order)
            S_init = set()
            conflict_count = [0] * (N+1)
            for v in order:
                if conflict_count[v] == 0:
                    S_init.add(v)
                    for u in S_init:
                        if u == v: continue
                        for w in bad_with[v][u]:
                            if w not in S_init:
                                conflict_count[w] += 1

        S_best = local_search(S_init, rng)
        if len(S_best) > len(best_global):
            best_global = S_best
            
    return sorted(list(best_global))

# EVOLVE-BLOCK-END
