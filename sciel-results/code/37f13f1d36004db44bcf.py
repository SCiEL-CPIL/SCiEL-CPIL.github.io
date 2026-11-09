import time
import math
from functools import lru_cache

def construct():
    N = 512
    start_time = time.time()
    TIME_LIMIT = 50.0  # Safety margin
    
    # Precompute gcd for speed? Or just use math.gcd.
    # math.gcd is C-implemented and fast.
    
    # Build adjacency list
    # adj[i] is a set of j such that lcm(i, j) <= N
    # Note: i and j are 1-indexed in problem statement, but we use 0-indexed for arrays?
    # Let's stick to 1..N for clarity in lcm calc, or 0..N-1.
    # Let's use 0..N-1 representing values 1..N.
    
    adj = [set() for _ in range(N)]
    
    for i in range(N):
        u = i + 1
        for j in range(i + 1, N):
            v = j + 1
            # Check lcm(u, v) <= N
            # lcm(u, v) = u * v // gcd(u, v)
            g = math.gcd(u, v)
            if u * v // g <= N:
                adj[i].add(j)
                adj[j].add(i)
        
        # Self-loop not needed for clique definition usually, 
        # but in BK, we consider vertices in P.
        # A clique is a set of vertices where every pair is connected.
        
    # Bron-Kerbosch with pivot
    # R: current clique
    # P: candidates that can be added to R
    # X: candidates that have already been processed (to avoid duplicates)
    
    best_clique = set()
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        # Time check
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if not P and not X:
            # Found a maximal clique
            if len(R) > len(best_clique):
                best_clique = set(R)
            return
        
        # Choose pivot u from P union X
        # Standard heuristic: pick u with max degree in P
        # If P is empty, pick from X? No, if P is empty, we are at a leaf.
        # The condition for leaf is P empty AND X empty.
        # If P is empty but X is not, we still check if it's a maximal clique?
        # Actually, standard BK:
        # if P and X are both empty, R is a maximal clique.
        # Otherwise, pick pivot u from P U X.
        # For each v in P \ N(u):
        #    BK(R U {v}, P \ N(v), X \ N(v))
        #    P = P \ {v}
        #    X = X U {v}
        
        # Optimization: If |R| + |P| <= |best_clique|, prune.
        if len(R) + len(P) <= len(best_clique):
            return
            
        # Pick pivot
        # We want to minimize the number of recursive calls.
        # u should be in P U X.
        # Let's pick u that maximizes |P \cap N(u)|.
        # This minimizes |P \ N(u)|.
        
        union_px = P | X
        if not union_px:
            return
            
        # Find u
        max_deg = -1
        u = -1
        # Iterate over union_px
        # To be fast, we can just pick the first one if N is small, but heuristic helps.
        # For N=512, iterating is fine.
        for node in union_px:
            # Count neighbors in P
            # len(P & adj[node])
            deg = len(P & adj[node])
            if deg > max_deg:
                max_deg = deg
                u = node
                if deg == len(P):
                    break # Can't do better than all of P
        
        # Iterate over P \ N(u)
        # We need to remove u's neighbors from P for the iteration set
        # But we must update P and X in the loop.
        
        # To avoid modifying P while iterating, we take a snapshot of the nodes to process
        # nodes_to_process = list(P - adj[u])
        # But note: if u is in P, adj[u] does not contain u (no self loops in our adj sets).
        # So P - adj[u] includes u if u is in P.
        
        # Efficient iteration:
        # We convert P to a list to iterate, but we must be careful with updates.
        # Standard approach:
        # For v in P \ N(u):
        #    ...
        #    P.remove(v)
        #    X.add(v)
        
        # Since sets are mutable, we can iterate over a copy of the set difference.
        # However, creating a new set/list at every node can be slow.
        # Alternative: iterate over P, and skip if v in adj[u].
        
        # Let's use a list of candidates from P that are NOT neighbors of u.
        # candidates = [v for v in P if v not in adj[u]]
        
        # Optimization: 
        # If we pick u such that it is in P, then u is not in adj[u] (assuming no self loops).
        # So u is a candidate.
        
        # Let's try to be smart.
        # We will iterate over a copy of P.
        p_list = list(P)
        
        for v in p_list:
            if v in adj[u]:
                continue
            
            # Recursive call
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            
            # Update P and X
            P.remove(v)
            X.add(v)
            
            # Pruning check again?
            # The check at the top of the function handles it for the next iteration's context?
            # No, the check is at the start. We should check here too to break early if possible?
            # Actually, the standard BK doesn't break the loop early based on size, 
            # but the recursive call will return quickly if pruned.
            # However, if len(R) + len(P) <= len(best_clique) after removal, we can break?
            # Yes, because P only shrinks.
            if len(R) + len(P) <= len(best_clique):
                break

    # Initial call
    # R = empty
    # P = all nodes {0, ..., N-1}
    # X = empty
    P_initial = set(range(N))
    bron_kerbosch(set(), P_initial, set())
    
    # Convert best_clique (indices) back to values (1-indexed)
    # The problem asks for a subset of [1, 512].
    result = sorted([v + 1 for v in best_clique])
    
    # Ensure non-empty (should be, as 1 is always compatible with everyone? 
    # lcm(1, x) = x <= 512. So {1} is always a clique. 
    # Actually, is 1 compatible with all? Yes.
    # So best_clique will never be empty.)
    
    if not result:
        return [1]
        
    return result

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(len(res))
