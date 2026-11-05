"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute gcd for all pairs
    n = 128
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute all bad triples
    # A bad triple {a,b,c} has gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            gab = gcd_table[a][b]
            for c in range(b + 1, n + 1):
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bad_triples.add((a, b, c))
    
    # For each vertex, compute the set of bad triples it's in
    vertex_triples = {v: set() for v in range(1, n + 1)}
    for (a, b, c) in bad_triples:
        vertex_triples[a].add((a, b, c))
        vertex_triples[b].add((a, b, c))
        vertex_triples[c].add((a, b, c))
    
    # For each pair of vertices, compute if they're in any bad triple together
    # And for a candidate vertex v and current set S, count how many bad triples
    # would be formed by adding v to S (i.e., triples containing v and two elements of S)
    
    # Precompute: for each pair (u,v), the list of w such that {u,v,w} is a bad triple
    pair_thirds = {}
    for (a, b, c) in bad_triples:
        pair_thirds.setdefault((a, b), []).append(c)
        pair_thirds.setdefault((b, a), []).append(c)
        pair_thirds.setdefault((a, c), []).append(b)
        pair_thirds.setdefault((c, a), []).append(b)
        pair_thirds.setdefault((b, c), []).append(a)
        pair_thirds.setdefault((c, b), []).append(a)
    
    def count_conflicts(v, S):
        """Count how many bad triples would be created by adding v to set S."""
        count = 0
        for u in S:
            for w in pair_thirds.get((v, u), []):
                if w in S:
                    count += 1
        return count
    
    # Greedy construction: start with empty set, repeatedly add the vertex
    # that minimizes conflicts (and among ties, prefer lower degree)
    S = set()
    remaining = set(range(1, n + 1))
    
    while remaining and time.time() < deadline:
        best_v = None
        best_conflicts = float('inf')
        best_degree = float('inf')
        for v in remaining:
            conf = count_conflicts(v, S)
            deg = len(vertex_triples[v])
            if conf < best_conflicts or (conf == best_conflicts and deg < best_degree):
                best_conflicts = conf
                best_degree = deg
                best_v = v
        if best_v is None:
            break
        S.add(best_v)
        remaining.discard(best_v)
    
    # Local search: try 1-for-1 swaps
    def is_valid_set(S_set):
        """Check if S_set has no bad triples."""
        for (a, b, c) in bad_triples:
            if a in S_set and b in S_set and c in S_set:
                return False
        return True
    
    # Verify initial solution
    if not is_valid_set(S):
        # Fallback: just return a small valid set
        S = {1}
    
    best_size = len(S)
    best_S = set(S)
    
    # Local search with swaps
    S_list = sorted(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            for j in range(1, n + 1):
                if j in S:
                    continue
                # Try swapping S_list[i] with j
                new_S = S - {S_list[i]} | {j}
                if len(new_S) != len(S):
                    continue
                if is_valid_set(new_S):
                    # Check if this swap is beneficial (it's neutral in size,
                    # but might enable future improvements)
                    # For now, just accept if valid (to explore state space)
                    # Actually, let's only accept if it leads to a valid set
                    # and we can potentially add more later
                    pass
        
        # Try 1-for-2: remove one, add two
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            removed = S_list[i]
            S_without = S - {removed}
            # Try to add two elements
            for a in range(1, n + 1):
                if a in S_without:
                    continue
                S_one = S_without | {a}
                if not is_valid_set(S_one):
                    continue
                for b in range(a + 1, n + 1):
                    if b in S_one:
                        continue
                    S_two = S_one | {b}
                    if is_valid_set(S_two) and len(S_two) > len(S):
                        S = S_two
                        best_size = len(S)
                        best_S = set(S)
                        S_list = sorted(S)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Try to add more elements one by one
    for v in range(1, n + 1):
        if time.time() > deadline:
            break
        if v in S:
            continue
        new_S = S | {v}
        if is_valid_set(new_S):
            S = new_S
            best_size = len(S)
            best_S = set(S)
    
    # Final check
    if not is_valid_set(S):
        S = best_S if is_valid_set(best_S) else {1}
    
    return sorted(S)
# EVOLVE-BLOCK-END
