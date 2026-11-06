import numpy as np
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120.0
    safety_margin = 2.0  # seconds before deadline to stop
    
    def build_paley(p):
        """Build Paley tournament on p vertices."""
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        # adjacency: adj[i][j] = True if i -> j
        adj = np.zeros((p, p), dtype=bool)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i][j] = True
                else:
                    adj[j][i] = True
        return adj
    
    def orientation_bits(adj):
        """Convert adjacency matrix to orientation bits list."""
        n = len(adj)
        bits = []
        for i in range(n):
            for j in range(i + 1, n):
                bits.append(0 if adj[i][j] else 1)
        return bits
    
    def check_valid(adj, n):
        """Check if every 4-set has a common dominator.
        A dominator of a 4-set {a,b,c,d} is a vertex v (in the set) such that
        v -> a, v -> b, v -> c, v -> d (v dominates all other three in the set).
        """
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Check if any vertex in the quad dominates the other three
            # v dominates u if adj[v][u] is True
            if (adj[a][b] and adj[a][c] and adj[a][d]):
                continue
            if (adj[b][a] and adj[b][c] and adj[b][d]):
                continue
            if (adj[c][a] and adj[c][b] and adj[c][d]):
                continue
            if (adj[d][a] and adj[d][b] and adj[d][c]):
                continue
            return False
        return True
    
    def prune_tournament(adj, n, max_time=None):
        """Try to remove vertices that are not unique dominators of any 4-set."""
        if max_time is None:
            max_time = deadline - safety_margin
        
        current_n = n
        adj = adj.copy()
        removed = set()
        
        # We need to find vertices that can be removed while maintaining validity.
        # A vertex v can be removed if, for every 4-set containing v, there is another
        # vertex in that 4-set that dominates the other three.
        # More simply: try removing vertices one by one and check validity.
        # This is O(n^5) per removal which is too slow.
        
        # Alternative: Greedy removal.
        # For each vertex v, check if it's "redundant".
        # v is redundant if for every triple (a,b,c) in V\{v}, at least one of {a,b,c}
        # dominates the other two AND also dominates v? No, that's not right.
        
        # The condition is: Every 4-set S has a dominator in S.
        # If we remove v, we must ensure that every 4-set S that contained v
        # still has a dominator in S\{v}.
        # This means for every triple T in V\{v} such that T ∪ {v} was a 4-set,
        # T must have a dominator in T.
        # But wait, T is a 3-set. A 3-set doesn't need to have a dominator, only 4-sets.
        # So if we remove v, we only care about 4-sets that do NOT contain v.
        # Those 4-sets were already valid before.
        # The 4-sets that CONTAINED v are now reduced to 3-sets, so they no longer
        # need to satisfy the condition.
        # Therefore, if the original tournament was valid, ANY subset is valid!
        
        # Let me re-read the task.
        # "Every set of 4 vertices must have a vertex with arcs to all four."
        # This means for any 4 distinct vertices {a,b,c,d}, there exists v in {a,b,c,d}
        # such that v->a, v->b, v->c, v->d.
        # If we take a subset W of V, any 4-set S in W is also a 4-set in V.
        # Since S had a dominator in V, and S ⊆ V, that dominator is in S ⊆ W.
        # So W is valid.
        
        # This means we can just return any subset of the valid Paley tournament.
        # The lower bound is 31. So we should return n=31.
        
        # Let's verify this logic is sound.
        # If f(4) is the minimum n such that there exists a tournament on n vertices
        # where every 4-set has a dominator, then any larger valid tournament
        # contains valid sub-tournaments.
        # So if Paley(83) is valid, then Paley(31) (as an induced sub-tournament) is valid.
        
        # So the answer is simply n=31, taking the first 31 vertices of Paley(83).
        
        return adj[:n, :n]
    
    # Step 1: Build Paley(83)
    adj = build_paley(83)
    
    # Step 2: Verify it's valid (optional, but good for sanity check)
    # check_valid(adj, 83) -- this is expensive, skip for now.
    
    # Step 3: Return a subset of size 31
    # Since any subset of a valid tournament is valid, n=31 should work.
    n_target = 31
    sub_adj = adj[:n_target, :n_target]
    
    # Verify the subset is valid (just in case my logic is wrong)
    if time.time() < deadline - safety_margin:
        if not check_valid(sub_adj, n_target):
            # If n=31 is not valid, try larger n
            for n_try in range(32, 84):
                if time.time() > deadline - safety_margin:
                    break
                sub_adj = adj[:n_try, :n_try]
                if check_valid(sub_adj, n_try):
                    n_target = n_try
                    break
    
    bits = orientation_bits(sub_adj)
    return {'n': n_target, 'orientation': bits}
# EVOLVE-BLOCK-END
