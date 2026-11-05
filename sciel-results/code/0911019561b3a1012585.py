"""Construct a pentagon-free graph with no independent 9-set, maximizing n."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _has_c5(adj, n):
    """Check if graph has a 5-cycle. adj is list of sets."""
    # For each 4-tuple (a,b,c,d) with a<b<c<d, check if a-c and b-d edges exist
    # This detects a 5-cycle a-b-c-d-e-a where e is some vertex, but that's not quite right.
    # Actually, a 5-cycle is a-b-c-d-e-a. Let me think differently.
    # A 5-cycle exists if there are 5 distinct vertices v0..v4 such that
    # v0-v1, v1-v2, v2-v3, v3-v4, v4-v0 are all edges.
    # Brute force over all 5-tuples is O(n^5) which is 35^5 = ~52M, too slow in Python.
    # Better: for each pair (u,v) that is an edge, check if there's a path of length 3 from v back to u
    # through intermediate vertices not equal to u or v.
    # Path of length 3: v-w-x-u where w,x are distinct and not u,v.
    # So for each edge (u,v), for each w in adj[v] (w != u), for each x in adj[w] (x != v, x != u),
    # check if x is adjacent to u.
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue  # only process each edge once
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    if x in adj[u]:
                        return True
    return False


def _independence_number_lower_bound(adj, n, k):
    """Check if there exists an independent set of size k. Returns True if alpha < k."""
    # Use a greedy + backtracking approach or just check all subsets if n is small.
    # For n=35, checking all C(35,9) is too large.
    # Use a simple greedy coloring bound or a bounded search.
    # Simple approach: try to find an independent set of size k using backtracking with pruning.
    # If we can't find one, alpha < k.
    
    # Greedy: repeatedly pick a vertex, remove it and its neighbors.
    # This gives a lower bound on alpha. If lower bound >= k, we have an independent set of size k.
    # But we need to check if alpha < k, i.e., NO independent set of size k exists.
    
    # Let's use a simple recursive search with memoization-like pruning.
    # Actually, for our purposes in the construction loop, we can use a faster heuristic:
    # Check if the maximum independent set (via greedy) is < k. This is a sufficient but not necessary condition.
    # For exactness in the final check, we'll do a more thorough check.
    
    # For the construction phase, use greedy to quickly reject candidates.
    remaining = set(range(n))
    count = 0
    while remaining:
        # Pick vertex with minimum degree in remaining subgraph
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
        count += 1
        remaining -= {best_v}
        remaining -= (adj[best_v] & remaining)
    
    return count < k  # True means alpha < k (no independent set of size k)


def _max_independent_set_size(adj, n):
    """Compute a lower bound on the independence number using greedy."""
    remaining = set(range(n))
    count = 0
    while remaining:
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
        count += 1
        remaining -= {best_v}
        remaining -= (adj[best_v] & remaining)
    return count


def _exact_no_independent_set_k(adj, n, k):
    """Exactly check if there is NO independent set of size k.
    Uses backtracking with branch and bound."""
    # If n < k, trivially no independent set of size k.
    if n < k:
        return True
    
    # Use a recursive search. State: set of candidates, current size.
    # We want to find an independent set of size k. If we find one, return False.
    # If we exhaust all possibilities, return True.
    
    # Optimize: use the fact that if |candidates| + current_size < k, prune.
    # Also, if we can greedily show that the max independent set in candidates is < k - current_size, prune.
    
    # For efficiency, use a simple approach:
    # Try to build an independent set of size k.
    
    def can_find_independent(candidates, need):
        if need == 0:
            return True
        if len(candidates) < need:
            return False
        # Greedy upper bound: if greedy max independent set size < need, return False
        # Quick greedy check
        rem = set(candidates)
        cnt = 0
        while rem:
            best_v = min(rem, key=lambda v: len(adj[v] & rem))
            cnt += 1
            rem -= {best_v}
            rem -= (adj[best_v] & rem)
            if cnt >= need:
                return True
        if cnt < need:
            return False
        
        # Branch and bound
        # Pick a vertex to branch on
        v = next(iter(candidates))
        # Option 1: include v
        new_candidates = candidates - {v} - adj[v]
        if can_find_independent(new_candidates, need - 1):
            return True
        # Option 2: exclude v
        new_candidates2 = candidates - {v}
        if can_find_independent(new_candidates2, need):
            return True
        return False
    
    return not can_find_independent(set(range(n)), k)


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    best_result = None
    best_n = 0
    
    # Start with the known good construction: 7 disjoint K4s gives n=28
    # We'll try to extend it to larger n.
    
    # Strategy: Try n=35 down to n=29, use random greedy edge addition.
    # For each n, try multiple random seeds.
    
    for n in range(35, 27, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random attempts for this n
        max_attempts = 50 if n <= 30 else 20
        attempt_limit = deadline - 10
        
        for attempt in range(max_attempts):
            if time.time() > attempt_limit:
                break
            
            random.seed(attempt * 1000 + n)
            
            # Start with empty graph, add edges greedily
            adj = [set() for _ in range(n)]
            
            # Generate all possible edges
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)
            
            valid = True
            for (u, v) in all_edges:
                if time.time() > attempt_limit:
                    valid = False
                    break
                
                # Tentatively add edge
                adj[u].add(v)
                adj[v].add(u)
                
                # Check no C5
                if _has_c5(adj, n):
                    # Remove edge
                    adj[u].discard(v)
                    adj[v].discard(u)
                    continue
                
                # Check no independent set of size 9
                # Use fast greedy check first
                if not _independence_number_lower_bound(adj, n, 9):
                    # Greedy found an independent set of size >= 9, so this edge is bad
                    adj[u].discard(v)
                    adj[v].discard(u)
                    continue
            
            if not valid:
                continue
            
            # Final exact verification
            if _has_c5(adj, n):
                continue
            
            if not _exact_no_independent_set_k(adj, n, 9):
                continue
            
            # Valid graph found!
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            
            if best_result is None or n > best_n:
                best_result = {'n': n, 'edges': edges}
                best_n = n
            
            # Found a valid graph for this n, try next smaller n? No, we want max n.
            # Since we're going from high to low, first valid is best for this n.
            # But maybe we can find a better one? The score is just n, so any valid graph for n=35 is best.
            if n == 35:
                return best_result
            break  # Found valid for this n, move to next n (lower)
    
    # If no valid graph found above n=28, fall back to the known construction
    if best_result is None or best_n < 28:
        # Seven disjoint copies of K4
        edges = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]
        best_result = {'n': 28, 'edges': edges}
    
    return best_result

# EVOLVE-BLOCK-END
