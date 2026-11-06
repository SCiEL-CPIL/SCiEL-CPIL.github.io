"""Find largest graph with no regular induced subgraph on 6+ vertices."""
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    # For a subset S of size k, it's regular if all vertices in S have the same degree within S.
    # Key: empty subgraph (k=0,1) and complete subgraph (k>=2) are always regular.
    # For k>=6, we need: NOT(all degrees in S equal)
    
    # Precompute: for each subset size k>=6, the "forbidden" degree patterns.
    # A subset S is regular iff all vertices in S have the same internal degree.
    
    def is_regular_subset(adj, subset):
        """Check if the induced subgraph on subset is regular."""
        k = len(subset)
        if k <= 1:
            return True
        # Compute degree of each vertex in subset within the induced subgraph
        sset = set(subset)
        degrees = []
        for v in subset:
            d = 0
            for u in subset:
                if u != v and adj[v][u]:
                    d += 1
            degrees.append(d)
        # Regular if all degrees equal
        return len(set(degrees)) == 1
    
    # Strategy: try to build graphs of increasing size, use backtracking
    # For n vertices, we assign edges one by one in a fixed order.
    # At each step, check if any completed 6+ subset becomes regular.
    
    # Better approach: use the fact that we can check constraints incrementally.
    # Order edges as (0,1), (0,2), ..., (0,n-1), (1,2), ...
    
    # For efficiency, precompute which subsets of size exactly 6 we need to check.
    # Actually, we need ALL subsets of size >= 6, but if no 6-subset is regular,
    # does that guarantee no larger subset is regular? NO.
    # A 7-vertex subset could be regular even if all its 6-subsets aren't.
    
    # So we must check all subsets of size >= 6.
    
    # Let's try a different approach: for a given n, do a SAT-like search.
    # Variables: edges of the complete graph on n vertices.
    # Constraints: for each subset S of size >= 6, NOT(all vertices in S have same degree in G[S]).
    
    # For n=20, number of edges = 190, number of subsets of size 6 = C(20,6) = 38760,
    # plus larger subsets. This is large but let's try.
    
    # Actually, let's think about this more carefully.
    # The constraint for subset S: it's NOT the case that all vertices in S have the same degree in G[S].
    # Equivalently: there exist two vertices u,v in S such that deg_S(u) != deg_S(v).
    
    # Let me try a randomized approach first to find a valid graph, then try to maximize n.
    
    def random_graph(n, seed=42):
        """Generate a random graph and check if it's valid."""
        import random
        rng = random.Random(seed)
        edges = []
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = adj[j][i] = True
                    edges.append([i, j])
        return adj, edges
    
    def check_valid(adj, n, verbose=False):
        """Check if graph has no regular induced subgraph on 6+ vertices."""
        # Check all subsets of size 6 to n
        vertices = list(range(n))
        for k in range(6, n+1):
            for subset in itertools.combinations(vertices, k):
                if is_regular_subset(adj, subset):
                    if verbose:
                        print(f"Regular subset of size {k}: {subset}")
                    return False
        return True
    
    # First, try to find valid graphs for n=20 using random search
    # If we find one, we know the answer is at least 20.
    
    best_n = 5
    best_result = {'n': 5, 'edges': []}
    
    # Try random graphs for n=20
    for attempt in range(1000):
        if elapsed() > DEADLINE - 10:
            break
        adj, edges = random_graph(20, seed=attempt)
        if check_valid(adj, 20):
            best_n = 20
            best_result = {'n': 20, 'edges': edges}
            if elapsed() > DEADLINE - 5:
                break
            # Try to see if n=21 is possible
            break
    
    # If we found n=20, try to find n=21
    if best_n >= 20:
        for attempt in range(500):
            if elapsed() > DEADLINE - 10:
                break
            adj, edges = random_graph(21, seed=attempt)
            if check_valid(adj, 21):
                best_n = 21
                best_result = {'n': 21, 'edges': edges}
                break
    
    # If random didn't work, try a more systematic approach for smaller n
    # to at least get a better score than 5.
    
    if best_n <= 5:
        # Try n=6,7,8,... with random search
        for target_n in range(6, 15):
            if elapsed() > DEADLINE - 10:
                break
            found = False
            for attempt in range(200):
                if elapsed() > DEADLINE - 10:
                    break
                adj, edges = random_graph(target_n, seed=attempt)
                if check_valid(adj, target_n):
                    best_n = target_n
                    best_result = {'n': target_n, 'edges': edges}
                    found = True
                    break
            if not found and target_n >= 8:
                # If we can't find valid graphs for this n, try smaller
                break
    
    if elapsed() > DEADLINE:
        return best_result
    
    return best_result
