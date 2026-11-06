import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1
        
        # Quick greedy lower bound
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            if best_v == -1:
                break
            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)
        
        if size >= 10:
            return True
        
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            
            # Pick vertex with max degree in candidates
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            return False
        
        return backtrack(mask_all, 0)

    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges

    def make_circulant(n, S):
        """Create circulant graph: vertex i connected to i+s mod n for s in S."""
        adj = [0] * n
        for s in S:
            s = s % n
            if s == 0:
                continue
            for i in range(n):
                j = (i + s) % n
                if j > i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def is_sum_free(S, n):
        """Check if S is sum-free mod n: no a,b,c in S with a+b≡c mod n."""
        S_set = set(s % n for s in S if s % n != 0)
        for a in S_set:
            for b in S_set:
                c = (a + b) % n
                if c in S_set:
                    return False
        return True

    def make_andrasfai(k):
        """Andrásfai graph And(k): vertices Z_{3k}, i~j iff (j-i) mod 3k in {1, 3k-1}."""
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    best_result = None

    # 1. Try known constructions
    adj9, n9 = make_andrasfai(9)
    if not has_independent_set_10(adj9, n9):
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}

    # 2. Systematic search over circulant graphs for n=40, 39, 38...
    for target_n in range(40, (best_result['n'] if best_result else 27), -1):
        if time.time() > deadline - 10:
            break
        
        # Generate candidate sum-free sets S
        # S should be "large enough" to keep alpha small
        # For circulant, alpha relates to the largest subset with no differences in S
        
        # Strategy: try various symmetric sum-free sets
        # A set S is symmetric if s in S implies n-s in S
        # For triangle-free, S must be sum-free
        
        found = False
        
        # Try different sizes of S
        # For n=39, we need alpha < 10
        # The graph is regular with degree 2*|S| (if S symmetric and doesn't contain 0)
        
        # Generate random symmetric sets and check sum-free
        max_attempts = 50000
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Random symmetric sum-free set
            # Pick a random subset of {1, ..., n//2}
            half = target_n // 2
            # Try various densities
            density = random.uniform(0.15, 0.45)
            S_half = set()
            for s in range(1, half + 1):
                if random.random() < density:
                    S_half.add(s)
            
            # Make symmetric
            S = set()
            for s in S_half:
                S.add(s)
                S.add(target_n - s)
            
            if len(S) < 4:
                continue
            
            # Check sum-free
            if not is_sum_free(S, target_n):
                continue
            
            # Create graph and check
            adj = make_circulant(target_n, S)
            
            # Quick degree check: if graph is too sparse, alpha likely >= 10
            # Average degree = 2*|S|
            # By Turan-type bounds, if avg deg is too low, alpha is large
            # For n vertices, if avg degree d, then alpha >= n^2/(2E) = n/d roughly
            # We need alpha < 10, so need d > n/10 roughly
            avg_deg = 2 * len(S_half)
            if avg_deg < target_n / 12:
                continue
            
            if not has_independent_set_10(adj, target_n):
                found = True
                best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
        
        if found:
            break
    
    # 3. If still not found large enough, try random triangle-free search
    if best_result is None or best_result['n'] < 35:
        def random_triangle_free(n, p):
            adj = [0] * n
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            for u, v in edges:
                common = adj[u] & adj[v]
                if common == 0:
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            return adj
        
        min_target = (best_result['n'] + 1) if best_result else 28
        for target_n in range(40, min_target - 1, -1):
            if time.time() > deadline - 5:
                break
            for p in [0.5, 0.6, 0.7, 0.8]:
                for _ in range(500):
                    if time.time() > deadline - 5:
                        break
                    adj = random_triangle_free(target_n, p)
                    if not has_independent_set_10(adj, target_n):
                        best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                        break
                if best_result and best_result['n'] == target_n:
                    break
            if best_result and best_result['n'] == target_n:
                break
    
    # Fallback
    if best_result is None:
        adj9, n9 = make_andrasfai(9)
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}

    # Final verification
    final_adj = [0] * best_result['n']
    for u, v in best_result['edges']:
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
    
    assert is_triangle_free(final_adj, best_result['n']), "Result is not triangle-free!"
    assert not has_independent_set_10(final_adj, best_result['n']), "Result has independent set of size 10!"
    
    return best_result
