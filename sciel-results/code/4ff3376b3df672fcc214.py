import time
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a search over the algebraic family of 3-regular circulant graphs.
    """
    start_time = time.time()
    deadline = start_time + 110

    # For k=3, n must be even.
    # The Moore bound for g=16 is 510.
    # Best known is 936.
    # We will search for n in a range starting from a reasonable lower bound.
    # Since 936 is the best known, we start a bit higher to be safe, or just start from 512.
    # However, finding a 3-regular circulant with girth 16 and n < 936 is unlikely (impossible if 936 is optimal).
    # But we don't know if 936 is optimal for circulants. It might be larger.
    # Let's start from n=512 and go up.
    # Actually, girth 16 requires n >= 510.
    # Let's try to find the smallest n.
    
    # To be efficient, we can skip n values that are impossible.
    # For a 3-regular graph with girth g, n >= 2*(1 + 2 + ... + 2^(g/2-1)) = 2*(2^(g/2)-1) = 2^(g/2+1)-2.
    # For g=16, MB = 510.
    
    # We will iterate n from 512 to, say, 1500.
    # For each n, we try random sets of 3 generators.
    
    # Optimization:
    # If we find a graph with n, we can stop.
    # To maximize score (minimize n), we should find the smallest n possible.
    
    # However, if we start from 512, we might waste time if no solution exists there.
    # Given the known 936, it is very likely that no circulant exists below 936 with girth 16.
    # But let's try a few small ones just in case, or jump to near 936.
    # Actually, let's just scan.
    
    # To speed up, we can use a heuristic:
    # Try n values that are multiples of 6? No, just even.
    
    # Let's try n in range(512, 1000, 2) first.
    # If not found, extend to 1500.
    
    # For each n, we try random generator sets.
    # Number of trials per n: 100-500.
    
    # Girth check:
    # BFS from each vertex, track distance and parent.
    # If we find a back-edge to a vertex at distance d, cycle length is d + dist[v] + 1.
    # We want min cycle length >= g.
    
    # To speed up girth check, we can stop early if we find a cycle < g.
    
    # Let's implement a fast girth checker.
    
    def check_girth(n, adj, target_girth):
        # adj is list of lists of neighbors
        min_girth = float('inf')
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < target_girth:
                                return min_girth
            if min_girth < target_girth:
                break
                
        return min_girth

    # We will try to find the smallest n.
    # To be safe and fast, we will start from a reasonable guess.
    # Since 936 is the known best, let's start from 900 or so?
    # No, let's start from 512 but limit trials per n.
    # If we don't find one in 512-1000, we might miss a smaller one, but it's unlikely.
    # Actually, if we find one at n=936, we stop.
    # If we find one at n=934, we stop.
    
    # Let's try n from 512 to 1500.
    # But to save time, we can prioritize n close to the known best if early ones fail quickly.
    
    # Actually, a better strategy:
    # 1. Try n in [512, 1000] with limited trials (e.g., 50 each).
    # 2. If not found, try n in [1000, 1500] with more trials.
    
    # But wait, if the optimal is 936, and it's a circulant, we should find it.
    # Is the 936 cage a circulant?
    # The Exoo et al. graph might not be circulant.
    # If it's not circulant, our search will fail to find it.
    # In that case, we need a fallback.
    # The fallback is the LUW construction from the previous code.
    
    # So:
    # 1. Try circulant search.
    # 2. If not found, use LUW.
    
    # Let's implement the circulant search first.
    
    best_n = float('inf')
    best_edges = None
    
    # Range to search
    # Let's start from 512.
    # To avoid wasting too much time, we'll limit the total number of trials.
    # Total budget: ~100s.
    # Each girth check is O(n^2) worst case, but usually fast.
    # For n=1000, BFS is fast.
    
    # Let's try n in range(512, 1200, 2).
    
    for n in range(512, 1200, 2):
        if time.time() > deadline - 10:
            break
            
        # Try random generator sets
        # Generators must be distinct and not inverses.
        # Also, the graph must be connected.
        
        # We can pre-filter some invalid sets?
        # For now, just random.
        
        trials = 100
        found = False
        
        for _ in range(trials):
            if time.time() > deadline - 5:
                break
                
            # Choose 3 distinct generators from 1 to n/2-1
            # Ensure no inverses: if a is chosen, n-a is not chosen.
            # Since we pick from 1..n/2-1, inverses are > n/2, so no conflict within the set.
            # Wait, if n is even, n/2 is its own inverse.
            # So we should avoid n/2 if we want simple graph?
            # If a = n/2, then 2a = 0 mod n.
            # This creates a 2-cycle? No, it's a loop in the Cayley graph sense?
            # In a simple graph, (u, u+a) is an edge.
            # If a = n/2, then u+n/2 is the other half.
            # It's a valid edge.
            # But if we have 3 generators, we just need them to be distinct.
            
            # Let's pick 3 distinct random integers from 1 to n-1.
            # Then take min(x, n-x) to get the "shortest" generator.
            # But we need to ensure the set of 6 elements {a, n-a, b, n-b, c, n-c} has size 6.
            # This means a != n-a (so a != 0, a != n/2).
            # So we should avoid 0 and n/2.
            
            valid_generators = [i for i in range(1, n) if i != n // 2]
            
            # Pick 3 distinct from valid_generators
            # But we want to avoid symmetry?
            # Just pick 3 distinct.
            
            try:
                gens = random.sample(valid_generators, 3)
            except ValueError:
                continue
                
            # Check if any are inverses of each other?
            # If a + b = 0 mod n, then b = n-a.
            # Since we picked distinct, a != b.
            # But if b = n-a, then {a, n-a} is a pair.
            # The set of neighbors would be {a, n-a, b, n-b, ...}.
            # If b = n-a, then n-b = a.
            # So the set of neighbors is {a, n-a, a, n-a, ...} -> only 2 distinct neighbors?
            # No, the generators are a, b, c.
            # Neighbors are u+a, u-a, u+b, u-b, u+c, u-c.
            # If b = n-a, then u+b = u-a and u-b = u+a.
            # So we only have 4 distinct neighbors: u+a, u-a, u+c, u-c.
            # This makes the graph 4-regular, not 3-regular?
            # No, the degree is the number of distinct neighbors.
            # If the generators are not "independent" in this sense, the degree is < 6?
            # Wait, a 3-regular graph has degree 3.
            # A circulant graph with generators S has degree |S| if S is symmetric (S = -S).
            # If S is not symmetric, the degree is 2|S|.
            # We want degree 3.
            # This is impossible for a circulant graph!
            # Because the degree of a circulant graph is always even (if we consider undirected edges, each generator a adds edges (u, u+a) and (u, u-a) which are the same edge viewed from different ends, but in the adjacency list of u, they are distinct neighbors unless a = -a mod n).
            # If a = -a mod n (i.e., 2a = 0 mod n), then u+a = u-a, so it's one neighbor.
            # So to have degree 3, we need one generator a such that 2a = 0 mod n (so a = n/2), and two other generators b, c such that b != -b, c != -c, b != -c, etc.
            # So one generator must be n/2.
            
            # Let's re-evaluate.
            # Degree 3 circulant:
            # One generator is n/2.
            # Two other generators b, c.
            # Neighbors: u+n/2, u+b, u-b, u+c, u-c.
            # This is 5 neighbors?
            # No, the degree is the number of distinct neighbors.
            # If b, c are distinct and not inverses, we have 5 neighbors.
            # We want degree 3.
            # This implies that the "generators" in the Cayley graph sense must form a set S such that the number of distinct elements in S \cup -S is 3.
            # This is impossible because if x is in S, -x is also in the neighbor set.
            # So the size of S \cup -S must be even, unless there are elements x such that x = -x.
            # If x = -x, it contributes 1.
            # If x != -x, it contributes 2.
            # So degree = k1 + 2*k2, where k1 is the number of self-inverse generators, k2 is the number of non-self-inverse pairs.
            # We want degree 3.
            # 3 = 1 + 2*1.
            # So we need one self-inverse generator (n/2) and one pair of non-self-inverse generators (b, -b).
            # So the generators are {n/2, b}.
            # This is a 2-generator circulant.
            # S = {n/2, b, -b}.
            # Neighbors: u+n/2, u+b, u-b.
            # This is 3 distinct neighbors (assuming b != n/2 and b != 0).
            
            # So we need to search for n and b such that the circulant graph with generators {n/2, b} has girth >= g.
            
            # This simplifies the search significantly!
            # We only need to choose b.
            
            # Let's restart the inner loop.
            
            a = n // 2
            # b can be any integer from 1 to n-1, excluding 0 and a.
            # Also, b and -b are the same pair.
            # So we can just try b in 1..n/2-1.
            
            for b in range(1, n // 2):
                if time.time() > deadline - 5:
                    break
                    
                # Build adjacency list
                adj = [[] for _ in range(n)]
                for u in range(n):
                    v1 = (u + a) % n
                    v2 = (u + b) % n
                    v3 = (u - b) % n
                    
                    # Ensure distinctness (should be distinct if b != 0, b != a)
                    # a = n/2.
                    # If b = n/2, then v2 = v1. But b < n/2, so b != a.
                    # So v1, v2, v3 are distinct.
                    
                    adj[u].append(v1)
                    adj[u].append(v2)
                    adj[u].append(v3)
                
                # Check girth
                girth = check_girth(n, adj, g)
                
                if girth >= g:
                    # Found a valid graph
                    best_n = n
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_edges = edges
                    found = True
                    break
                    
            if found:
                break
                
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback: LUW Construction
    # Re-implementing LUW from the previous code
    
    def _luw_construct(k, d):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try LUW with d=7
    try:
        n, edges = _luw_construct(3, 7)
        # Verify girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        girth = check_girth(n, adj, g)
        if girth >= g:
            return n, edges
    except Exception:
        pass

    # If all else fails, return a dummy or raise error
    # But we should have found something.
    raise Exception("Failed to construct graph")

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices, {len(edges)} edges.")
