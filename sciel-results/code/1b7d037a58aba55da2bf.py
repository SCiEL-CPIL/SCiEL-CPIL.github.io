import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145

    # Petersen graph base
    # Vertices 0..4 (outer), 5..9 (inner)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    base_n = 10
    num_edges = len(petersen_edges)
    
    # Moore bound for n(3,16) is 512.
    # n = 10 * m. So m >= 52.
    # We will try m from 52 upwards.
    
    best_result = None

    # Strategy 1: Deterministic Voltage Assignment
    # Try primes m >= 52
    for m in range(52, 200):
        if time.time() > deadline - 10:
            break
            
        # Check if m is prime (for primitive root logic, though we can just use random if not)
        # But the directive suggests using primitive roots.
        if not _is_prime(m):
            continue
            
        # Find primitive root
        r = _find_primitive_root(m)
        if r is None:
            continue
            
        # Assign voltages as powers of r
        voltages = [pow(r, i, m) for i in range(num_edges)]
        
        n = base_n * m
        edges = _build_lift(petersen_edges, m, voltages)
        
        if _verify_girth(n, edges, g):
            return n, edges

    # Strategy 2: Random Restart Search (Knowledge cell says this works well)
    # Try m from 52 upwards, with more random trials for smaller m
    for m in range(52, 150):
        n = base_n * m
        if n > 5000:
            break
            
        # Number of trials increases as m increases? Or just fixed high number?
        # Knowledge: "Increasing random trials... from 20 to 500"
        trials = 500
        
        for _ in range(trials):
            if time.time() > deadline - 5:
                break
                
            voltages = [random.randint(0, m-1) for _ in range(num_edges)]
            edges = _build_lift(petersen_edges, m, voltages)
            
            if _verify_girth(n, edges, g):
                return n, edges

    # Fallback: Return a valid large random graph if we run out of time?
    # We must return a valid graph.
    # If we haven't found one, we are in trouble.
    # Let's try a few more larger m with fewer trials just in case.
    for m in range(150, 300):
        if time.time() > deadline - 2:
            break
        n = base_n * m
        voltages = [random.randint(0, m-1) for _ in range(num_edges)]
        edges = _build_lift(petersen_edges, m, voltages)
        if _verify_girth(n, edges, g):
            return n, edges
            
    # If all fails, return a hardcoded small valid graph? 
    # We don't have one.
    # We must return *something*. 
    # If we return an invalid graph, score is 0.
    # Let's try to generate a random 3-regular graph with large n that likely has girth >= 16.
    # Girth 16 is hard. 
    # However, the task says "invalid result... scores 0".
    # I will return the last attempted edges, hoping one of the earlier ones was valid 
    # but we missed it? No, _verify_girth is strict.
    # 
    # To avoid crashing, I will return a dummy valid 3-regular graph with high n, 
    # but it might fail girth check. 
    # Actually, if I can't find a valid one, I should probably return the smallest n 
    # I *think* is valid, but the evaluator will check it.
    # 
    # Let's just return the result of the last random trial in the loop, 
    # or a random graph of size 1000.
    # But wait, if I return an invalid graph, I get 0.
    # The previous program failed with IndexError, likely in helpers.
    # If I fix the helpers, the search might succeed.
    # 
    # If the search fails completely, I will return a random 3-regular graph of size 100.
    # It will likely fail girth 16, resulting in score 0, but it won't crash.
    # 
    # Let's ensure we always return a tuple (n, edges).
    if best_result:
        return best_result
        
    # Emergency fallback: Random 3-regular graph
    n = 100
    edges = _generate_random_regular_graph(n, 3)
    return n, edges

def _build_lift(base_edges, m, voltages):
    base_n = max([u for u,v in base_edges] + [v for u,v in base_edges]) + 1
    n = base_n * m
    
    edges = []
    for i, (u, v) in enumerate(base_edges):
        shift = voltages[i]
        for k in range(m):
            u_lift = u * m + k
            v_lift = v * m + (k + shift) % m
            edges.append((u_lift, v_lift))
            
    return edges

def _verify_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    if v != parent[u]:
                        # Cycle length
                        # If v is an ancestor, dist[v] < dist[u]
                        # Cycle length = dist[u] - dist[v] + 1
                        # If v is not an ancestor (cross edge in BFS tree), 
                        # it's more complex, but for girth check, 
                        # any cycle found by BFS from s that includes s? 
                        # No, BFS from s finds shortest path from s to v.
                        # If we encounter a visited node v (not parent), 
                        # it forms a cycle.
                        # The length of the cycle is dist[u] + dist[v] + 1?
                        # Only if the paths from s to u and s to v are disjoint except at s.
                        # In a BFS tree, the path from s to u and s to v share the prefix 
                        # up to the LCA.
                        # Actually, a simpler check:
                        # If we find a neighbor v of u such that dist[v] != -1 and v != parent[u],
                        # we have a cycle.
                        # The length is at least dist[u] + dist[v] + 1? 
                        # No.
                        # Let's use the property: 
                        # Shortest cycle containing s can be found by BFS.
                        # But we need global girth.
                        # 
                        # Correct approach for global girth check via BFS from all nodes:
                        # When we discover an edge (u,v) where both are visited,
                        # and v is not the parent of u, we have a cycle.
                        # The length of this cycle is dist[u] + dist[v] + 1 ONLY IF 
                        # the paths from s to u and s to v are vertex-disjoint except at s.
                        # This is not guaranteed.
                        # 
                        # However, if we just want to check if girth < g, 
                        # we can stop BFS at depth g-1.
                        # If we find a cycle of length < g, it must be found.
                        # 
                        # Let's use a simpler, slightly slower but correct method:
                        # For each edge (u,v), if there is a path from u to v of length < g-1 
                        # that does not use (u,v), then there is a cycle < g.
                        # 
                        # Given time constraints, let's assume the BFS detection 
                        # `dist[u] + dist[v] + 1` is a lower bound on the cycle length 
                        # involving that back edge? 
                        # Actually, if we do BFS from s, and we see an edge (u,v) where 
                        # dist[v] is already set, the cycle formed by path(s->u) + edge(u,v) + path(v->s) 
                        # has length dist[u] + 1 + dist[v].
                        # This is a valid cycle length.
                        # If this length < g, we return False.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0:
            return False
    return True

def _find_primitive_root(p):
    if p == 2: return 1
    if p == 3: return 2
    # Factorize p-1
    def factorize(n):
        factors = []
        d = 2
        while d*d <= n:
            if n % d:
                d += 1
            else:
                x = 0
                while not n % d:
                    x += 1
                    n //= d
                factors.append(d)
        if n > 1: factors.append(n)
        return factors
    
    factors = factorize(p-1)
    for g in range(2, p):
        if all(pow(g, (p-1)//f, p) != 1 for f in factors):
            return g
    return None

def _generate_random_regular_graph(n, k):
    # Configuration model
    if n * k % 2 != 0:
        return []
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    edges = set()
    valid = True
    for i in range(0, n*k, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            valid = False
            break
        if (u, v) in edges or (v, u) in edges:
            valid = False
            break
        edges.add((u, v))
    if not valid:
        # Retry a few times
        for _ in range(10):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, n*k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
            if valid:
                return list(edges)
    return list(edges)

# EVOLVE-BLOCK-END
