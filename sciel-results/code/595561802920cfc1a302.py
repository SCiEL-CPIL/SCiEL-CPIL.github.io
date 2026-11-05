import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute conflict graph: edge (u,v) if (u*v) % (u+v) == 0
    # Use the characterization: let g=gcd(u,v), u'=u/g, v'=v/g, conflict iff (u'+v') | g
    conflicts = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = 0
            a, b = i, j
            while b:
                a, b = b, a % b
            g = a
            u_prime = i // g
            v_prime = j // g
            if g % (u_prime + v_prime) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Convert to sorted lists for faster access
    conflict_lists = [sorted(conflicts[i]) for i in range(N + 1)]
    
    def is_valid_set(s):
        """Check if set s is an independent set."""
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if s[j] in conflicts[s[i]]:
                    return False
        return True
    
    # Greedy construction: start with all vertices, remove conflicts
    # Strategy: try multiple greedy orders
    
    best_set = set()
    
    # Initial greedy: process vertices in order of increasing degree, add if no conflict
    def greedy_by_order(order):
        result = set()
        for v in order:
            if all(v not in conflicts[u] for u in result):
                result.add(v)
        return result
    
    # Try degree-based ordering (ascending degree first - less constrained first)
    degrees = [len(conflicts[i]) for i in range(N + 1)]
    order_asc = sorted(vertices, key=lambda x: degrees[x])
    order_desc = sorted(vertices, key=lambda x: -degrees[x])
    order_random = vertices[:]
    random.shuffle(order_random)
    order_odd = [v for v in vertices if v % 2 == 1]
    order_even = [v for v in vertices if v % 2 == 0]
    
    candidates = [
        greedy_by_order(order_asc),
        greedy_by_order(order_desc),
        greedy_by_order(order_random),
        set(order_odd),  # all odd
        set(order_even),  # all even
    ]
    
    for c in candidates:
        if len(c) > len(best_set):
            best_set = set(c)
    
    # Local search: try to add vertices, then swap
    def local_search(s, time_limit):
        s = set(s)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding each excluded vertex
            excluded = [v for v in vertices if v not in s]
            random.shuffle(excluded)
            for v in excluded:
                if all(v not in conflicts[u] for u in s):
                    s.add(v)
                    improved = True
            # Try swaps: remove one, add one or more
            for u in list(s):
                temp = s - {u}
                # Try adding any excluded vertex that doesn't conflict with temp
                for v in excluded:
                    if v not in s and all(v not in conflicts[x] for x in temp):
                        new_s = temp | {v}
                        if len(new_s) > len(s):
                            s = new_s
                            improved = True
                            break
                if improved:
                    break
        return s
    
    # Run local search from best candidate
    time_limit = deadline - 5.0
    best_set = local_search(best_set, time_limit)
    
    # Multiple restarts with random perturbations
    while time.time() < deadline - 5.0:
        # Start from current best, remove a random element, re-run local search
        perturbed = set(best_set)
        if len(perturbed) > 1:
            to_remove = random.choice(list(perturbed))
            perturbed.discard(to_remove)
        
        result = local_search(perturbed, deadline - 5.0)
        if len(result) > len(best_set):
            best_set = result
    
    # Final validation
    result_list = sorted(best_set)
    # Verify validity
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = result_list[i], result_list[j]
            if (a * b) % (a + b) == 0:
                # Should not happen, but fallback to odd numbers
                return list(range(1, 257, 2))
    
    return result_list
# EVOLVE-BLOCK-END
