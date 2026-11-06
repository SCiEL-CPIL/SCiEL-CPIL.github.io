import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    def get_k4s_and_ik4s():
        k4s = []
        ik4s = []
        for quad in combinations(range(n), 4):
            is_k4 = True
            is_ik4 = True
            for u, v in combinations(quad, 2):
                if adj[u] & (1 << v):
                    is_ik4 = False
                else:
                    is_k4 = False
            if is_k4:
                k4s.append(quad)
            if is_ik4:
                ik4s.append(quad)
        return k4s, ik4s

    k4s, ik4s = get_k4s_and_ik4s()
    best_n = n
    best_edges = edges[:]

    def try_add_vertex():
        # Local search for the new vertex's connections
        x = [random.randint(0, 1) for _ in range(n)]
        for _ in range(10000):
            viol_k4 = None
            viol_ik4 = None
            
            # Check violations
            for c in k4s:
                s = sum(x[u] for u in c)
                if s == 4:
                    viol_k4 = c
                    break
            if viol_k4 is None:
                for c in ik4s:
                    s = sum(x[u] for u in c)
                    if s == 0:
                        viol_ik4 = c
                        break
            
            if viol_k4 is None and viol_ik4 is None:
                return x
            
            if viol_k4 is not None:
                candidates = [u for u in viol_k4 if x[u] == 1]
                if candidates:
                    u = random.choice(candidates)
                    x[u] = 0
            else:
                candidates = [u for u in viol_ik4 if x[u] == 0]
                if candidates:
                    u = random.choice(candidates)
                    x[u] = 1
        return None

    attempts = 0
    while time.time() < deadline and n < 42:
        x = try_add_vertex()
        if x is not None:
            new_v = n
            adj.append(0)
            for u in range(n):
                if x[u]:
                    adj[new_v] |= (1 << u)
                    adj[u] |= (1 << new_v)
                    edges.append((u, new_v) if u < new_v else (new_v, u))
            n += 1
            if n > best_n:
                best_n = n
                best_edges = edges[:]
            k4s, ik4s = get_k4s_and_ik4s()
            attempts = 0
        else:
            attempts += 1
            if attempts > 20:
                # Try to perturb existing graph
                # Pick a random edge and flip it, check validity
                if edges:
                    u, v = random.choice(edges)
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edges.remove((u, v))
                    
                    # Check if still valid
                    # Simple check: recompute k4s/ik4s and see if any become size 5?
                    # This is expensive. Just restart if perturbation fails?
                    # For now, just break if stuck.
                    break
                else:
                    break

    return best_n, best_edges

# EVOLVE-BLOCK-END
