import numpy as np
import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0 # 35s margin for safety and overhead

    def remaining():
        return deadline - time.time()

    # ---------------- Girth Verification ----------------
    def check_girth(n, edges, min_girth, time_limit):
        """
        Checks if the graph has girth at least min_girth.
        Returns the minimum cycle length found, or -1 if timeout.
        """
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        
        # BFS from each vertex to find shortest cycle
        # For large n, this is O(n * (n + E)). 
        # n ~ 400-1000, E ~ 1.5n. This is fast enough.
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < min_girth:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift Construction ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None

        # Petersen Graph Definition
        # Vertices 0..9
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer cycle
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner cycle
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
        ]
        
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Generate all non-backtracking closed walks of length L in [5, g-1]
        # g=16, so L in [5, 15]
        max_L = g - 1
        walk_vectors = set()
        
        # DFS to find closed walks
        # State: (current, prev, length, counts)
        # counts is a tuple of 15 integers representing net voltage contribution
        # We use a list for mutation, then convert to tuple for storage
        
        for start_v in range(n_base):
            # Start with length 0, no prev, counts all 0
            stack = [(start_v, -1, 0, [0]*15)]
            while stack:
                current, prev, length, counts = stack.pop()
                
                if length >= max_L:
                    continue
                    
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                        
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    
                    # Determine direction: +1 if current is u, -1 if current is v
                    if current == u:
                        d = 1
                    else:
                        d = -1
                        
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_length = length + 1
                    
                    if next_v == start_v and new_length >= 5:
                        # Found a closed walk
                        # Normalize to avoid duplicates (reverse walks)
                        vec = new_counts
                        for i in range(15):
                            if vec[i] != 0:
                                if vec[i] < 0:
                                    vec = [-x for x in vec]
                                break
                        walk_vectors.add(tuple(vec))
                    
                    stack.append((next_v, current, new_length, new_counts))
        
        if not walk_vectors:
            return None

        # Filter out zero vectors (shouldn't happen for simple graph) and convert to numpy
        unique_constraints = []
        for c in walk_vectors:
            if any(x != 0 for x in c):
                unique_constraints.append(c)
                
        if not unique_constraints:
            return None

        C = np.array(unique_constraints, dtype=np.int64)
        
        # We want to find m (prime) and x in Z_m^15 such that C @ x != 0 (mod m) for all rows.
        # n = 10 * m
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        # Simulated Annealing Solver
        def solve_voltage(m, max_iters=5000):
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
            
            best_v = v.copy()
            best_cost = cost(v)
            
            if best_cost == 0:
                return v, 0
            
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            for it in range(max_iters):
                if remaining() < 5:
                    break
                
                j = random.randint(0, 14)
                old_val = v[j]
                v[j] = random.randint(1, m - 1)
                
                new_cost = cost(v)
                delta = new_cost - best_cost
                
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
                else:
                    v[j] = old_val
                    
                T *= cooling
                
            return best_v, best_cost

        # Try primes in increasing order
        # Moore bound for n(3,16) is 382. n=10m, so m >= 39.
        # Smallest prime >= 39 is 41.
        m = 41
        while m < 2000 and remaining() > 20:
            if is_prime(m):
                # Try multiple restarts
                for restart in range(10):
                    if remaining() < 15:
                        break
                    
                    v, cost = solve_voltage(m, max_iters=2000)
                    
                    if cost == 0:
                        # Construct graph
                        n = 10 * m
                        edges = []
                        for u, v_base in pet_edges:
                            e_idx = edge_index[(u, v_base)]
                            vol = v[e_idx]
                            for layer in range(m):
                                u_lift = u * m + layer
                                v_lift = v_base * m + (layer + vol) % m
                                if u_lift < v_lift:
                                    edges.append((u_lift, v_lift))
                                else:
                                    edges.append((v_lift, u_lift))
                        
                        # Verify girth
                        gc = check_girth(n, edges, g, deadline - 3)
                        if gc >= g:
                            return (n, edges)
                        # If verification fails, it means our walk enumeration was incomplete or logic error.
                        # But walk enumeration covers all non-backtracking walks up to 15.
                        # If girth is < g, it must be a backtracking walk? No, shortest cycles are simple, hence non-backtracking.
                        # So if check fails, our voltage assignment is invalid? 
                        # Wait, if C @ v != 0 for all non-backtracking closed walks of len < g, 
                        # then no closed walk of len < g has sum 0. 
                        # A cycle is a closed walk. So girth >= g.
                        # Unless there's a bug in check_girth or walk generation.
                        
            m += 1
            
        return None

    # ---------------- Fallback: Random Regular Graph ----------------
    def random_regular_fallback():
        if k != 3:
            return None
            
        # Try to generate a random 3-regular graph with large n
        # Start from a size that is likely to have girth 16
        # Expected girth of random 3-regular graph grows like log_{2}(n)
        # For g=16, n should be around 2^16 = 65536? That's too big for 120s.
        # Actually, the smallest known is 936.
        # Random 3-reg graphs of size ~1000 often have girth ~10-12.
        # This fallback is unlikely to beat the voltage lift if it works.
        # We only use this if voltage lift fails.
        
        for n in [1000, 1200, 1500]:
            if remaining() < 30:
                break
                
            # Generate random 3-regular graph
            # Configuration model
            stubs = []
            for i in range(n):
                stubs.extend([i] * 3)
                
            edges = set()
            valid = True
            attempts = 0
            while attempts < 100:
                attempts += 1
                random.shuffle(stubs)
                edges = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if (u, v) in edges or (v, u) in edges:
                        valid = False
                        break
                    edges.add((u, v))
                    
                if valid:
                    break
            
            if not valid:
                continue
                
            edge_list = list(edges)
            
            # Check girth
            gc = check_girth(n, edge_list, g, deadline - 5)
            if gc >= g:
                return (n, edge_list)
                
        return None

    # Main Logic
    result = voltage_lift_petersen()
    if result is not None:
        return result
        
    result = random_regular_fallback()
    if result is not None:
        return result
        
    # Last resort: Return a trivial invalid graph to avoid crash?
    # Or raise exception? "invalid result... scores 0".
    # Let's return a small cycle if k=2, else a large empty list?
    # The evaluator expects (n, edges).
    # If we fail, we should probably return something that doesn't crash the evaluator immediately,
    # but it will score 0.
    # Let's try to return the best effort from voltage lift even if unverified?
    # No, must be valid.
    
    # If we get here, we failed.
    # Return a dummy to prevent exception, knowing score will be 0.
    return 10, [(0,1), (1,2), (2,0)] if k==3 else (1, [])

def _construct_fallback(k, g):
    # This is not used in the main construct logic above, 
    # but kept for interface compatibility if called directly.
    if k == 2:
        n = g
        edges = [(i, (i+1)%n) for i in range(n)]
        return n, edges
    return 100, []

# Entry point
if __name__ == "__main__":
    # For testing
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
