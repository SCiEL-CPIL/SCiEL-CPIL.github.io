"""Optimized tournament construction for Schütte k=4 using exact bitset checking."""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def _build_in_bits(n, orientation):
        """Build in-neighborhood bitsets from orientation list.
        in_bits[j] has bit i set if i -> j.
        """
        in_bits = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    # i -> j
                    in_bits[j] |= (1 << i)
                else:
                    # j -> i
                    in_bits[i] |= (1 << j)
        return in_bits
    
    def _count_violations(n, in_bits):
        """Count exact number of violated 4-sets using bitsets."""
        violations = 0
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                for c in range(b + 1, n - 1):
                    for d in range(c + 1, n):
                        if (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]) == 0:
                            violations += 1
        return violations
    
    def _find_violated_quad(n, in_bits, rng):
        """Find a violated 4-set. Returns None if none found (valid)."""
        # First try random sampling
        for _ in range(200):
            S = rng.choice(n, size=4, replace=False)
            a, b, c, d = sorted(S)
            if (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]) == 0:
                return (a, b, c, d)
        # If not found by sampling, do exhaustive
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                for c in range(b + 1, n - 1):
                    for d in range(c + 1, n):
                        if (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]) == 0:
                            return (a, b, c, d)
        return None
    
    def _orient_from_adj(n, adj):
        """Convert adjacency matrix to orientation list."""
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                orientation.append(0 if adj[i, j] == 1 else 1)
        return orientation
    
    def _paleyn_tournament(p):
        """Paley tournament on p vertices (p prime, p ≡ 3 mod 4)."""
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                orientation.append(0 if diff in squares else 1)
        return p, orientation
    
    def _local_search(n, orientation, max_time=30.0):
        """Local search to eliminate violations."""
        adj = np.zeros((n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    adj[i, j] = 1
                else:
                    adj[j, i] = 1
                idx += 1
        
        in_bits = _build_in_bits(n, orientation)
        rng = np.random.default_rng(42)
        
        # Get initial violation count
        violations = _count_violations(n, in_bits)
        if violations == 0:
            return n, orientation, True
        
        best_orientation = orientation[:]
        best_violations = violations
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < start_time + max_time:
            # Find a violated quad
            quad = _find_violated_quad(n, in_bits, rng)
            if quad is None:
                return n, best_orientation, True
            
            a, b, c, d = quad
            # Try flipping edges within this quad to create a dominator
            # For each vertex v not in quad, check if flipping edges to a,b,c,d helps
            # Simpler: try flipping each edge in the quad
            improved = False
            for e1 in range(4):
                for e2 in range(e1 + 1, 4):
                    u = [a, b, c, d][e1]
                    v = [a, b, c, d][e2]
                    # Flip edge between u and v
                    old_dir = adj[u, v]
                    adj[u, v] = 1 - old_dir
                    adj[v, u] = old_dir
                    # Update in_bits
                    if old_dir == 1:  # was u -> v, now v -> u
                        in_bits[v] &= ~(1 << u)
                        in_bits[u] |= (1 << v)
                    else:  # was v -> u, now u -> v
                        in_bits[u] &= ~(1 << v)
                        in_bits[v] |= (1 << u)
                    
                    new_violations = _count_violations(n, in_bits)
                    if new_violations < best_violations:
                        best_violations = new_violations
                        best_orientation = _orient_from_adj(n, adj)
                        no_improve_count = 0
                        improved = True
                    else:
                        # Revert
                        adj[u, v] = old_dir
                        adj[v, u] = 1 - old_dir
                        if old_dir == 1:
                            in_bits[v] |= (1 << u)
                            in_bits[u] &= ~(1 << v)
                        else:
                            in_bits[u] |= (1 << v)
                            in_bits[v] &= ~(1 << u)
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                # Try random edge flip
                pair_idx = rng.integers(0, n * (n - 1) // 2)
                # Find (i,j) from linear index
                i = 0
                while i < n - 1:
                    cnt = n - 1 - i
                    if pair_idx < cnt:
                        j = i + 1 + pair_idx
                        break
                    pair_idx -= cnt
                    i += 1
                else:
                    i = n - 2
                    j = n - 1
                
                old_dir = adj[i, j]
                adj[i, j] = 1 - old_dir
                adj[j, i] = old_dir
                if old_dir == 1:
                    in_bits[j] &= ~(1 << i)
                    in_bits[i] |= (1 << j)
                else:
                    in_bits[i] &= ~(1 << j)
                    in_bits[j] |= (1 << i)
                
                new_violations = _count_violations(n, in_bits)
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_orientation = _orient_from_adj(n, adj)
                    no_improve_count = 0
                else:
                    # Revert
                    adj[i, j] = old_dir
                    adj[j, i] = 1 - old_dir
                    if old_dir == 1:
                        in_bits[j] |= (1 << i)
                        in_bits[i] &= ~(1 << j)
                    else:
                        in_bits[i] |= (1 << j)
                        in_bits[j] &= ~(1 << i)
        
        # Final check
        final_in_bits = _build_in_bits(n, best_orientation)
        final_violations = _count_violations(n, final_in_bits)
        return n, best_orientation, final_violations == 0
    
    # Strategy: try multiple starting points and n values
    
    best_result = None  # (n, orientation)
    
    # Try n from 31 upward
    for n in range(31, 60):
        if time.time() > deadline - 10:
            break
        
        # Try random tournament
        rng = np.random.default_rng(n * 1000 + 42)
        orientation = rng.integers(0, 2, size=n * (n - 1) // 2).tolist()
        
        in_bits = _build_in_bits(n, orientation)
        violations = _count_violations(n, in_bits)
        
        if violations == 0:
            best_result = (n, orientation)
            break
        
        # Try local search on random
        n_ls, orient_ls, valid_ls = _local_search(n, orientation, max_time=8.0)
        if valid_ls:
            best_result = (n, orient_ls)
            break
        
        # Try Paley if n is prime ≡ 3 mod 4
        is_prime = True
        for x in range(2, int(n**0.5) + 1):
            if n % x == 0:
                is_prime = False
                break
        if is_prime and n % 4 == 3:
            n_p, orient_p = _paleyn_tournament(n)
            in_bits_p = _build_in_bits(n_p, orient_p)
            violations_p = _count_violations(n_p, in_bits_p)
            if violations_p == 0:
                best_result = (n_p, orient_p)
                break
            n_ls2, orient_ls2, valid_ls2 = _local_search(n_p, orient_p, max_time=8.0)
            if valid_ls2:
                best_result = (n_p, orient_ls2)
                break
    
    # Fallback: use Paley 67 if nothing found
    if best_result is None:
        if time.time() < deadline - 5:
            n_p, orient_p = _paleyn_tournament(67)
            in_bits_p = _build_in_bits(n_p, orient_p)
            violations_p = _count_violations(n_p, in_bits_p)
            if violations_p == 0:
                best_result = (n_p, orient_p)
            else:
                n_ls, orient_ls, valid_ls = _local_search(n_p, orient_p, max_time=20.0)
                best_result = (n_ls, orient_ls)
        else:
            n_p, orient_p = _paleyn_tournament(67)
            best_result = (n_p, orient_p)
    
    n_final, orient_final = best_result
    
    # Final verification
    in_bits_final = _build_in_bits(n_final, orient_final)
    # Quick sampled check
    rng = np.random.default_rng(999)
    for _ in range(5000):
        S = rng.choice(n_final, size=4, replace=False)
        a, b, c, d = sorted(S)
        if (in_bits_final[a] & in_bits_final[b] & in_bits_final[c] & in_bits_final[d]) == 0:
            # Found violation, try one more local search
            if time.time() < deadline - 2:
                n_final, orient_final, _ = _local_search(n_final, orient_final, max_time=3.0)
            break
    
    return {'n': n_final, 'orientation': orient_final}
