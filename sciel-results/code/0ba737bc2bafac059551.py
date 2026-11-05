"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _has_triangle_circulant(n, S):
    """Check if circulant graph C_n(S) has a triangle."""
    # Triangle exists if there are a, b in S such that a+b in S
    # (Since edges are i ~ i+a, triangle 0-a-(a+b) requires a, b, a+b in S)
    S_set = set(S)
    for a in S:
        for b in S:
            if (a + b) % n in S_set:
                # Check if a, b, a+b are distinct and form a triangle
                # 0, a, a+b. Distinct if a!=0, b!=0, a+b!=0 mod n.
                # If a+b == 0, then b = -a. Triangle 0, a, 0? No, 0 is vertex.
                # Vertices 0, a, a+b.
                # If a+b == 0, vertices 0, a, 0. Not a triangle.
                # If a+b == a, b=0. Not in S.
                # If a+b == b, a=0. Not in S.
                # So if a+b in S and a+b not in {0, a, b}, it's a triangle.
                if (a + b) % n != 0 and (a + b) % n != a and (a + b) % n != b:
                    return True
    return False

def _alpha_circulant(n, S):
    """Estimate or calculate independence number of C_n(S)."""
    # For small n, we can use bitmask or simple greedy + local search
    # But for n up to 40, bitmask is fast enough for verification.
    # adj_mask[i] = bitmask of neighbors
    adj_mask = [0] * n
    for i in range(n):
        for s in S:
            j = (i + s) % n
            adj_mask[i] |= (1 << j)
    
    # Max Independent Set using Branch and Bound
    # candidates: bitmask of vertices available
    # current_size: size of current independent set
    best = 0
    
    def dfs(candidates, current_size):
        nonlocal best
        if current_size + bin(candidates).count('1') <= best:
            return
        if candidates == 0:
            if current_size > best:
                best = current_size
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with max degree in candidates?
        # Or just lowest bit for simplicity in this search phase
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        
        # Option 1: Include v
        # New candidates are those not adjacent to v
        new_cand = candidates & ~adj_mask[v] & ~v_bit
        dfs(new_cand, current_size + 1)
        
        # Option 2: Exclude v
        dfs(candidates & ~v_bit, current_size)

    all_mask = (1 << n) - 1
    dfs(all_mask, 0)
    return best

def _verify_circulant(n, S):
    if _has_triangle_circulant(n, S):
        return False
    alpha = _alpha_circulant(n, S)
    if alpha >= 10:
        return False
    return True

def _circulant_to_edges(n, S):
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if i < j:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try to find a valid circulant graph for n from 40 down to 20
    # We prioritize larger n
    
    for n in range(40, 20, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random sum-free sets S
        # S should be symmetric? No, if S is not symmetric, we add -S to make it undirected?
        # Standard circulant definition: edges (i, i+s) for s in S.
        # For undirected, if s in S, -s should be in S? Or we just iterate all s.
        # If we define edges for all s in S, and S contains both s and -s, it's fine.
        # To ensure triangle-free, we just need no a,b in S such that a+b in S.
        
        found = False
        # Number of attempts depends on n
        max_attempts = 2000 if n < 35 else 5000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Generate a random subset of {1, ..., n-1}
            # To be efficient, we can start with a small set and grow it
            # Or just pick random size
            # Typical density for triangle-free is low
            # Max size of sum-free set in Z_n is roughly n/2?
            # But we need low independence number.
            
            # Strategy: Randomly select a set S
            # Ensure symmetry: if s in S, add (n-s) % n
            # Check sum-free property during generation?
            
            S = set()
            # Try to build S incrementally
            # Randomly shuffle 1..n-1
            candidates = list(range(1, n))
            random.shuffle(candidates)
            
            for c in candidates:
                # Check if adding c (and -c) breaks sum-free property
                # New elements: c, -c
                # Check pairs (x, y) in S U {c, -c}
                # Specifically:
                # 1. c + c = 2c in S?
                # 2. c + (-c) = 0 (ok)
                # 3. c + s in S for any s in S?
                # 4. -c + s in S for any s in S?
                # 5. c + (-c) ...
                
                # Let's just check if S U {c, n-c} is sum-free
                new_S = set(S)
                new_S.add(c)
                new_S.add((n - c) % n)
                
                is_sum_free = True
                # Check all pairs a, b in new_S
                # If a+b in new_S and a+b not in {0, a, b}, it's a triangle
                # Optimization: only check pairs involving new elements?
                # No, simple check for small S is fine.
                for a in new_S:
                    for b in new_S:
                        s = (a + b) % n
                        if s in new_S:
                            if s != 0 and s != a and s != b:
                                is_sum_free = False
                                break
                    if not is_sum_free:
                        break
                
                if is_sum_free:
                    S = new_S
                    # We might stop adding if S gets too large?
                    # Large S -> high degree -> low alpha.
                    # But we need to check alpha.
                    
            # If S is empty, skip
            if not S:
                continue
                
            # Verify triangle-free (redundant if we built it sum-free, but safe)
            if _has_triangle_circulant(n, list(S)):
                continue
                
            # Check Independence Number
            alpha = _alpha_circulant(n, list(S))
            if alpha < 10:
                # Found a valid graph!
                best_n = n
                best_edges = _circulant_to_edges(n, list(S))
                found = True
                break
        
        if found:
            # We found a valid graph for this n.
            # Can we try to improve alpha or just accept?
            # Since we go from high n to low n, the first found is the best n.
            # We can try to find a better S for the same n?
            # Not necessary if n is maxed out.
            # But maybe we can find n=40?
            # If we found n=39, we can stop.
            # The loop breaks on n descent.
            break
    
    # Fallback: If no circulant graph found, use the known Andrásfai graph (n=26)
    if best_n == 0:
        n0 = 26
        edges0 = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        best_edges = edges0
    else:
        # Final verification of the found circulant graph
        # Just in case
        pass # Already verified in loop

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
