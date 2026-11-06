"""
Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples 
with no red K4^(3) and no blue K5^(3).

Strategy:
1. Represent the coloring of triples (a,b,c) as a low-rank binary tensor:
   color(a,b,c) = XOR_{k=1}^K (u_k[a] * v_k[b] * w_k[c])
   where u, v, w are binary matrices of size n x K.
   This reduces the search space from 2^(n choose 3) to 3 * n * K bits.
   
2. For a target n (e.g., 34 or 35), initialize random basis vectors.
3. Use stochastic local search to flip bits in u, v, w to minimize violations.
   A flip in u_k[a] changes the color of all triples (a, b, c) for all b,c.
   We can efficiently compute the change in violations (delta) for such a flip.
   
4. If we find a valid coloring for n=35, return it. Else return best n (likely 34).
"""

import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # seconds; return with margin

def construct():
    def _build_data(n):
        """Precompute quad and pent structures for fast violation counting."""
        m = n * (n - 1) * (n - 2) // 6
        quads = []
        for q in combinations(range(n), 4):
            a, b, c, d = q
            quads.append((a, b, c, d))
        
        pents = []
        for p in combinations(range(n), 5):
            pents.append(p)
            
        return quads, pents

    def _compute_colors(n, K, u, v, w):
        """
        Compute the full coloring array of size m from the low-rank basis.
        color(a,b,c) = XOR_{k} (u[k,a] & v[k,b] & w[k,c])
        """
        # u, v, w are (K, n) arrays
        # We want an array of size (n, n, n) effectively, but flattened.
        # Efficient way:
        # T[a,b,c] = sum_k u[k,a] v[k,b] w[k,c] mod 2
        # This is a rank-K tensor.
        
        # We can compute this using matrix multiplication tricks or loops.
        # Given n <= 40 and K is small (e.g., 5-10), loops might be okay if vectorized.
        
        # Let's construct the full 3D array T of shape (n, n, n)
        # T[a,b,c] = 0
        # for k in range(K):
        #    T += u[:, a] * v[:, b] * w[:, c]  -> this is not quite right for broadcasting
        
        # Better:
        # T = np.zeros((n, n, n), dtype=np.int8)
        # for k in range(K):
        #    uk = u[k] # shape (n,)
        #    vk = v[k]
        #    wk = w[k]
        #    # Outer product of uk, vk, wk
        #    # T += uk[:, None, None] * vk[None, :, None] * wk[None, None, :]
        #    T += (uk[:, None, None] * vk[None, :, None] * wk[None, None, :]) % 2
        
        # Then extract the upper triangular part (a<b<c).
        
        T = np.zeros((n, n, n), dtype=np.int8)
        for k in range(K):
            uk = u[k]
            vk = v[k]
            wk = w[k]
            T += (uk[:, None, None] * vk[None, :, None] * wk[None, None, :]) % 2
        
        # Flatten to 1D in lexicographic order of triples (a,b,c) with a<b<c
        colors = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    colors.append(T[a, b, c])
        return np.array(colors, dtype=np.int8)

    def _count_violations(n, colors, quads, pents):
        """Count red K4 and blue K5 violations."""
        # Map triple (a,b,c) to index in colors
        # We need a fast lookup or precomputed indices for quads/pents.
        
        # Precompute indices for quads and pents
        # This is expensive to do inside the loop, so we should precompute 
        # the indices for each quad/pent once.
        
        pass # Placeholder, see _build_index_data

    def _build_index_data(n):
        """Precompute indices of triples in quads and pents."""
        # Create a dictionary mapping (a,b,c) to index in lexicographic list
        # m = n(n-1)(n-2)/6
        idx_map = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx_map[(a,b,c)] = idx
                    idx += 1
        
        quads = []
        for q in combinations(range(n), 4):
            a, b, c, d = q
            quads.append([
                idx_map[(a,b,c)],
                idx_map[(a,b,d)],
                idx_map[(a,c,d)],
                idx_map[(b,c,d)]
            ])
            
        pents = []
        for p in combinations(range(n), 5):
            pents.append([
                idx_map[t] for t in combinations(p, 3)
            ])
            
        return quads, pents, idx_map

    def _evaluate(n, K, u, v, w, quads, pents):
        """
        Evaluate the number of violations for a given basis.
        Returns (red_k4, blue_k5, colors_array)
        """
        colors = _compute_colors(n, K, u, v, w)
        
        # Count red K4
        red_k4 = 0
        for q_idxs in quads:
            if all(colors[i] == 0 for i in q_idxs):
                red_k4 += 1
                
        # Count blue K5
        blue_k5 = 0
        for p_idxs in pents:
            if all(colors[i] == 1 for i in p_idxs):
                blue_k5 += 1
                
        return red_k4, blue_k5, colors

    def _local_search_basis(n, K, u, v, w, quads, pents, time_budget):
        """
        Local search on the basis vectors u, v, w.
        Move: flip a bit u[k,a], v[k,b], or w[k,c].
        """
        t_end = time.time() + time_budget
        u = u.copy()
        v = v.copy()
        w = w.copy()
        
        red_k4, blue_k5, colors = _evaluate(n, K, u, v, w, quads, pents)
        total_v = red_k4 + blue_k5
        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        best_total = total_v
        
        while time.time() < t_end:
            if total_v == 0:
                break
                
            # Pick a random basis vector and a random bit to flip
            # 0: u, 1: v, 2: w
            mat_type = random.randint(0, 2)
            k = random.randint(0, K-1)
            vertex = random.randint(0, n-1)
            
            # Save old state
            old_u, old_v, old_w = u.copy(), v.copy(), w.copy()
            
            # Apply flip
            if mat_type == 0:
                u[k, vertex] ^= 1
            elif mat_type == 1:
                v[k, vertex] ^= 1
            else:
                w[k, vertex] ^= 1
                
            # Evaluate new state
            new_red, new_blue, new_colors = _evaluate(n, K, u, v, w, quads, pents)
            new_total = new_red + new_blue
            
            # Accept if better or random acceptance (simulated annealing style, but simple greedy for now)
            if new_total < best_total:
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                best_total = new_total
                
            # If worse, revert (or keep with some probability? Let's stick to greedy hill climbing for stability first)
            if new_total > total_v:
                u, v, w = old_u, old_v, old_w
            else:
                total_v = new_total
                red_k4, blue_k5 = new_red, new_blue
                colors = new_colors
                
            # If we are stuck, maybe try a random restart of the basis? 
            # Or just keep going.
            
        return best_u, best_v, best_w, best_total

    # Main Logic
    best_n = 0
    best_colors = None
    
    # Target n=35 first? Or n=34?
    # n=34 is known to be possible. n=35 is the goal.
    # Let's try n=34 first to get a valid baseline, then try n=35.
    
    for n_target in [34, 35]:
        if time.time() > _START + _DEADLINE - 20:
            break
            
        quads, pents, idx_map = _build_index_data(n_target)
        
        # Try different ranks K
        found = False
        for K in [3, 4, 5, 6]:
            if found or time.time() > _START + _DEADLINE - 10:
                break
                
            # Initialize random basis
            u = (np.random.rand(K, n_target) > 0.5).astype(np.int8)
            v = (np.random.rand(K, n_target) > 0.5).astype(np.int8)
            w = (np.random.rand(K, n_target) > 0.5).astype(np.int8)
            
            # Run local search
            time_budget = min(30.0, _START + _DEADLINE - 15 - time.time())
            if time_budget <= 5:
                continue
                
            best_u, best_v, best_w, best_v_count = _local_search_basis(
                n_target, K, u, v, w, quads, pents, time_budget
            )
            
            if best_v_count == 0:
                # Valid coloring found!
                colors = _compute_colors(n_target, K, best_u, best_v, best_w)
                best_n = n_target
                best_colors = colors.tolist()
                found = True
                break # Try next n if this was 34, or stop if 35
                
        if n_target == 35 and found:
            break # We found 35, which is the max we care about (or close to it)
            
    if best_colors is None:
        # Fallback: try a simpler local search on colors directly for n=34 if basis failed?
        # Or just return a small valid n.
        # Let's try a quick greedy color search for n=30 as a fallback.
        n_target = 30
        quads, pents, idx_map = _build_index_data(n_target)
        m = n_target * (n_target - 1) * (n_target - 2) // 6
        colors = (np.random.rand(m) < 0.5).astype(np.int8)
        
        # Simple greedy
        t_end = time.time() + 10
        for _ in range(1000):
            if time.time() > t_end: break
            t = random.randint(0, m-1)
            old_c = colors[t]
            new_c = 1 - old_c
            # Check delta? Too slow without precomputed adjacencies.
            # Just flip and evaluate? O(N^3) per eval is too slow.
            # Skip fallback for now, return 0 if nothing found? 
            # The task says "Maximize n". If we return 0, score is 0.
            # We should ensure we at least return a valid small n.
            
            # Let's just return n=9 valid coloring as absolute fallback.
            pass

    if best_colors is None:
        n = 9
        triples = list(combinations(range(n), 3))
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        best_n = n

    return {'n': best_n, 'colors': best_colors}
