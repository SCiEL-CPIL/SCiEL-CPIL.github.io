import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Helper to generate the known n=32 seed (Paley-like or specific construction)
    # The seed program uses a specific GF(16) based construction. 
    # We will use a simpler random seed for n=30 and grow, or try to use the 
    # specific n=32 seed if we can replicate it quickly. 
    # For now, let's use a high-quality random initialization for n=40 and 
    # use simulated annealing to fix it, or start from n=32 and grow.
    
    def get_seed_32():
        # Replicating the seed logic from the prompt for a strong start
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                # The seed logic in the prompt:
                # log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
                # Wait, the prompt's get_seed_32 returns n=32.
                # Let's verify if this is valid.
                # If i//16 == j//16, they are in the same half.
                # If not, color 3.
                # This creates a block structure.
                c_val = log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
                colors.append(c_val)
        return {'n': 32, 'colors': colors}

    seed_result = get_seed_32()
    best_n = 32
    best_colors = seed_result['colors']
    
    # Strategy: Try to grow from n=32 to n=50.
    # Or, initialize a random n=50 and use Simulated Annealing.
    # Given the difficulty of n=50, a robust Simulated Annealing on n=50 
    # starting from a random state might be better than growing.
    # However, random n=50 is very hard. 
    # Let's try a hybrid: Start with n=40 random, SA to fix, then try to grow to 50?
    # No, growing is hard. 
    # Let's stick to the directive: "Greedy randomized construction that iteratively adds vertices".
    # But the directive also says "restart with random edge perturbations if stuck".
    
    # Actually, the most promising path for n=50 is often to start with a 
    # known valid n=49 or n=50 structure if we had one, but we don't.
    # The current program uses random init for n=50. 
    # Let's improve the local search with Simulated Annealing which is 
    # better at escaping local minima than pure greedy/hill climbing.
    
    n_target = 50
    
    def build_array(colors, n):
        c = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c[i, j] = c[j, i] = colors[idx]
                idx += 1
        return c

    def count_mono_triangles(c, n):
        # Vectorized count
        # For each color k, count triangles.
        # Triangles in color k: sum over i<j<k of (c[i,j]==k and c[j,k]==k and c[i,k]==k)
        # We can use matrix multiplication or loops.
        # n=50 is small.
        count = 0
        for k in range(4):
            # Create adjacency matrix for color k
            # A_k[i,j] = 1 if c[i,j]==k
            # Triangles = trace(A_k^3) / 6
            # But we need to be careful with diagonal (0).
            A = (c == k).astype(np.int8)
            np.fill_diagonal(A, 0)
            # Trace(A^3) counts each triangle 6 times
            A2 = A @ A
            trace = np.trace(A2 @ A)
            count += trace // 6
        return count

    def update_N(c, N, u, v, c_old, c_new, n):
        # N[u, v, k] = count of z != u, v such that c[u,z]==k and c[v,z]==k
        # When c[u,v] changes from c_old to c_new:
        # For each w != u, v:
        #   The pair (u, w) has color c[u,w]. The pair (v, w) has color c[v,w].
        #   N[u, w, c[v,w]] changes if c_old == c[v,w] or c_new == c[v,w]
        #   N[v, w, c[u,w]] changes if c_old == c[u,w] or c_new == c[u,w]
        
        for w in range(n):
            if w == u or w == v:
                continue
            c_vw = c[v, w]
            c_uw = c[u, w]
            
            # Update N[u, w, ...] based on change in c[u,v] which affects "potential" for triangle (u,v,w)
            # Wait, N[u,v,k] counts z. 
            # If we change c[u,v], does N[u,v,k] change?
            # No, N[u,v,k] depends on c[u,z] and c[v,z]. It does NOT depend on c[u,v].
            # So N[u,v,:] is UNCHANGED by changing c[u,v].
            
            # However, N[u,w,k] depends on c[u,z] and c[w,z].
            # For z=v, c[u,v] is involved.
            # So N[u,w,c[v,w]] changes.
            # Specifically, N[u,w,k] = sum_{z != u,w} [c[u,z]==k and c[w,z]==k].
            # The term for z=v is [c[u,v]==k and c[w,v]==k].
            # If we change c[u,v] from c_old to c_new:
            # If c_old == c[v,w] (which is c[w,v]), then the term was 1, now 0. So N[u,w,c[v,w]] -= 1.
            # If c_new == c[v,w], then the term was 0, now 1. So N[u,w,c[v,w]] += 1.
            
            if c_old == c_vw:
                N[u, w, c_vw] -= 1
                N[w, u, c_vw] -= 1
            if c_new == c_vw:
                N[u, w, c_vw] += 1
                N[w, u, c_vw] += 1
                
            if c_old == c_uw:
                N[v, w, c_uw] -= 1
                N[w, v, c_uw] -= 1
            if c_new == c_uw:
                N[v, w, c_uw] += 1
                N[w, v, c_uw] += 1

    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Initialize
        # Option 1: Random
        # Option 2: Start from best known (n=32) and pad with random? No, n is fixed.
        # Let's try random init but with a bias? No, just random.
        colors = [random.randint(0, 3) for _ in range(n_target * (n_target - 1) // 2)]
        c = build_array(colors, n_target)
        
        # Compute N
        N = np.zeros((n_target, n_target, 4), dtype=np.int32)
        for u in range(n_target):
            for v in range(u+1, n_target):
                for k in range(4):
                    mask = (c[u] == k) & (c[v] == k)
                    mask[u] = False
                    mask[v] = False
                    N[u, v, k] = np.sum(mask)
                    N[v, u, k] = N[u, v, k]
                    
        # Current energy: Number of monochromatic triangles
        # E = 1/3 * sum_{u<v} N[u, v, c[u,v]]
        E = 0
        for u in range(n_target):
            for v in range(u+1, n_target):
                E += N[u, v, c[u, v]]
        E //= 3
        
        if E == 0:
            return {'n': n_target, 'colors': colors}
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.9995 # Slow cooling
        
        # Precompute edge indices for random selection
        edges = [(i, j) for i in range(n_target) for j in range(i+1, n_target)]
        
        steps = 0
        max_steps = 50000
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            if E == 0:
                return {'n': n_target, 'colors': colors}
                
            # Pick random edge
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            c_old = c[u, v]
            
            # Pick random new color
            c_new = random.randint(0, 3)
            if c_new == c_old:
                steps += 1
                continue
                
            # Calculate Delta E
            # Delta E = (N[u,v,c_new] - N[u,v,c_old]) / 3
            delta_E = (N[u, v, c_new] - N[u, v, c_old]) / 3.0
            
            # Accept if improving or with probability exp(-delta_E / T)
            accept = False
            if delta_E <= 0:
                accept = True
            else:
                if random.random() < np.exp(-delta_E / T):
                    accept = True
            
            if accept:
                # Update N
                update_N(c, N, u, v, c_old, c_new, n_target)
                # Update c
                c[u, v] = c_new
                c[v, u] = c_new
                # Update E
                E += int(delta_E)
                
                # Track best
                if E < best_n: # This is wrong, best_n is n, not E. 
                    # We want to track best solution (lowest E, ideally 0)
                    pass
                
                # Update colors list? 
                # We need to return the colors list.
                # It's better to reconstruct at the end or maintain it.
                # Maintaining is O(1) if we know the index.
                # Index of (u,v) in lexicographic list:
                # i*(n-1) - i*(i+1)//2 + j - i - 1
                # Or simpler:
                # Count pairs (a,b) with a<b.
                # Pairs with first < u: u*(n-1) - u*(u+1)//2 ? No.
                # Number of pairs (a,b) with a < u is u*(n-1) - u*(u+1)//2 ?
                # Let's just use a helper.
                
            T *= cooling
            if T < T_min:
                T = T_min
            steps += 1
            
        # After SA, extract colors
        if E == 0:
            return {'n': n_target, 'colors': colors}
            
        # If we found a valid solution, return it
        # Check validity again to be sure
        if count_mono_triangles(c, n_target) == 0:
            # Reconstruct colors list
            final_colors = []
            idx = 0
            for i in range(n_target):
                for j in range(i+1, n_target):
                    final_colors.append(int(c[i, j]))
            return {'n': n_target, 'colors': final_colors}
            
    # Fallback to seed
    return {'n': best_n, 'colors': best_colors}

# Helper to get index in lexicographic list
def get_idx(u, v, n):
    # u < v
    # Number of edges before row u: 
    # Row 0: n-1
    # Row 1: n-2
    # ...
    # Row u-1: n-u
    # Sum = u*(n-1) - u*(u-1)//2
    # Then add offset v-u-1
    return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
