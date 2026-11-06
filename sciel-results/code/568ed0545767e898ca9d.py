import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    N = 24
    # Precompute roots of unity for dense sampling
    # Use a large number of points for initial scan
    num_samples = 2048
    theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    z = np.exp(1j * theta)
    
    # Seed polynomial: Golay-Rudin-Shapiro
    def generate_grs(n):
        p = [1]
        while len(p) < n:
            p = p + p
        return p[:n]
    
    seed_poly = generate_grs(N)
    
    def eval_score(poly):
        # poly is a list of 1s and -1s
        # Compute P(z) for all z in z_samples
        # P(z) = sum c_k z^k
        # Using matrix multiplication or iterative sum
        # Since N is small, simple loop is fine, but vectorized is faster
        # We can use np.polyval or just sum
        # Let's use the property that z^k can be precomputed? No, z changes.
        # But we can compute P(z) for all z at once.
        # z is (M,). poly is (N,).
        # We want (M,) result.
        # P_j = sum_{k=0}^{N-1} c_k z_j^k
        # This is a dot product for each j.
        # We can form a Vandermonde matrix? That's MxN. 2048x24 is small.
        # But constructing it every time is overhead.
        
        # Alternative: use np.dot with powers of z?
        # z_powers = z[:, None] ** np.arange(N) -> (M, N)
        # This is expensive to compute every time.
        
        # Better: iterative evaluation or precomputed powers?
        # Since we only flip one bit at a time in local search, we can update the score incrementally?
        # No, max norm is non-linear.
        
        # Let's just compute it directly.
        # c = np.array(poly)
        # P = np.zeros(M, dtype=complex)
        # for k in range(N):
        #     P += c[k] * (z ** k)
        
        # Vectorized:
        # powers = np.vstack([z**k for k in range(N)]).T # (M, N)
        # This is M*N operations. 2048*24 = ~50k ops. Very fast.
        
        # To avoid re-computing powers, we can precompute them once.
        # But z is fixed.
        # Let's precompute Z_powers outside.
        
        pass # Placeholder

    # Precompute powers of z
    # Z_powers[i, k] = z[i] ** k
    Z_powers = np.zeros((num_samples, N), dtype=complex)
    Z_powers[:, 0] = 1
    for k in range(1, N):
        Z_powers[:, k] = Z_powers[:, k-1] * z
        
    def get_max_norm(poly):
        c = np.array(poly, dtype=complex)
        # P = Z_powers @ c
        P = Z_powers @ c
        abs_P = np.abs(P)
        max_val = np.max(abs_P)
        # Refine the peak
        # Find the index of the max
        idx = np.argmax(abs_P)
        # Golden section search around theta[idx]
        # Interval [theta[idx-1], theta[idx+1]]
        # Handle wrap-around
        t_left = theta[idx-1] if idx > 0 else theta[-1] - 2*np.pi
        t_right = theta[(idx+1) % num_samples] if idx < num_samples - 1 else theta[0] + 2*np.pi
        
        # If wrap-around, adjust
        if t_right <= t_left:
            t_right += 2 * np.pi
            # If idx is 0, t_left is theta[-1]-2pi, t_right is theta[1].
            # Wait, if idx=0, t_left = theta[-1] - 2pi, t_right = theta[1].
            # This is correct.
            
        def f(t):
            zz = np.exp(1j * t)
            # Compute P(zz)
            # P = sum c_k zz^k
            # We can use Horner's method or just sum
            # Since we only evaluate at 1 point, it's fast.
            val = 0.0 + 0.0j
            for k in range(N-1, -1, -1):
                val = val * zz + c[k]
            return abs(val)
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        a = t_left
        b = t_right
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        fc = f(c_pt)
        fd = f(d_pt)
        
        for _ in range(50): # 50 iterations is enough for 1e-9 accuracy on small interval
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = f(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = f(d_pt)
                
        refined_max = max(fc, fd)
        return refined_max

    # Local Search
    def local_search(poly):
        current_poly = list(poly)
        current_score = get_max_norm(current_poly)
        improved = True
        while improved:
            improved = False
            # Try flipping each bit
            for i in range(N):
                # Flip bit i
                old_val = current_poly[i]
                current_poly[i] = -old_val
                new_score = get_max_norm(current_poly)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Continue searching from this new state (first improvement)
                    # Or restart the loop? Restarting is safer for steepest descent.
                    # Let's break and restart the inner loop to find steepest descent?
                    # No, just keep going. First improvement is faster.
                    # But we might get stuck in a local min.
                    # For N=24, first improvement is usually fine.
                    pass
                else:
                    current_poly[i] = old_val
        return current_poly, current_score

    best_score = float('inf')
    best_poly = None
    
    # Start with seed
    seed_poly_list = list(seed_poly)
    # Map to 1/-1
    # GRS usually produces 1s. We need +/- 1.
    # The standard GRS is all 1s in the construction, but the property is about the sequence.
    # Actually, the GRS sequence is defined recursively.
    # Let's just use the generated list.
    # The prompt's seed code:
    # p, q = [1], [1]
    # while len(p) < 24:
    #     p, q = p + q, p + [-c for c in q]
    # return p[:24]
    # This returns a list of 1s and -1s.
    
    seed_poly_list = []
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    seed_poly_list = p[:24]
    
    # Run local search from seed
    poly_s, score_s = local_search(seed_poly_list)
    if score_s < best_score:
        best_score = score_s
        best_poly = poly_s
        
    # Random restarts
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Random init
        rand_poly = list(rng.choice([-1, 1], size=N))
        # Run local search
        poly_r, score_r = local_search(rand_poly)
        if score_r < best_score:
            best_score = score_r
            best_poly = poly_r
            
        # Check if we found the optimum?
        # Theoretical optimum is 1.2583758.
        # If best_score is very close, we can stop.
        if best_score < 1.258376:
            break
            
    # Final polish?
    # The local search should have converged.
    
    return best_poly

# EVOLVE-BLOCK-END
