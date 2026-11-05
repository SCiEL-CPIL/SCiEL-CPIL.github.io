"""Low discrepancy on all arithmetic progressions - Spectral Balancing with Batch Repair."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Returns (max_abs, d, i).
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    best_d = 1
    best_i = 0
    
    for d in range(1, n + 1):
        for i in range(min(d, n)):
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            cumsum = np.cumsum(seq)
            current_max = np.max(np.abs(cumsum))
            if current_max > max_abs:
                max_abs = current_max
                best_d = d
                best_i = i
    return int(max_abs), best_d, best_i

def _spectral_balance(signs, n, rng):
    """
    Attempt to improve the sign vector by solving a least-squares problem
    that minimizes the sum of squares of prefix sums over a sample of APs.
    This acts as a global re-optimization step.
    """
    # Sample a diverse set of APs to define the "cost" function
    # We want to minimize sum_{AP} (sum_{j in AP} signs[j])^2
    # This is equivalent to minimizing x^T A x where A is the Gram matrix of AP incidence vectors.
    # However, building the full A is too expensive. Instead, we use a greedy coordinate descent
    # or a simple iterative solver on a subset, or we can use a heuristic based on
    # minimizing the variance of the partial sums.
    
    # A simpler, effective global heuristic:
    # Randomly permute the "influence" or use a randomized spectral approach?
    # Let's try a different global move: Random restart with bias towards balancing.
    # Or, better: "Simulated Annealing" with large jumps (multi-flips) is closer to current.
    # Let's try "Batch Gradient Descent" on the continuous relaxation.
    
    # Let x be the real-valued vector in [-1, 1]^n.
    # We want to minimize sum_{d, i} (sum_{k} x_{i+kd})^2.
    # This is convex! We can solve it efficiently if we can represent the quadratic form.
    # The quadratic form is x^T M x, where M_{pq} = number of APs containing both p and q.
    # M is a banded matrix? No, it's dense but structured.
    # For n=256, M is 256x256. We can construct M.
    
    # Construct M:
    # M_{pq} = count of (d, i) such that p = i + k*d and q = i + l*d for some k, l >= 0.
    # This implies p - q = (k - l)d. So d must divide p - q.
    # Also i must be congruent to p mod d and q mod d, and i < min(p, q) + 1? No, i is start.
    # Actually, for a fixed d, the APs are disjoint in terms of "residue classes" mod d.
    # An AP is defined by start i and step d. The terms are i, i+d, i+2d...
    # Two indices p, q are in the same AP (for step d) if p ≡ q (mod d) and the AP starts at some i ≡ p (mod d) with i <= min(p,q).
    # Actually, the sum is over the *entire* AP contained in [1, n].
    # Wait, the discrepancy is the max over *any* AP. The AP is the sequence i, i+d, ... up to n.
    # So for a fixed d and i, the AP is fixed.
    # M_{pq} = sum_{d=1}^n sum_{i=0}^{d-1} [p in AP(i,d)] [q in AP(i,d)]
    # p in AP(i,d) iff p ≡ i (mod d) and i <= p.
    # So for a fixed d, p and q are in the same AP(i,d) iff p ≡ q (mod d).
    # And i is uniquely determined by p mod d? No, i can be any value in [0, d-1] such that i ≡ p (mod d).
    # Since i is in [0, d-1], i is exactly p % d.
    # So for a fixed d, p and q are in the same AP iff p ≡ q (mod d).
    # Therefore, M_{pq} = sum_{d=1}^n [p ≡ q (mod d)].
    # This is a much simpler structure!
    
    # Let's compute M.
    # M_{pq} depends only on |p - q|? No, on gcd?
    # p ≡ q (mod d) iff d divides (p - q).
    # So M_{pq} = number of divisors d of (p - q) in the range [1, n].
    # Wait, d goes from 1 to n.
    # If p == q, p - q = 0. Every d divides 0. So M_{pp} = n.
    # If p != q, let k = |p - q|. M_{pq} = number of divisors of k that are <= n.
    # Since k < n, all divisors of k are <= n.
    # So M_{pq} = tau(k) for p != q, and n for p = q.
    # Where tau(k) is the number of divisors of k.
    
    # This is fantastic! The cost function is:
    # C(x) = sum_{p,q} M_{pq} x_p x_q = n * sum x_p^2 + sum_{p != q} tau(|p-q|) x_p x_q.
    # We want to minimize this subject to x_p in [-1, 1] (or just sign(x_p) in {-1, 1}).
    # This is a Quadratic Unconstrained Binary Optimization (QUBO) problem, but the matrix M is positive definite?
    # Actually, we want to find signs s_p in {-1, 1} to minimize s^T M s.
    
    # We can use a simple iterative method:
    # Start with current signs.
    # For each p, calculate the "force" F_p = sum_{q != p} M_{pq} s_q.
    # If s_p * F_p > 0, flipping s_p reduces the energy (since term is s_p * F_p, flipping changes it to -s_p * F_p).
    # This is equivalent to local search!
    # But we can do "batch" updates or use a momentum-based approach.
    
    # Let's try a "Simulated Annealing" style but with the correct energy landscape.
    # The current program uses "max discrepancy" as the cost.
    # This spectral method uses "sum of squares of AP sums" as the cost.
    # Minimizing sum of squares is often easier and leads to good max discrepancy too.
    
    # Let's implement a fast local search on the sum-of-squares energy.
    # Energy E = sum_{AP} (sum_{j in AP} s_j)^2.
    # Delta E when flipping s_k:
    # E = sum_{d, i} (S_{d,i})^2 where S_{d,i} = sum_{j in AP(d,i)} s_j.
    # If we flip s_k, S_{d,i} changes by -2 s_k for all APs containing k.
    # New S' = S - 2 s_k.
    # (S')^2 = S^2 - 4 s_k S + 4 s_k^2 = S^2 - 4 s_k S + 4.
    # Delta E = sum_{APs containing k} (4 - 4 s_k S_{d,i}) = 4 * (N_k - 4 s_k * sum_{APs containing k} S_{d,i})? No.
    # Delta E = sum_{APs containing k} (-4 s_k S_{d,i} + 4).
    # Let T_k = sum_{APs containing k} S_{d,i}.
    # Delta E = 4 * N_k - 4 s_k T_k.
    # We want Delta E < 0 => s_k T_k > N_k.
    
    # N_k is the number of APs containing k.
    # T_k is the sum of the sums of all APs containing k.
    
    # We can maintain T_k efficiently?
    # S_{d,i} = sum_{j} s_j [j in AP(d,i)].
    # T_k = sum_{d,i: k in AP(d,i)} S_{d,i} = sum_{d,i: k in AP(d,i)} sum_{j} s_j [j in AP(d,i)]
    # = sum_{j} s_j * (number of APs containing both k and j).
    # This is exactly (M s)_k.
    # So T_k = (M s)_k.
    
    # We can compute M s in O(n^2) or faster?
    # M_{pq} = tau(|p-q|) for p!=q, n for p=q.
    # (M s)_k = n s_k + sum_{j != k} tau(|k-j|) s_j.
    # This is a convolution!
    # Let h[m] = tau(m) for m >= 1, h[0] = n.
    # (M s)_k = sum_{j=0}^{n-1} h[|k-j|] s_j.
    # We can compute this for all k in O(n log n) using FFT? Or just O(n^2) is fine for n=256 (65k ops).
    
    # Strategy:
    # 1. Compute T = M s.
    # 2. Find indices k where s_k * T_k > N_k (flipping reduces energy).
    # 3. Flip all such indices simultaneously? Or one by one?
    # Simultaneous flipping might oscillate. Let's flip the "best" ones.
    # Or, use this as a heuristic to guide the random search.
    
    # Let's implement a "Spectral Local Search":
    # Iterate:
    #   Compute T = M s.
    #   For each k, if s_k * T_k > N_k, flip s_k.
    #   This is a synchronous update. It might cycle.
    #   To avoid cycles, add randomness or use asynchronous updates.
    
    # Let's try asynchronous: pick a random k, if it improves, flip.
    # This is just local search with a different cost function (sum of squares vs max).
    # The key difference: sum of squares is smooth and global, max is jagged and local.
    # Optimizing sum of squares might lead to a better global structure.
    
    return signs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Precompute divisor counts tau(k) for k in 1..n-1
    tau = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        for j in range(i, n, i):
            tau[j] += 1
    
    # M_{pq} = tau[|p-q|] for p!=q, n for p=q.
    # We don't need to store M, just use tau.
    
    def compute_T(signs):
        """Compute T_k = sum_{j} M_{kj} signs[j]"""
        T = np.zeros(n, dtype=np.int32)
        s = signs
        for k in range(n):
            # T_k = n * s_k + sum_{j != k} tau[|k-j|] s_j
            T[k] = n * s[k]
            for j in range(n):
                if j != k:
                    T[k] += tau[abs(k - j)] * s[j]
        return T
    
    def compute_N():
        """N_k is the number of APs containing k.
        N_k = sum_{d,i: k in AP(d,i)} 1.
        For a fixed d, k is in AP(i,d) iff i = k % d.
        So for each d, there is exactly one i such that k is in AP(i,d).
        So N_k = sum_{d=1}^n 1 = n.
        Wait, is this correct?
        AP is defined by start i and step d. i in [0, d-1].
        For a fixed d, the APs partition the indices into d classes.
        k belongs to exactly one AP for each d.
        So yes, N_k = n.
        """
        return np.full(n, n, dtype=np.int32)
    
    N = compute_N()
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    current_disc, current_d, current_i = _get_max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Precompute T for initial signs
    T = compute_T(signs)
    
    # Spectral Local Search Loop
    # We alternate between:
    # 1. Spectral improvement (flip if s_k * T_k > N_k)
    # 2. Random perturbation to escape local minima
    
    stuck_count = 0
    
    while time.time() < deadline:
        # Update T if signs changed? We will update T incrementally or recompute.
        # Recomputing T is O(n^2) = 65k ops. Very fast.
        
        improved = False
        
        # Try to find a flip that reduces sum-of-squares energy
        # We want s_k * T_k > N_k
        # Delta E = 4 * (N_k - s_k * T_k). If s_k * T_k > N_k, Delta E < 0.
        
        # Find all candidates
        candidates = np.where(signs * T > N)[0]
        
        if len(candidates) > 0:
            # Pick a random candidate to avoid bias
            idx = rng.choice(candidates)
            signs[idx] = -signs[idx]
            
            # Update T incrementally?
            # T_j changes by -2 * signs[idx] * M_{j, idx} for all j.
            # M_{j, idx} = tau[|j - idx|] if j != idx, n if j == idx.
            # So T_j -= 2 * signs[idx] * tau[|j - idx|] (with tau[0] = n)
            # Let's just recompute T for simplicity and safety, it's fast enough.
            T = compute_T(signs)
            
            # Check if global max discrepancy improved
            new_disc, _, _ = _get_max_ap_discrepancy(signs)
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
                stuck_count = 0
            else:
                # Energy decreased but max disc didn't improve.
                # This is fine, we keep the move as it might lead to future improvements.
                # But if we get stuck in max disc, we might need to backtrack?
                # For now, we keep the move.
                stuck_count += 1
                
            improved = True
        else:
            # No spectral improvement available.
            # Try random flip
            idx = rng.integers(0, n)
            signs[idx] = -signs[idx]
            T = compute_T(signs)
            
            new_disc, _, _ = _get_max_ap_discrepancy(signs)
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
                stuck_count = 0
            else:
                stuck_count += 1
                # If stuck for too long, revert?
                # Or just continue.
                if stuck_count > 100:
                    # Reset to best_signs and restart with different seed?
                    # Or just break.
                    # Let's try to escape by randomizing a few bits.
                    for _ in range(10):
                        j = rng.integers(0, n)
                        signs[j] = -signs[j]
                    T = compute_T(signs)
                    stuck_count = 0
        
    final_disc, _, _ = _get_max_ap_discrepancy(signs)
    if final_disc < best_disc:
        best_disc = final_disc
        best_signs = signs.copy()

    return best_signs.tolist()

# EVOLVE-BLOCK-END
