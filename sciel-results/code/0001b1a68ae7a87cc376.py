import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256
    
    # 1. Precompute all 4-term APs
    # We need the indices for each AP to build the QUBO matrix
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # 2. Construct the QUBO Matrix Q
    # Objective: Minimize sum over all APs of (x0-x1)^2 + (x1-x2)^2 + (x2-x3)^2
    # In terms of x in {0,1}:
    # (x_u - x_v)^2 = x_u + x_v - 2 x_u x_v
    # So E = sum_{AP} [ (x0+x1-2x0x1) + (x1+x2-2x1x2) + (x2+x3-2x2x3) ]
    # This is a quadratic form x^T Q x + b^T x.
    # We can fold the linear terms into Q by adding a constant or just handling them.
    # Standard QUBO is x^T Q x.
    # Let's build Q such that x^T Q x approximates the penalty.
    # Note: x_i^2 = x_i.
    # Term x_u + x_v contributes to linear terms.
    # Term -2 x_u x_v contributes to off-diagonal.
    
    # To use a pure quadratic solver or energy landscape, we can map to spins s_i in {-1, 1}.
    # x_i = (1 + s_i) / 2.
    # (x_u - x_v)^2 = ((1+s_u)/2 - (1+s_v)/2)^2 = (1/4)(s_u - s_v)^2 = (1/4)(s_u^2 - 2s_us_v + s_v^2) = (1/2)(1 - s_u s_v).
    # So minimizing sum (x_u-x_v)^2 is equivalent to maximizing sum s_u s_v.
    # This is an Ising Model / MaxCut-like problem!
    # We want to maximize H = sum_{edges} s_u s_v.
    # The edges are (i0,i1), (i1,i2), (i2,i3) for each AP.
    
    # Let's build the adjacency matrix W for the Ising model.
    # W[u, v] = number of times (u,v) appears as an adjacent pair in an AP.
    W = np.zeros((n, n), dtype=np.float64)
    
    for ap in aps_arr:
        i0, i1, i2, i3 = ap
        # Add edges
        W[i0, i1] += 1
        W[i1, i0] += 1
        W[i1, i2] += 1
        W[i2, i1] += 1
        W[i2, i3] += 1
        W[i3, i2] += 1
        
    # The energy to minimize is E = - sum W[u,v] s_u s_v (plus constants).
    # We will perform Simulated Annealing on the continuous spin vector s.
    # Update rule: s_i <- s_i + noise - gradient.
    # Gradient of E w.r.t s_i is -2 * sum_j W[i,j] s_j.
    # So we want to move s_i in the direction of sum_j W[i,j] s_j.
    
    # Initialize s randomly in {-1, 1} or continuous?
    # Continuous annealing allows escaping local minima better if we project back to {-1,1} at the end.
    # Or we can do discrete SA on s in {-1,1} but using the Ising formulation.
    # The previous program did discrete SA on x. This is structurally similar but the "energy" is different.
    # To make it materially different, let's use a Continuous Annealing (Mean Field) approach.
    
    # Initialize s
    s = np.random.choice([-1, 1], size=n)
    
    def count_mono_from_s(s_vec):
        # Convert to x
        x = (s_vec + 1) // 2
        # Count monochromatic APs
        # An AP is monochromatic if s_i0=s_i1=s_i2=s_i3
        # This is equivalent to (s_i0*s_i1) == 1 AND (s_i1*s_i2) == 1 AND (s_i2*s_i3) == 1
        # Let's just use the x conversion for exact counting to be safe, it's fast enough for checks.
        c = (s_vec + 1) // 2
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
    
    current_score = count_mono_from_s(s)
    best_score = current_score
    best_s = s.copy()
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    # We will iterate. At each step, we calculate the "local field" h_i = sum_j W[i,j] s_j.
    # The probability of flipping s_i depends on delta E = -2 s_i h_i (if we flip, new E = E - 2 s_i h_i - 2 s_i h_i? No.)
    # If s_i flips to -s_i, the change in H = sum W s s is:
    # Old contribution: s_i * h_i
    # New contribution: -s_i * h_i
    # Delta H = -2 s_i h_i
    # Delta E = - Delta H = 2 s_i h_i
    # If Delta E < 0 (i.e., s_i h_i < 0), we accept.
    # This is exactly the same local move as the previous program's delta calculation!
    # The previous program used A[p, :] @ s where A was the co-occurrence matrix.
    # Here W is the adjacency matrix of the "AP graph".
    # Are they the same?
    # Previous: A[i,j] = count of APs containing both i and j.
    # Current: W[i,j] = count of APs where i and j are ADJACENT in the AP.
    # These are different matrices!
    # A[i,j] includes pairs (i0, i2), (i0, i3), (i1, i3) as well.
    # W[i,j] only includes (i0, i1), (i1, i2), (i2, i3).
    # Minimizing sum (x_u-x_v)^2 for adjacent pairs is a RELAXATION of the full monochromatic constraint.
    # It is easier to satisfy adjacent equality than full 4-term equality.
    # However, if we minimize this, we get long runs of same color.
    # Let's verify if this heuristic leads to better solutions.
    # The "energy landscape" is smoother because it only looks at local neighbors in the AP structure, not all pairs.
    
    # Let's run the annealing on the Ising model defined by W.
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with perturbation
            s = best_s.copy()
            # Flip 5%
            num_flips = n // 20
            flip_idx = np.random.choice(n, num_flips, replace=False)
            s[flip_idx] *= -1
            T = 5.0
            current_score = count_mono_from_s(s)
            if current_score < best_score:
                best_score = current_score
                best_s = s.copy()
            continue
            
        # Calculate local fields h = W @ s
        h = W @ s
        
        # Pick random index
        p = np.random.randint(0, n)
        
        # Calculate delta E for flipping s[p]
        # Delta E = 2 * s[p] * h[p]
        delta_E = 2 * s[p] * h[p]
        
        if delta_E < 0:
            s[p] *= -1
            current_score = count_mono_from_s(s) # Update score exactly
            if current_score < best_score:
                best_score = current_score
                best_s = s.copy()
        else:
            prob = np.exp(-delta_E / T)
            if np.random.random() < prob:
                s[p] *= -1
                current_score = count_mono_from_s(s)
                if current_score < best_score:
                    best_score = current_score
                    best_s = s.copy()
                    
        T *= cooling
        
    # Final conversion
    best_x = (best_s + 1) // 2
    return [int(c) for c in best_x]
